import os
import re
import csv
import math
import time
import json
import random
import finnhub
import datasets
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, datetime, timedelta
from collections import defaultdict
from datasets import Dataset
from openai import OpenAI
from finnhub.client import FinnhubAPIException  # 导入异常类
import gradio as gr

START_DATE = "2023-11-20"
END_DATE = "2024-11-20"

DATA_DIR = f"./DOW-30_data"
os.makedirs(DATA_DIR, exist_ok=True)

finnhub_client = finnhub.Client(api_key="")

client = OpenAI(
    api_key = '',
    base_url=""
    )

# stocks for training
DOW_30 = [
    "UNH", "CRM", "VZ", "V", "WBA", "WMT", "DIS", "DOW","AXP",
    "AMGN", "AAPL", "BA", "CAT", "CSCO", "CVX", "GS", "HD", "HON",
    "IBM", "INTC", "JNJ","KO", "JPM", "MCD", "MMM", "MRK", "MSFT", "NKE","PG", "TRV"]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SYSTEM_PROMPT = "You are a professional stock financial analyst. Your task is to analyze a target company and its industry peers based on relevant news and basic financials from the past weeks, then provide company analysis and compare companies and determine the best-performing company. " \
    "Firstly, for analysis part, for each company, your answer format should be as follows:\n[Financial performance]:1. ...\n[Market trends and news]:1. ...\n[Stock price movement prediction]:...\n"\
    "Secondly, for comparison part, your anser format should be as follows:\n[Comparison Result (Best Company)]:...\n [Reasons]:1. ...\n[Peer Companies Analysis]:1. ...\n"\

def n_weeks_before(date_string, n):
    
    date = datetime.strptime(date_string, "%Y-%m-%d") - timedelta(days=7*n)

    return date.strftime("%Y-%m-%d")


def get_stock_data(stock_symbol, steps):
    try:
        # 下载股票数据
        stock_data = yf.download(stock_symbol, start=steps[0], end=steps[-1], progress=False)
        if stock_data.empty:
            print(f"Warning: No data available for {stock_symbol}")
            return None  # 如果没有数据，返回 None

        dates, prices = [], []
        available_dates = stock_data.index.strftime("%Y-%m-%d").tolist()  # 格式化日期为字符串

        # 提取每个步骤的价格
        for date in steps[:-1]:
            matched = False
            for i, available_date in enumerate(available_dates):
                if available_date >= date:  # 找到最近的日期
                    prices.append(stock_data['Close'].iloc[i])  # 追加收盘价
                    dates.append(datetime.strptime(available_date, "%Y-%m-%d"))  # 转为 datetime
                    matched = True
                    break
            if not matched:  # 如果没有匹配到
                print(f"Warning: No price data available for {stock_symbol} at step {date}")
                prices.append(None)
                dates.append(datetime.strptime(date, "%Y-%m-%d"))

        # 添加最后一天的数据
        dates.append(datetime.strptime(available_dates[-1], "%Y-%m-%d"))
        prices.append(stock_data['Close'].iloc[-1])

        return pd.DataFrame({
            "Start Date": dates[:-1],
            "End Date": dates[1:],
            "Start Price": prices[:-1],
            "End Price": prices[1:]
        })
    except Exception as e:
        print(f"Error downloading data for {stock_symbol}: {e}")
        return None  # 捕获错误并返回 None

def get_news(symbol, data):
    
    news_list = []
    
    for end_date, row in data.iterrows():
        start_date = row['Start Date'].strftime('%Y-%m-%d')
        end_date = row['End Date'].strftime('%Y-%m-%d')
        print(symbol, ': ', start_date, ' - ', end_date)
        #time.sleep(1) # control qpm
        try:
            weekly_news = finnhub_client.company_news(symbol, _from=start_date, to=end_date)
            #print(len(weekly_news))
            weekly_news_delet = weekly_news[:math.ceil(len(weekly_news)*0.05)]
            #print(len(weekly_news_delet))
            weekly_news_delet = [
            {
                "date": datetime.fromtimestamp(n['datetime']).strftime('%Y%m%d%H%M%S') if n['datetime'] > 0 else 'Invalid Date',
                "headline": n['headline'],
                "summary": n['summary'],
            } for n in weekly_news_delet
            ]
            weekly_news_delet.sort(key=lambda x: x['date'])
            news_list.append(json.dumps(weekly_news_delet))
        except FinnhubAPIException as e:
            news_list=[json.dumps([])]
        #print(news_list)
        #news_list_delet = news_list[:3]#math.ceil(len(news_list)*0.1)]
        # 检查长度是否匹配
        #if len(news_list_delet) == 0:  # 若为空，填充默认值
        #    news_list_delet = [json.dumps([])] * 1#math.ceil(len(data)*0.3) # 填充空的 JSON 对象
    #print(len(news_list_delet))
    
    data['News'] = news_list
    
    return data

def get_company_prompt(symbol):
    try:
        profile = finnhub_client.company_profile2(symbol=symbol)
        if not profile:
            raise gr.Error(f"Failed to find company profile for symbol {symbol} from finnhub!")
            
        company_template = "[Company Introduction]:\n\n{name} is a leading entity in the {finnhubIndustry} sector. Incorporated and publicly traded since {ipo}, the company has established its reputation as one of the key players in the market. As of today, {name} has a market capitalization of {marketCapitalization:.2f} in {currency}, with {shareOutstanding:.2f} shares outstanding." \
            "\n\n{name} operates primarily in the {country}, trading under the ticker {ticker} on the {exchange}. As a dominant force in the {finnhubIndustry} space, the company continues to innovate and drive progress within the industry."

        formatted_str = company_template.format(**profile)

    except FinnhubAPIException as e:
        formatted_str = None 
    
    
    return formatted_str


def get_prompt_by_row(symbol, row):

    start_date = row['Start Date'] if isinstance(row['Start Date'], str) else row['Start Date'].strftime('%Y-%m-%d')
    end_date = row['End Date'] if isinstance(row['End Date'], str) else row['End Date'].strftime('%Y-%m-%d')
    #term = 'increased' if row['End Price'] > row['Start Price'] else 'decreased'
    term = 'increased' if row['End Price'].iloc[0] > row['Start Price'].iloc[0] else 'decreased'
    #head = "From {} to {}, {}'s stock price {} from {:.2f} to {:.2f}. Company news during this period are listed below:\n\n".format(
    #    start_date, end_date, symbol, term, row['Start Price'], row['End Price'])
    head = "From {} to {}, {}'s stock price {} from {:.2f} to {:.2f}. Company news during this period are listed below:\n\n".format(
        start_date, end_date, symbol, term, row['Start Price'].iloc[0], row['End Price'].iloc[0])
    
    news = json.loads(row["News"])
    news = ["[Headline]: {}\n[Summary]: {}\n".format(
        n['headline'], n['summary']) for n in news if n['date'][:8] <= end_date.replace('-', '') and \
        not n['summary'].startswith("Looking for stock market analysis and research with proves results?")]

    basics = json.loads(row['Basics'])
    if basics:
        basics = "Some recent basic financials of {}, reported at {}, are presented below:\n\n[Basic Financials]:\n\n".format(
            symbol, basics['period']) + "\n".join(f"{k}: {v}" for k, v in basics.items() if k != 'period')
    else:
        basics = "[Basic Financials]:\n\nNo basic financial reported."
    
    return head, news, basics


def sample_news(news, k=5):
    
    return [news[i] for i in sorted(random.sample(range(len(news)), k))]


def get_current_basics(symbol, curday):

    basic_financials = finnhub_client.company_basic_financials(symbol, 'all')
    if not basic_financials['series']:
        raise gr.Error(f"Failed to find basic financials for symbol {symbol} from finnhub!")
        
    final_basics, basic_list, basic_dict = [], [], defaultdict(dict)
    
    for metric, value_list in basic_financials['series']['quarterly'].items():
        for value in value_list:
            basic_dict[value['period']].update({metric: value['v']})

    for k, v in basic_dict.items():
        v.update({'period': k})
        basic_list.append(v)
        
    basic_list.sort(key=lambda x: x['period'])
    
    for basic in basic_list[::-1]:
        if basic['period'] <= curday:
            break
            
    return basic
    

def get_all_prompts_online(symbol, data, curday, with_basics=True):

    company_prompt = get_company_prompt(symbol)
    if company_prompt is None:
        return None, None
    else:
        prev_rows = []

        for row_idx, row in data.iterrows():
            head, news, _ = get_prompt_by_row(symbol, row)
            prev_rows.append((head, news, None))
            
        prompt = ""
        for i in range(-len(prev_rows), 0):
            prompt += "\n" + prev_rows[i][0]
            sampled_news = sample_news(
                prev_rows[i][1],
                min(5, len(prev_rows[i][1]))
            )
            if sampled_news:
                prompt += "\n".join(sampled_news)
            else:
                prompt += "No relative news reported."
            
        period = "{} to {}".format(curday, n_weeks_before(curday, -1))
        
        if with_basics:
            basics = get_current_basics(symbol, curday)
            basics = "Some recent basic financials of {}, reported at {}, are presented below:\n\n[Basic Financials]:\n\n".format(
                symbol, basics['period']) + "\n".join(f"{k}: {v}" for k, v in basics.items() if k != 'period')
        else:
            basics = "[Basic Financials]:\n\nNo basic financial reported."

        info = company_prompt + '\n' + prompt + '\n' + basics
        prompt = info + f"\n\nBased on all the information before {curday}, let's first analyze the positive developments and potential concerns for {symbol}. Come up with 2-4 most important factors respectively and keep them concise. Most factors should be inferred from company related news. " \
            f"Then make your prediction of the {symbol} stock price movement for next week ({period}). Provide a summary analysis to support your prediction."
            
        return info, prompt


def construct_prompt(ticker, curday, n_weeks, use_basics):

    '''
    try:
        steps = [n_weeks_before(curday, n) for n in range(n_weeks + 1)][::-1]
    except Exception:
        raise gr.Error(f"Invalid date {curday}!")
        
    '''
    steps = [n_weeks_before(curday, n) for n in range(n_weeks + 1)][::-1]
    data = get_stock_data(ticker, steps)
    if data is None:
        return None, None
    else:
        data = get_news(ticker, data)
        data['Basics'] = [json.dumps({})] * len(data)
        # print(data)
        
        info, prompt = get_all_prompts_online(ticker, data, curday, use_basics)
        if info is None:
            return None, None
        else:
            #prompt = B_INST + B_SYS + SYSTEM_PROMPT + E_SYS + prompt + E_INST
            return info, prompt

def get_peer_comparison(ticker, curday, n_weeks=1, use_basics=True):

   
    # 获取特定ticker的同行公司
    peers_lst = finnhub_client.company_peers(ticker)
    # system prompt
    #SYSTEM_PROMPT = "You are an expert financial analyst. Please provide a detailed comparison of the following companies."

    if not peers_lst:
        raise gr.Error(f"Failed to find peer companies for symbol {ticker} from finnhub!")

    # 获取目标公司（如TSLA）的分析
    company_info, company_prompt = construct_prompt(ticker, curday, n_weeks, use_basics)
    
    # 初始化对比分析的prompt
    comparison_prompt = f"Below is the analysis of {ticker}. Please compare this company with its peers and give your analysis on which one is the best based on the following dimensions:\n1. Financial performance\n2. Market trends and news\n3. Stock price movement prediction\n\n{company_prompt}\n\n"

    # 对每个同行公司进行分析，并构建对比prompt
    count = 0
    for peer in peers_lst[1:]:
        if count < 3:
            peer_info, peer_prompt = construct_prompt(peer, curday, n_weeks, use_basics)
            if peer_info is None:
                continue
            comparison_prompt += f"\n\n[Analysis of {peer}]:\n{peer_prompt}"
            count+=1
        else:
            break
        

    # 生成模型输入
    comparison_prompt = B_INST + B_SYS + SYSTEM_PROMPT + E_SYS + comparison_prompt + E_INST
    #comparison_prompt = B_INST + comparison_prompt + E_INST
    #print("=============================")
    #print(comparison_prompt)
    #print("=============================")

    # 调用OpenAI模型进行对比分析
    completion = client.chat.completions.create(
                            model="deepseek-chat",
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": comparison_prompt}
                            ]
                        )
    answer = completion.choices[0].message.content

    return company_info, comparison_prompt, answer

def get_curday():
    return date.today().strftime("%Y-%m-%d")

def generate_date_list_backward(start_date, num_dates, interval_days=7):

    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    date_list = [
        (start_date_obj - timedelta(days=i * interval_days)).strftime("%Y-%m-%d")
        for i in range(num_dates)
    ]
    return date_list



def append_to_csv(filename, input_data, output_data):
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([input_data, output_data])

        
def initialize_csv(filename):
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["prompt", "answer"])


def query_gpt4(symbol_list, min_past_weeks=1, max_past_weeks=3, with_basics=True):

    for symbol in symbol_list:
        print(symbol)

        csv_file = f'{DATA_DIR}/{symbol}_gpt-4.csv' if with_basics else \
                   f'{DATA_DIR}/{symbol}_nobasics_gpt-4.csv'
        
        if not os.path.exists(csv_file):
            initialize_csv(csv_file)
            pre_done = 0
        else:
            df = pd.read_csv(csv_file)
            pre_done = len(df)

        start_date = "2024-11-26"
        num_dates = 30  
        date_list = generate_date_list_backward(start_date, num_dates)
        
        for date in date_list:
            
            print(date)
            company_info, prompt, answer = get_peer_comparison(symbol, date, 1, use_basics=True)
            time.sleep(1)
            #print(prompt)
            print("*************************************************************************")
            print(answer)
            print("**********************************************************")
            append_to_csv(csv_file, prompt, answer)
      
query_gpt4(DOW_30,1,1,False)