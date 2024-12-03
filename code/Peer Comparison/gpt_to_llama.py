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

DATA_DIR = f"./DOW-30_data_5"
os.makedirs(DATA_DIR, exist_ok=True)

finnhub_client = finnhub.Client(api_key="")

client = OpenAI(
    api_key = '',
    base_url=""
    )

# stocks for training
DOW_30 = [
     "AMGN"]#, "AAPL", "BA", "CAT", "CSCO", "CVX", "GS", "HD", "HON",
    #"IBM", "INTC", "JNJ", "KO", "JPM", "MCD", "MMM", "MRK", "MSFT", "NKE",
    #"PG", "TRV", "UNH", "CRM", "VZ", "V", "WBA", "WMT", "DIS", "DOW","AXP"
#]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

'''
SYSTEM_PROMPT = "You are a professional stock financial analyst. Your task is to analyze a target company and its industry peers based on relevant news and basic financials from the past weeks, then provide company analysis and compare companies and determine the best-performing company. " \
    #"Firstly, for analysis part, for each company, your answer format should be as follows: \n ### Analysis of company symbol \n[Financial performance]:1. ...\n[Market trends and news]:1. ...\n[Stock price movement prediction]:...\n"\
    "For comparison part, your anser format should be as follows:\n[Comparison Result (Best Company)]:...\n [Reasons]:1. ...\n[Peer Companies Analysis]:1. ...\n"\
'''

SYSTEM_PROMPT = "You are a professional stock financial analyst. Your task is to analyze a target company and its industry peers based on relevant news and basic financials for the last week, then provide company analysis, compare companies and determine the best-performing company. " \
    "Firstly, for the analysis part, your answer format should be:\n\n ### Analysis of COMPANY_SYMBOL\n\n"\
    "Secondly, for the comparison part, your anser format should be as follows:\n\n[Comparison Result (Best Company)]:...\n\n[Reasons]:1. ...\n\n[Peer Companies Analysis]:1. ...\n"\

def gpt4_to_llama(symbol, with_basics=True):
    
    csv_file = f'{DATA_DIR}/{symbol}_gpt-4.csv' if with_basics else \
                   f'{DATA_DIR}/{symbol}_nobasics_gpt-4.csv'
    
    df = pd.read_csv(csv_file)
    
    #prompts, answers, periods, labels = [], [], [], []
    prompts, answers= [], []
    
    for i, row in df.iterrows():
        
        prompt, answer = row['prompt'], row['answer']
        '''
        res = re.search(r"Then let's assume your prediction for next week \((.*)\) is ((:?up|down) by .*%).", prompt)
        
        period, label = res.group(1), res.group(2)
#         label = label.replace('more than 5', '5+')
        '''
        prompt = re.sub(
            r"Then let's assume your prediction for next week \((.*)\) is (up|down) by ((:?.*)%). Provide a summary analysis to support your prediction. The prediction result need to be inferred from your analysis at the end, and thus not appearing as a foundational factor of your analysis.", 
            f"Then make your prediction of the {symbol} stock price movement for next week. Provide a summary analysis to support your prediction.",
            prompt
        )
        prompt = re.sub(r"^\n+", "", prompt)
        try:
            answer = re.sub(
                r"\[Prediction & Analysis\]:\s*",
                f"[Prediction & Analysis]:\nPrediction: \nAnalysis: ",
                answer
            )
        except Exception:
            #print(symbol, i)
            #print(answer)
            continue
            
        #new_system_prompt = SYSTEM_PROMPT.replace(':\n...', '\nPrediction: ...\nAnalysis: ...')
#         new_system_prompt = SYSTEM_PROMPT.replace(':\n...', '\nPrediction: {Up|Down} by {1-2|2-3|3-4|4-5|5+}%\nAnalysis: ...')
        
        prompt = B_INST + B_SYS + SYSTEM_PROMPT + E_SYS + prompt + E_INST
        print("*************************")
        print(prompt)
        print("*************************")
        print(answer)
        print("*************************")
        prompts.append(prompt)
        answers.append(answer)
        #periods.append(period)
        #labels.append(label)
        
    return {
        "prompt": prompts,
        "answer": answers,
        #"period": periods,
        #"label": labels,
    }


def create_dataset(symbol_list, train_ratio=0.8, with_basics=True):

    train_dataset_list = []
    test_dataset_list = []

    for symbol in symbol_list:

        data_dict = gpt4_to_llama(symbol, with_basics)
#         print(data_dict['prompt'][-1])
#         print(data_dict['answer'][-1])
        symbols = [symbol] * len(data_dict['prompt'])
        data_dict.update({"symbol": symbols})

        dataset = Dataset.from_dict(data_dict)
        train_size = round(train_ratio * len(dataset))

        train_dataset_list.append(dataset.select(range(train_size)))
        test_dataset_list.append(dataset.select(range(train_size, len(dataset))))

    train_dataset = datasets.concatenate_datasets(train_dataset_list)
    test_dataset = datasets.concatenate_datasets(test_dataset_list)

    dataset = datasets.DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    
    return dataset
   


dow30_v3_dataset = create_dataset(DOW_30, 0.9,True)
print(len(dow30_v3_dataset["train"]))
#dow30_v3_dataset.save_to_disk('fingpt-forecaster-dow30-llama_6')