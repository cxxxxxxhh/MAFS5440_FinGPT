import os
import re
import time
import json
import random
import finnhub
import torch
import gradio as gr
import pandas as pd
import yfinance as yf
from pynvml import *
from peft import PeftModel
from collections import defaultdict
from datetime import date, datetime, timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime



os.environ["FINNHUB_API_KEY"] = ""
finnhub_client = finnhub.Client(api_key=os.environ["FINNHUB_API_KEY"])


def print_gpu_utilization():
    
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")  

# Define the plot_recommendation_trends function
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import finnhub


# Define the plot_recommendation_trends function
def plot_recommendation_trends(symbol, n_months=4):
    data = finnhub_client.recommendation_trends(symbol) 

    # Extract data
    periods = [entry['period'] for entry in data][:n_months]
    buy = [entry['buy'] for entry in data][:n_months]
    hold = [entry['hold'] for entry in data][:n_months]
    sell = [entry['sell'] for entry in data][:n_months]
    strong_buy = [entry['strongBuy'] for entry in data][:n_months]
    strong_sell = [entry['strongSell'] for entry in data][:n_months]

    # Convert dates to month + year format
    periods_formatted = [datetime.strptime(period, '%Y-%m-%d').strftime('%B %Y') for period in periods]

    # Set background color and grid
    plt.figure(figsize=(10, 6))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot the chart
    bar_width = 0.35
    index = np.arange(len(periods))

    bars5 = plt.bar(index, strong_sell, bar_width, label='Strong Sell', color='#8B0000')
    bars4 = plt.bar(index, sell, bar_width, bottom=strong_sell, label='Sell', color='#FF6347')
    bars3 = plt.bar(index, hold, bar_width, bottom=[i+j for i,j in zip(strong_sell, sell)], label='Hold', color='#FFA500')
    bars2 = plt.bar(index, buy, bar_width, bottom=[i+j+k for i,j,k in zip(strong_sell, sell, hold)], label='Buy', color='#008000')
    bars1 = plt.bar(index, strong_buy, bar_width, bottom=[i+j+k+l for i,j,k,l in zip(strong_sell, sell, hold, buy)], label='Strong Buy', color='#006400')

    # Add numbers to each bar
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0,
                 strong_sell[bars1.index(bar)] + sell[bars1.index(bar)] + hold[bars1.index(bar)] + buy[bars1.index(bar)] + height / 2.0,
                 '%d' % int(height), ha='center', va='bottom', color='white')

    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0,
                 strong_sell[bars2.index(bar)] + sell[bars2.index(bar)] + hold[bars2.index(bar)] + height / 2.0,
                 '%d' % int(height), ha='center', va='bottom', color='white')

    for bar in bars3:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0,
                 strong_sell[bars3.index(bar)] + sell[bars3.index(bar)] + height / 2.0,
                 '%d' % int(height), ha='center', va='bottom', color='white')

    for bar in bars4:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0,
                 strong_sell[bars4.index(bar)] + height / 2.0,
                 '%d' % int(height), ha='center', va='bottom', color='white')

    for bar in bars5:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0,
                 height / 2.0,
                 '%d' % int(height), ha='center', va='bottom', color='white')

    # Set axis labels and title
    plt.xlabel('Period', fontsize=12)
    plt.ylabel('#Analysts', fontsize=12)
    plt.title(f'{symbol} Stock Recommendations Trends', fontsize=14)
    plt.xticks(index, periods_formatted)
    
    # Calculate the maximum value among all the bars and set y-axis limit slightly higher than the maximum value
    max_value = max([sum(x) for x in zip(strong_sell, sell, hold, buy, strong_buy)])
    plt.ylim(0, max_value * 1.1)  # Modify y-axis range to be slightly higher than the maximum value
    
    plt.legend()

    # Save the chart as an image file
    image_path = f"{symbol}_recommendation_trends.png"
    plt.savefig(image_path)

    # Output the number of analysts recommending buy, sell, etc. for each month
    output = ""
    for i in range(len(periods)):
        output += f"{periods_formatted[i]}: {strong_buy[i]} Strong Buy, {buy[i]} Buy, {hold[i]} Hold, {sell[i]} Sell, {strong_sell[i]} Strong Sell\n"
    
    return output, image_path

# predict
def predict(ticker, n_months):
    
    print_gpu_utilization()
    
    recommend_output, image_path = plot_recommendation_trends(ticker, n_months)

    return recommend_output, image_path

# Gradio
demo = gr.Interface(
    predict,
    inputs=[
        gr.Textbox(
            label="Ticker",
            value="AXP",
            info="Companies from Dow-30 are recommended"
        ),
        gr.Slider(
            minimum=1,
            maximum=4,
            value=4,
            step=1,
            label="n_months",
            info="Number of months to display in the recommendation trends chart"
        )
    ],
    outputs=[
        gr.Textbox(
            label="Recommendations"
        ),
        gr.Image(
            label="Recommendation Trends Chart"
        )
    ],
    title="FinGPT-Forecaster-Recommendations",
    description="""FinGPT-Forecaster-Recommendations returns market recommendations for the specified company (e.g., strong buy, buy, sell, etc.) and provides a statistical overview of these recommendations over the past few months.
This is just a demo showing what this model is capable of. Results inferred from randomly chosen news can be strongly biased.
For more detailed and customized implementation, refer to our FinGPT project: <https://github.com/AI4Finance-Foundation/FinGPT>
**Disclaimer: Nothing herein is financial advice, and NOT a recommendation to trade real money. Please use common sense and always first consult a professional before trading or investing.**
"""
)

demo.launch(share=True)