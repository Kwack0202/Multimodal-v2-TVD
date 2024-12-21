from common_imports import *

import pandas as pd
import matplotlib.pyplot as plt
import os

import matplotlib.pyplot as plt

base_path = './csv/4_model_results/'

model_types = []

# 폴더 이름을 읽어와서 리스트에 추가
if os.path.exists(base_path):
    for folder_name in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, folder_name)):
            model_types.append(folder_name)
            
days = ['1day', '5day']
tickers = [
        "AAPL", "MSFT", "NVDA", "GOOG", "AMZN",
    
        "BRK-B", "LLY", "AVGO", "TSLA", "JPM", 
    
        "WMT", "UNH", "V", "XOM", "MA", 
    
        "PG", "COST", "JNJ", "ORCL", "HD", 
    
        "BAC", "KO", "NFLX", "MRK",  "CVX", 
    
        "CRM", "ADBE", "AMD", "PEP", "TMO"
        ]

for ticker in tickers:
    for day in days:
        for model in model_types:
            Fusion_df = pd.read_csv(f'./csv/8_Backtesting_Final/full_period/{ticker}_{model}_{day}.csv')
                            
            # 날짜를 인덱스로 설정
            Fusion_df['Date'] = pd.to_datetime(Fusion_df['Date'])
            Fusion_df.set_index('Date', inplace=True)

            # 매수 (Buy)와 매도 (sell) 신호에 대한 인덱스 추출
            buy_signals = Fusion_df[Fusion_df['action'] == 'Buy']
            sell_signals = Fusion_df[Fusion_df['action'] == 'sell']

            # 주식 가격과 신호를 시각화
            plt.figure(figsize=(12, 6))
            plt.tight_layout()
            plt.plot(Fusion_df.index, Fusion_df['Close'], label='Close', color='black', alpha = 0.5)
            plt.scatter(buy_signals.index, buy_signals['Close'], label='Buy', marker='^', color='g', lw=2, s = 50)
            plt.scatter(sell_signals.index, sell_signals['Close'], label='Sell', marker='v', color='r', lw=2, s = 50)

            plt.title(f'Buy Sell Signal : {ticker}', fontsize=20)  # 제목 폰트 크기 키움
            plt.xlabel('Date', fontsize=20)  # x축 폰트 크기 키움
            plt.ylabel('Price', fontsize=20)  # y축 폰트 크기 키움
            plt.legend(fontsize=14)  # 범례 폰트 크기 키움
            plt.grid(True)

            output_directory = f'./plot/trading_plot/full_period/{model}/'
            os.makedirs(output_directory, exist_ok=True)
                    
            plt.savefig(os.path.join(output_directory, f'{ticker}_{day}.png'))
            plt.close()