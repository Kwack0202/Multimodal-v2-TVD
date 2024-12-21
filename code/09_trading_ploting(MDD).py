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
            # 날짜 순으로 정렬 (필요한 경우)
            Fusion_df = Fusion_df.sort_values('Date')

            # 시각화
            plt.figure(figsize=(12, 6))
            plt.plot(Fusion_df['Date'], Fusion_df['Drawdown_return'], label='Drawdown', linewidth=3, color='darkblue')
            plt.title(f'Drawdown : {ticker}', fontsize=20)  # 제목 폰트 크기 키움
            plt.xlabel('Date', fontsize=20)  # x축 폰트 크기 키움
            plt.ylabel('Drawdown', fontsize=20)  # y축 폰트 크기 키움
            plt.legend(fontsize=14)  # 범례 폰트 크기 키움
            plt.grid(True)
            
            output_directory = f'./plot/Drawdown/full_period/{model}/'
            os.makedirs(output_directory, exist_ok=True)
                    
            plt.savefig(os.path.join(output_directory, f'{ticker}_{day}.png'))
            plt.close()
