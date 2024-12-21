from common_imports import *

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


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
            df = pd.read_csv(f'./csv/8_Backtesting_Final/full_period/{ticker}_{model}_{day}.csv')
            
            # 날짜를 Datetime 형식으로 변환
            df['Date'] = pd.to_datetime(df['Date'])
            
            # 시각화
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Y축 0에 선 추가
            ax.axhline(y=0, color='gray', linestyle='--')
            
            # PER 값에 비례한 원 그리기
            marker_size = 30 * abs(df['PER'])
            colors = ['red' if x >= 0 else 'blue' for x in df['PER']]
            ax.scatter(df['Date'], df['PER'], s=marker_size, alpha=0.5, color=colors, label='Sell Signal Return')
            
            ax.set_xlabel('Date', fontsize=20)  # x축 폰트 크기 키움
            ax.set_ylabel('PER', fontsize=20)  # y축 폰트 크기 키움
            ax.set_title('Sell Signal Return Visualization', fontsize=20)  # 제목 폰트 크기 키움
            ax.legend(fontsize=14)  # 범례 폰트 크기 키움
            
            plt.xticks(rotation=45)
            
            # 파일 이름에서 확장자 제거하고 저장
            output_directory = f'./plot/return_visual/full_period/{model}'
            os.makedirs(output_directory, exist_ok=True)
                
            plt.savefig(os.path.join(output_directory, f'{ticker}_{day}.png'))
            plt.close()
