from common_imports import *


import pandas as pd
import matplotlib.pyplot as plt
import os

# Define colors for LSTM and VIT
lstm_colors = ['lightblue', 'royalblue', 'blue', 'navy']  # Example shades of blue
vit_colors = ['lightcoral', 'indianred', 'firebrick', 'darkred']  # Example shades of red

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

            cumulative_profits = {
                'Fusion': Fusion_df['Cum_Profit']
            }
            
            seed_money = 10000
                        
            baseline_close = Fusion_df['Close'][0]     
            
            quantity = seed_money / baseline_close
            
            Fusion_df['Investment_Value'] = Fusion_df['Close'] * quantity 
            Fusion_df['Close_Relative'] = Fusion_df['Investment_Value'] - seed_money
                       
            # 날짜를 인덱스로 설정
            Fusion_df['Date'] = pd.to_datetime(Fusion_df['Date'])
            Fusion_df.set_index('Date', inplace=True)
                
            plt.figure(figsize=(12, 6))
            plt.tight_layout()

            for i, (profit_model, cumulative_profit) in enumerate(cumulative_profits.items()):
                if profit_model == 'Fusion':
                    plt.plot(Fusion_df.index, cumulative_profit, label=profit_model, color='purple', linewidth=3)

            plt.plot(Fusion_df.index, Fusion_df['Close_Relative'], label='Buy & Hold', linestyle='--')

            plt.title(f'Cumulative Profit Comparison: {ticker}', fontsize=20)  # 제목 폰트 크기 키움
            plt.xlabel('Date', fontsize=20)  # x축 폰트 크기 키움
            plt.ylabel('Cumulative Profit', fontsize=20)  # y축 폰트 크기 키움
            plt.legend(fontsize=14)  # 범례 폰트 크기 키움
            plt.grid(True)

            output_directory = f'./plot/Cumulative_plot/full_period/{model}/'
            os.makedirs(output_directory, exist_ok=True)

            plt.savefig(os.path.join(output_directory, f'{ticker}_{day}_cum.png'))
            plt.close()
