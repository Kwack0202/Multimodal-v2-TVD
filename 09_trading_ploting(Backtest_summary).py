from common_imports import *

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import glob
import os

period_list = ['full_period', 'YOY_2021', 'YOY_2022', 'YOY_2023']

tickers = [
        "AAPL", "MSFT", "NVDA", "GOOG", "AMZN",
    
        "BRK-B", "LLY", "AVGO", "TSLA", "JPM", 
    
        "WMT", "UNH", "V", "XOM", "MA", 
    
        "PG", "COST", "JNJ", "ORCL", "HD", 
    
        "BAC", "KO", "NFLX", "MRK",  "CVX", 
    
        "CRM", "ADBE", "AMD", "PEP", "TMO"
        ]

days = ['1day', '5day']

if not os.path.exists(f'./csv/result_summary/integrate/'):
    os.makedirs(f'./csv/result_summary/integrate/')
    
# Subdirectories
base_path = './csv/4_model_results/'

model_types = []

# 폴더 이름을 읽어와서 리스트에 추가
if os.path.exists(base_path):
    for folder_name in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, folder_name)):
            model_types.append(folder_name)
            
for period in period_list:
    df = pd.read_csv(f"./csv/result_summary/individual/Final_results_Summary_{period}.csv", index_col=0)
    
    df = df[['Data_Frame_Name', 'No_Trade', 'Winning_Ratio', 'Payoff_Ratio', 'Profit_Factor', 'MDD_return', 'last_Cum_PER']]
    
    df.replace([0, np.inf, -np.inf], 1, inplace=True)
    
    split_cols = df['Data_Frame_Name'].str.extract(r'^([^_]+)_(.*)_([^_]+)$')
    split_cols.columns = ['stock', 'model', 'label']

    # 새로운 컬럼을 앞에 추가하고 기존 컬럼 제거
    df = pd.concat([split_cols, df.drop('Data_Frame_Name', axis=1)], axis=1)
    
    # 분석에 사용할 메트릭 컬럼 지정
    metrics = ['No_Trade', 'Winning_Ratio', 'Payoff_Ratio', 'Profit_Factor', 'MDD_return', 'last_Cum_PER']
    
    # 'model'과 'label'별로 그룹화한 후, 메트릭의 평균, 표준편차, 최대값, 최소값 계산
    grouped = df.groupby(['model', 'label'])[metrics].agg(['mean', 'std', 'max', 'min'])
    
    # 결과를 CSV로 저장
    grouped.to_csv(f"./csv/result_summary/integrate/metric_summary_{period}.csv")
    

df = pd.read_csv(f"./csv/result_summary/individual/Final_results_Summary_full_period.csv", index_col=0)

df = df[['Data_Frame_Name', 'No_Trade', 'Winning_Ratio', 'Payoff_Ratio', 'Profit_Factor', 'MDD_return', 'last_Cum_PER']]
    
df.replace([0, np.inf, -np.inf], 1, inplace=True)

split_cols = df['Data_Frame_Name'].str.extract(r'^([^_]+)_(.*)_([^_]+)$')
split_cols.columns = ['stock', 'model', 'label']

# 새로운 컬럼을 앞에 추가하고 기존 컬럼 제거
df = pd.concat([split_cols, df.drop('Data_Frame_Name', axis=1)], axis=1)

# ==========================================================================================================

# 스타일 설정 (선택 사항)
sns.set(style="whitegrid")

# 모델별로 반복
for model in model_types:

    for backtesting in ['Payoff_Ratio', 'Profit_Factor']: 

        if 'MM_to_seq' in model:
            model_name = "Cross-Attention"
        elif 'Seq_to_MM' in model:
            model_name = "Self-Attention"
        else:
            model_name = model
        
        if model_name == "Cross-Attention" and backtesting == 'Payoff_Ratio':
            palette = {
                '1day': 'skyblue',
                '5day': 'darkblue'
                }
        
        elif model_name == "Cross-Attention" and backtesting == 'Profit_Factor':        
            palette = {
                '1day': 'lightgreen',
                '5day': 'green'
                }
        
        elif model_name == "Self-Attention" and backtesting == 'Payoff_Ratio':        
            palette = {
                '1day': '#DA70D6', # 연한 보라
                '5day': '#800080' # 진한 보라
                }
        
        elif model_name == "Self-Attention" and backtesting == 'Profit_Factor':        
            palette = {
                '1day': '#FFDAB9', # 연한 노랑
                '5day': '#FF8C00' # 진한 금색
                }
        else:
            palette = {
                '1day': '#FFB6C1',
                '5day': '#DC143C'
                }
            
        # 해당 모델의 모든 라벨 데이터 필터링
        filtered_df = df[df['model'] == model].reset_index(drop=True)
        
        # 그리드 스펙 설정: KDE 플롯과 박스 플롯을 위한 2행 1열 레이아웃
        fig = plt.figure(figsize=(20, 14))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        
        # 첫 번째 서브플롯: KDE 플롯 (두 라벨 비교)
        ax0 = plt.subplot(gs[0])
        
        for label in days:
            subset = filtered_df[filtered_df['label'] == label]
            sns.kdeplot(
                data=subset,
                x=backtesting,
                fill=True,
                label=label,
                ax=ax0,
                color=palette.get(label, 'gray'),  # 팔레트에 없는 라벨은 회색으로 표시
                alpha=0.6  # 투명도 조절
            )
        
        # 붉은색 수직선 추가 (x=1 위치)
        ax0.axvline(x=1, color='red', linestyle='--', linewidth=1)
        
        ax0.set_title(f'Distribution of {backtesting} {model_name}', fontsize=35)  # 폰트 크기 키움
        ax0.set_ylabel('Density', fontsize=30)  # 폰트 크기 키움
        ax0.legend(title='Label', fontsize=26)  # 범례 폰트 크기 키움
        
        # X축과 Y축 틱 라벨 폰트 크기 조정
        ax0.tick_params(axis='both', which='major', labelsize=24)
        
        # 두 번째 서브플롯: 박스 플롯 (두 라벨 비교, 가로 방향)
        ax1 = plt.subplot(gs[1], sharex=ax0)
        
        sns.boxplot(
            x=backtesting,
            y='label',
            data=filtered_df,
            palette=palette,
            orient='h',
            ax=ax1
        )
        
        # 박스 플롯에도 동일한 붉은색 수직선 추가 (x=1 위치)
        ax1.axvline(x=1, color='red', linestyle='--', linewidth=1)
        
        ax1.set_xlabel(f'{backtesting}', fontsize=30)  # 폰트 크기 키움
        ax1.set_ylabel('Label', fontsize=30)  # 폰트 크기 키움
        
        # X축과 Y축 틱 라벨 폰트 크기 조정
        ax1.tick_params(axis='both', which='major', labelsize=24)
        
        # 레이아웃 조정
        plt.tight_layout()
        
        # 저장 디렉토리 설정 및 생성
        output_directory = './plot/backtesting_summary/full_period/'
        os.makedirs(output_directory, exist_ok=True)
        
        # 파일 저장
        plt.savefig(os.path.join(output_directory, f'{backtesting}_{model}_comparison.png'))
        
        # 플롯 닫기
        plt.close()
