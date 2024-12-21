from common_imports import *

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# 설정
period_list = ['full_period', 'YOY_2021', 'YOY_2022', 'YOY_2023']

tickers = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN",
    "BRK-B", "LLY", "AVGO", "TSLA", "JPM",
    "WMT", "UNH", "V", "XOM", "MA",
    "PG", "COST", "JNJ", "ORCL", "HD",
    "BAC", "KO", "NFLX", "MRK", "CVX",
    "CRM", "ADBE", "AMD", "PEP", "TMO"
]

days = ['1day', '5day']

# 출력 디렉토리 생성
os.makedirs('./csv/result_summary/integrate/', exist_ok=True)
os.makedirs('./plot/Box_plot/full_period/', exist_ok=True)

# 모델 타입을 고정 (필요에 따라 동적 읽기 가능)
model_types = [
    'benchmark_MM_single_120', 'benchmark_MM_single_20', 'benchmark_MM_single_5',
    'benchmark_MM_single_60', 'benchmark_only_IMG', 'benchmark_only_TA',
    'MM_to_seq_(LSTM_25_512_12)_(ViT_512_12_16_1024)_(MHAL_512_16)_(MLP_2048_512)',
    'Seq_to_MM_(LSTM_25_512_12_512)_(ViT_512_12_16_1024_512)_(MHAL_2048_16)_(MLP_1024_512)'
]

# CSV 파일 경로 설정
csv_file_path = './csv/result_summary/individual/Final_results_Summary_full_period.csv'  # 실제 CSV 파일 경로로 변경

# CSV 파일 읽기
df = pd.read_csv(csv_file_path)

# 'Data_Frame_Name'을 'Stock', 'Model', 'Label'로 분리
split_cols = df['Data_Frame_Name'].str.extract(r'^([^_]+)_(.*)_([^_]+)$')
split_cols.columns = ['Stock', 'Model', 'Label']

# 분리된 컬럼을 원본 데이터프레임과 결합
df = pd.concat([split_cols, df.drop('Data_Frame_Name', axis=1)], axis=1)

# model_types에 있는 모델만 필터링
df = df[df['Model'].isin(model_types)].reset_index(drop=True)

# 모델 이름 매핑 함수 정의
def map_model_display(model):
    if model.startswith('benchmark_MM_single_'):
        # 'benchmark_MM_single_120' -> 'Multimodal_120'
        trailing_number = model.split('_')[-1]
        return f'Multimodal_{trailing_number}'
    elif model.startswith('benchmark_only_'):
        # 'benchmark_only_IMG' -> 'Unimodal_IMG'
        suffix = model.split('_')[-1]
        return f'Unimodal_{suffix}'
    elif model.startswith('MM_to_seq_'):
        return 'Cross-Attention'
    elif model.startswith('Seq_to_MM_'):
        return 'Self-Attention'
    else:
        return model  # 다른 모델은 그대로 유지

# 'Model_Display' 컬럼 생성
df['Model_Display'] = df['Model'].apply(map_model_display)

# Define model groups
group1 = [
    'Multimodal_5', 
    'Multimodal_20', 
    'Multimodal_60', 
    'Multimodal_120', 
    'Cross-Attention'
]

group2 = [
    'Unimodal_TA', 
    'Unimodal_IMG', 
    'Self-Attention'
]

# Define order for each group
order_group1 = group1.copy()
order_group2 = group2.copy()

# 벤치마크 모델들의 색상 정의
group1_benchmark_colors = [
    '#FFB6C1',  # Multimodal_5: 연한 분홍색
    '#FF69B4',  # Multimodal_20: 조금 더 진한 분홍색
    '#FF1493',  # Multimodal_60: 더욱 진한 분홍색
    '#C71585'   # Multimodal_120: 짙은 분홍색
]

group2_benchmark_colors = [
    '#FFB6C1',    # Unimodal_TA: 연한 분홍
    '#DC143C'     # Unimodal_IMG: 진한 분홍
]

# 박스플롯 및 스캐터 플롯 생성 함수 정의 (세로 방향)
def create_and_save_plots(label_df, label, 
                          order1, benchmark_colors1, 
                          order2, benchmark_colors2):
    metrics = [
        ('Payoff_Ratio', 'Payoff Ratio', 'Payoff Ratio', 'payoff_ratio'),
        ('Profit_Factor', 'Profit Factor', 'Profit Factor', 'profit_factor')
    ]
    
    for y_column, y_label, plot_title, filename_suffix in metrics:
        # Filter data where y_column <= 3
        filtered_df = label_df[label_df[y_column] <= 3]
        
        if filtered_df.empty:
            print(f"No data available for {y_label} <= 3 for label: {label}")
            continue
        
        # Define plot groups
        for group_order, benchmark_colors, group_name in [
            (order1, benchmark_colors1, 'Cross-Attention'),
            (order2, benchmark_colors2, 'Self-Attention')
        ]:
            # Assign the additional color based on group and metric
            if group_name == 'Cross-Attention':
                if y_column == 'Payoff_Ratio':
                    additional_color = 'skyblue' if label == '1day' else 'darkblue'
                elif y_column == 'Profit_Factor':
                    additional_color = 'lightgreen' if label == '1day' else 'green'
            elif group_name == 'Self-Attention':
                if y_column == 'Payoff_Ratio':
                    additional_color = '#DA70D6' if label == '1day' else '#800080'  # 연한 보라 / 진한 보라
                elif y_column == 'Profit_Factor':
                    additional_color = '#FFDAB9' if label == '1day' else '#FF8C00'  # 연한 노랑 / 진한 금색
            
            # Complete palette by adding the additional color
            palette = benchmark_colors + [additional_color]
            
            # Further filter the dataframe for the current group
            group_df = filtered_df[filtered_df['Model_Display'].isin(group_order)]
            
            if group_df.empty:
                print(f"No data available for {group_name} in {label} for {y_label}")
                continue
            
            # Adjust figure size based on the number of models in the group
            plt.figure(figsize=(max(20, len(group_order)*0.6), 18))
            
            # Boxplot
            sns.boxplot(
                x='Model_Display', 
                y=y_column, 
                data=group_df, 
                order=group_order, 
                showfliers=False,
                palette=palette
            )
            
            # Stripplot
            sns.stripplot(
                x='Model_Display', 
                y=y_column, 
                data=group_df, 
                order=group_order, 
                color='black', 
                alpha=0.6, 
                jitter=True,
                dodge=True,
                size=8
            )
            
            # Add horizontal line at y=1.0
            plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2)
            
            # Set plot titles and labels
            plt.title(f'{plot_title} by {group_name} for {label}', fontsize=25)
            plt.ylabel(y_label, fontsize=20)  # 폰트 크기를 20으로 증가
            plt.xlabel('Model', fontsize=20)
            plt.xticks(rotation=45, fontsize=20)
            plt.yticks(fontsize=20)  # Y축 눈금 폰트 크기를 20으로 증가
            plt.tight_layout()
            
            # Define plot path with group name
            plot_path = os.path.join(
                './plot/Box_plot/full_period/', 
                f'{filename_suffix}_{label}_{group_name}.png'
            )
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"Saved {y_label} boxplot with scatter for {label} ({group_name}) at {plot_path}")

# 각 라벨별로 박스플롯과 스캐터 플롯 생성
for label in days:
    label_df = df[df['Label'] == label]
    
    if label_df.empty:
        print(f"No data found for label: {label}")
        continue
    
    create_and_save_plots(
        label_df, label, 
        order_group1, group1_benchmark_colors, 
        order_group2, group2_benchmark_colors
    )

print("박스플롯 생성 및 저장이 완료되었습니다.")
