from common_imports import *

stock_codes = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN",
    
    "BRK-B", "LLY", "AVGO", "TSLA", "JPM", 
    
    "WMT", "UNH", "V", "XOM", "MA", 
    
    "PG", "COST", "JNJ", "ORCL", "HD", 
    
    "BAC", "KO", "NFLX", "MRK",  "CVX", 
    
    "CRM", "ADBE", "AMD", "PEP", "TMO"
    ]
days = ['1day', '5day']

# =========================================================================================
# Only TA
for stock_code in stock_codes:
    for day in days:
        # 데이터 로드
        data_paths = {
            '5': f'./csv/3_Multimodal_data/v2_25/{day}/{stock_code}_5_TA.pkl',
            '20': f'./csv/3_Multimodal_data/v2_25/{day}/{stock_code}_20_TA.pkl',
            '60': f'./csv/3_Multimodal_data/v2_25/{day}/{stock_code}_60_TA.pkl',
            '120': f'./csv/3_Multimodal_data/v2_25/{day}/{stock_code}_120_TA.pkl'
        }

        # 데이터를 담을 리스트 초기화
        data_frames = []

        # 각 pkl 파일을 로드하고 컬럼 이름을 변경한 후 리스트에 추가
        for key, path in data_paths.items():
            df = pd.read_pickle(path)
            df = df.rename(columns={
                'TA_X': f'TA_{key}_X'
            })
            data_frames.append(df)

        # 데이터 프레임 병합
        merged_df = pd.concat(data_frames, axis=1)

        # 중복된 Y 컬럼을 제거하고, 하나의 Y 컬럼만 남김
        y_column = merged_df['Y'].iloc[:, 0]
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

        # Y 컬럼을 맨 오른쪽으로 이동
        merged_df = merged_df.drop(columns=['Y'])
        merged_df['Y'] = y_column

        # 병합된 데이터 프레임을 저장
        merged_df.to_pickle(f'./csv/3_Multimodal_data/v2_25/{day}/{stock_code}_all_TA.pkl')

# =========================================================================================
# Only img
for stock_code in stock_codes:
    for day in days:
        # 데이터 로드
        data_paths = {
            '5': f'./csv/3_Multimodal_data/v2_25/{day}/{stock_code}_5_img.pkl',
            '20': f'./csv/3_Multimodal_data/v2_25/{day}/{stock_code}_20_img.pkl',
            '60': f'./csv/3_Multimodal_data/v2_25/{day}/{stock_code}_60_img.pkl',
            '120': f'./csv/3_Multimodal_data/v2_25/{day}/{stock_code}_120_img.pkl'
        }

        # 데이터를 담을 리스트 초기화
        data_frames = []

        # 각 pkl 파일을 로드하고 컬럼 이름을 변경한 후 리스트에 추가
        for key, path in data_paths.items():
            df = pd.read_pickle(path)
            df = df.rename(columns={
                'img_X': f'img_{key}_X'
            })
            data_frames.append(df)

        # 데이터 프레임 병합
        merged_df = pd.concat(data_frames, axis=1)

        # 중복된 Y 컬럼을 제거하고, 하나의 Y 컬럼만 남김
        y_column = merged_df['Y'].iloc[:, 0]
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

        # Y 컬럼을 맨 오른쪽으로 이동
        merged_df = merged_df.drop(columns=['Y'])
        merged_df['Y'] = y_column

        # 병합된 데이터 프레임을 저장
        merged_df.to_pickle(f'./csv/3_Multimodal_data/v2_25/{day}/{stock_code}_all_img.pkl')

# =========================================================================================
# TA + img        
for stock_code in stock_codes:
    for day in days:
        # 데이터 로드
        data_paths = {
            '5': f'./csv/3_Multimodal_data/v2_25/{day}/{stock_code}_5_update.pkl',
            '20': f'./csv/3_Multimodal_data/v2_25/{day}/{stock_code}_20_update.pkl',
            '60': f'./csv/3_Multimodal_data/v2_25/{day}/{stock_code}_60_update.pkl',
            '120': f'./csv/3_Multimodal_data/v2_25/{day}/{stock_code}_120_update.pkl'
        }

        # 데이터를 담을 리스트 초기화
        data_frames = []

        # 각 pkl 파일을 로드하고 컬럼 이름을 변경한 후 리스트에 추가
        for key, path in data_paths.items():
            df = pd.read_pickle(path)
            df = df.rename(columns={
                'TA_X': f'TA_{key}_X',
                'img_X': f'img_{key}_X'
            })
            data_frames.append(df)

        # 데이터 프레임 병합
        merged_df = pd.concat(data_frames, axis=1)

        # 중복된 Y 컬럼을 제거하고, 하나의 Y 컬럼만 남김
        y_column = merged_df['Y'].iloc[:, 0]
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

        # Y 컬럼을 맨 오른쪽으로 이동
        merged_df = merged_df.drop(columns=['Y'])
        merged_df['Y'] = y_column

        # 병합된 데이터 프레임을 저장
        merged_df.to_pickle(f'./csv/3_Multimodal_data/v2_25/{day}/{stock_code}_all_update.pkl')