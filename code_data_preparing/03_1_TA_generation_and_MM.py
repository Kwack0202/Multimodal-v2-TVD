from common_imports import *
from data_generate import *

stock_codes = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN",
    
    "BRK-B", "LLY", "AVGO", "TSLA", "JPM", 
    
    "WMT", "UNH", "V", "XOM", "MA", 
    
    "PG", "COST", "JNJ", "ORCL", "HD", 
    
    "BAC", "KO", "NFLX", "MRK",  "CVX", 
    
    "CRM", "ADBE", "AMD", "PEP", "TMO"
    ]

seq_lens = [5, 20, 60, 120]

days = ['1day', '5day']

output_dir = './csv/2_TA_csv/v2_25/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for stock_code in tqdm(stock_codes):
    process_stock_data(stock_code, seq_lens, output_dir)
    
    data = pd.read_csv(f'./csv/2_TA_csv/v2_25/{stock_code}_update.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    mask = (data['Date'].dt.year == 2011)
    filtered_stock_data = data[mask]

    for time_steps in tqdm(seq_lens):
        idx = filtered_stock_data.index[0] - time_steps + 1
        data_temp = data.iloc[idx:, :]
        data_temp = data_temp.reset_index(drop=True)

        for day in days:
            features = data_temp[['ADX', 'AROON_down', 'AROON_up', 'AROONOSC', 
                                  'BOP', 
                                  'CCI', 'CMO', 
                                  'DX', 
                                  'MFI', 'MINUS_DI', 'MOM', 
                                  'PLUS_DI', 'PPO', 
                                  'ROC', 'ROCR', 'RSI', 
                                  'STOCH_slowk', 'STOCH_slowd', 'STOCHF_fastk', 'STOCHF_fastd', 'STOCHRSI_fastk', 'STOCHRSI_fastd',
                                  'TRIX',
                                  'ULTOSC',
                                  'WILLR']]
            
            if day == '1day':
                target = data_temp['Signal_origin']
            else:
                target = data_temp['Signal_trend']

            scaler = MinMaxScaler()
            features_scaled = scaler.fit_transform(features)

            X = []
            Y = []
            img_paths = []

            for i in range(len(features_scaled) - time_steps + 1):
                X.append(features_scaled[i:i+time_steps])
                Y.append(target[i+time_steps-1])
                img_paths.append(f'./candle_img/{stock_code}/{time_steps}/{i}.png')

            save_folder = f'./csv/3_Multimodal_data/v2_25/{day}/'
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            
            # ==========================================================================
            # Save TA_X and Y
            df_save_TA = pd.DataFrame({
                'TA_X': X,
                'Y': Y
            })
            with open(f'{save_folder}/{stock_code}_{time_steps}_TA.pkl', 'wb') as f_TA:
                pickle.dump(df_save_TA, f_TA)

            # ==========================================================================
            # Save img_X and Y
            df_save_img = pd.DataFrame({
                'img_X': img_paths,
                'Y': Y
            })
            with open(f'{save_folder}/{stock_code}_{time_steps}_img.pkl', 'wb') as f_img:
                pickle.dump(df_save_img, f_img)
            
            # ==========================================================================
            # Save TA_X + img_X and Y      
            df_save = pd.DataFrame({
                'TA_X': X,
                'img_X': img_paths,
                'Y': Y
            })

            with open(f'{save_folder}/{stock_code}_{time_steps}_update.pkl', 'wb') as f:
                pickle.dump(df_save, f)