from common_imports import *

# ===================================================================================================
# Subdirectories
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


opposite_count = 1
split_num = 753


for model in model_types:
    for ticker in tickers:
        for day in days:
            stock_data = pd.read_csv(f'./csv/1_origin_data/{ticker}.csv').iloc[-split_num:, 0:6].reset_index(drop=True)
            
            model_results_data = pd.read_csv(f'./csv/5_Up_Down_signal/{model}/{day}/{ticker}.csv', index_col=0).reset_index(drop=True)
            
            trading_data = pd.concat([stock_data, model_results_data], axis=1)
                        
            # ===========================================================================================
            # Buy Sell action 생성
            action = "No action"
            counter = 0
            initial_position_set = False

            for i in range(len(trading_data)):
                curr_pos = trading_data.loc[i, 'Predicted']
                if i == 0:
                    prev_pos = 0
                else:
                    prev_pos = trading_data.loc[i-1, 'Predicted']

                if not initial_position_set:
                    if curr_pos == 0:
                        action = "No action"
                    else:
                        action = "Buy"
                        initial_position_set = True
                else:
                    last_action = trading_data.loc[i-1, f'action']

                    if last_action == "sell":
                        if curr_pos == 0:
                            action = "No action"
                            initial_position_set = False
                        else:
                            action = "Buy"
                            counter = 0
                    else:
                        if curr_pos == 1:
                            action = "Holding"
                            counter = 0
                        else:
                            counter += 1
                            if counter == opposite_count:
                                action = "sell"
                                counter = 0
                            else:
                                action = "Holding"
                
                if i == len(trading_data) - 1:
                    action = "sell"
                
                trading_data.loc[i, f'action'] = action

            output_dir = f'./csv/6_Buy_Sell_signal/full_period/{model}/{day}/'
            os.makedirs(output_dir, exist_ok=True) 
            trading_data.to_csv(f'./csv/6_Buy_Sell_signal/full_period/{model}/{day}/{ticker}.csv', index=True)