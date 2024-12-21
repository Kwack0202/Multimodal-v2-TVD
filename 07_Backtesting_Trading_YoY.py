from common_imports import *

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
years= ['2021', '2022', '2023']

commission_rate = 0.0005

'''
<변수 설명 부분>

a) 1주당 거래될 때의 전략 평가입니다
EPS : 주당 순이익
PER : 주당 수익률
Cum_EPS : 누적 EPS를 의미

b) 시드 금액을 기준으로 거래될 때의 전략 평가입니다.
Quantity : 최초 시드 기준 구매 수량(소수점 포함)
Total_Profit : 구매 물량을 매도할 때의 수익입니다
Seed_money : 최초 시드를 기준으로 매도 시기에 변화되는 시드값을 반환합니다.
      
'''

for model in model_types:
    for ticker in tickers:
        for day in days:
            for year in years:
                df = pd.read_csv(f"./csv/6_Buy_Sell_signal/YOY_{year}/{model}/{day}/{ticker}.csv", index_col=0)
                
                # 새로운 데이터프레임 생성
                new_data = {
                    'Date': [],
                    'EPS': [],
                    'PER': [],
                    'Cum_EPS': [],
                    'Quantity' : [],
                    'Total_Profit' : [],
                    'Seed_money' : []
                }

                buy_price = None
                cumulative_profit = 0
                cumulative_profit_ratio = 0 
                
                seed_money = 10000
                quantity = 0
                Total_Profit = 0         

                for index, row in df.iterrows():
                    if row[f'action'] == 'Buy':
                        buy_price = row['Close']
                        quantity = seed_money / buy_price
                        
                        new_data['Date'].append(row['Date'])
                        new_data['EPS'].append(0)
                        new_data['Cum_EPS'].append(cumulative_profit)
                        new_data['PER'].append(0)
                        new_data['Quantity'].append(quantity)
                        new_data['Total_Profit'].append(0)
                        new_data['Seed_money'].append(seed_money)
                        
                    elif row[f'action'] == 'sell' and buy_price is not None:
                        if index + 1 < len(df):
                            next_row = df.iloc[index + 1]  # 다음 행을 가져오기
                            # sell_price = next_row['Open']
                            sell_price = row['Close']
                            
                            profit = sell_price - buy_price - (sell_price * commission_rate)
                            cumulative_profit += profit
                                                
                            return_ = profit / buy_price * 100
                            cumulative_profit_ratio += return_
                            
                            Total_Profit = profit * quantity
                            seed_money += profit * quantity
                            
                            new_data['Date'].append(row['Date'])
                            new_data['EPS'].append(profit)
                            new_data['Cum_EPS'].append(cumulative_profit)
                            new_data['PER'].append(return_)
                            new_data['Quantity'].append(0)
                            new_data['Total_Profit'].append(Total_Profit)
                            new_data['Seed_money'].append(seed_money)
                        else:
                            # 다음 행이 없는 경우 해당 행의 Close로 매도
                            sell_price = row['Close']
                            
                            profit = sell_price - buy_price - (sell_price * commission_rate)
                            cumulative_profit += profit
                                                    
                            return_ = profit / buy_price * 100
                            cumulative_profit_ratio += return_
                            
                            Total_Profit = profit * quantity
                            seed_money += profit * quantity
                            
                            new_data['Date'].append(row['Date'])
                            new_data['EPS'].append(profit)
                            new_data['Cum_EPS'].append(cumulative_profit)
                            new_data['PER'].append(return_)
                            new_data['Quantity'].append(0)
                            new_data['Total_Profit'].append(Total_Profit)
                            new_data['Seed_money'].append(seed_money)
                            
                    else:
                        new_data['Date'].append(row['Date'])
                        new_data['EPS'].append(0)
                        new_data['Cum_EPS'].append(cumulative_profit)
                        new_data['PER'].append(0)
                        new_data['Quantity'].append(0)
                        new_data['Total_Profit'].append(0)
                        new_data['Seed_money'].append(seed_money)
                        
                # 새로운 데이터프레임 생성
                new_df = pd.DataFrame(new_data)
    
                # "Date" 열을 기준으로 두 데이터프레임 병합
                merged_df = pd.merge(df, new_df, on='Date', how='outer')
                
                merged_df = merged_df[['Date', 'Open', 'High', 'Low', 'Close', f'Predicted', f'action', 'EPS', 'Cum_EPS', 'PER', 'Quantity', 'Total_Profit', 'Seed_money']]
                
                if not os.path.exists(f'./csv/7_Backtesting/YOY_{year}/{model}/{day}/'):
                    os.makedirs(f'./csv/7_Backtesting/YOY_{year}/{model}/{day}/')
                    
                merged_df.to_csv(f'./csv/7_Backtesting/YOY_{year}/{model}/{day}/{ticker}.csv', index=True)