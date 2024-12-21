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

<추가 변수 설명>
목표 : 매수가 진행된 시점 이후에 변화되는 지표값을 반환합니다.

'''

for year in years:
    summary_data = []
    for ticker in tickers:
        for day in days:
            for model in model_types:
                                            
                # 데이터 프레임 Load
                df = pd.read_csv(f'./csv/7_Backtesting/YOY_{year}/{model}/{day}/{ticker}.csv', index_col=0)
                df.columns.values[5:7] = ["Predicted", "action"]
                
                # ============================================================================================
                '''
                a) 1주당 거래 평가입니다.
                '''        
                df['미실현_EPS'] = 0

                buy_price = 0  # Buy 시점의 close 가격
                holding = False  # Buy 후 Holding 중 여부 확인

                # 데이터프레임 순회
                for index, row in df.iterrows():
                    if row['action'] == 'Buy':
                        buy_price = row['Close']
                        holding = True
                    elif row['action'] == 'Holding' and holding:
                        # Buy 후 Holding 중인 동안, close 가격에 따라 미실현_EPS 업데이트
                        df.at[index, '미실현_EPS'] = row['Close'] - buy_price
                    elif row['action'] == 'sell' and holding:
                        # Sell 시점에서 미실현_EPS 업데이트
                        df.at[index, '미실현_EPS'] = row['Close'] - buy_price
                        buy_price = 0
                        holding = False
                    elif row['action'] == 'No action' and holding:
                        # Holding 중이지만 Buy 신호 이후 No action 인 경우, 미실현_EPS은 그대로 0 유지
                        pass
                
                # 'action' 컬럼이 'sell'인 경우에만 Cum_EPS를 0으로 설정하면서 계산
                df['New_Column'] = df['Cum_EPS']
                
                first_sell_index = df[df['action'] == 'sell'].index[0]
                df.at[first_sell_index, 'New_Column'] = 0
                
                # 'action' 열 값이 'sell'인 행을 찾아서 대체
                mask = df['action'] == 'sell'
                df.loc[mask, 'New_Column'] = df['New_Column'].shift(1)
                df.loc[mask, 'Cum_EPS'] = df['Cum_EPS'].shift(1)

                df['미실현_Cum_EPS'] = df['New_Column'] + df['미실현_EPS']
                
                # 컬럼 제거
                df = df.drop('New_Column', axis=1)
                
                # ============================================================================================
                '''
                b) 시드 거래 평가입니다.
                '''     
                df['미실현_Total_Profit'] = 0

                buy_price = 0  # Buy 시점의 close 가격
                holding = False  # Buy 후 Holding 중 여부 확인
                seed_money = 10000
                quantity = 0
                Total_Profit = 0 

                # 데이터프레임 순회
                for index, row in df.iterrows():
                    if row['action'] == 'Buy':
                        buy_price = row['Close']
                        holding = True
                        quantity = seed_money / buy_price
                        
                    elif row['action'] == 'Holding' and holding:
                        # Buy 후 Holding 중인 동안, close 가격에 따라 미실현_EPS 업데이트
                        df.at[index, '미실현_Total_Profit'] = (row['Close'] - buy_price) * quantity
                    elif row['action'] == 'sell' and holding:
                        # Sell 시점에서 미실현_EPS 업데이트
                        df.at[index, '미실현_Total_Profit'] = (row['Close'] - buy_price) * quantity
                        buy_price = 0
                        holding = False
                        quantity = 0
                        Total_Profit = 0 
                    elif row['action'] == 'No action' and holding:
                        # Holding 중이지만 Buy 신호 이후 No action 인 경우, 미실현_EPS은 그대로 0 유지
                        pass
                
                # 'action' 컬럼이 'sell'인 경우에만 Cum_EPS를 0으로 설정하면서 계산
                df['New_Column'] = df['Seed_money']
                
                first_sell_index = df[df['action'] == 'sell'].index[0]
                df.at[first_sell_index, 'New_Column'] = 0
                
                # 'action' 열 값이 'sell'인 행을 찾아서 대체
                mask = df['action'] == 'sell'
                df.loc[mask, 'New_Column'] = df['New_Column'].shift(1)
                df.loc[mask, 'Seed_money'] = df['Seed_money'].shift(1)

                df['미실현_Seed_money'] = df['New_Column'] + df['미실현_Total_Profit']
                
                # 컬럼 제거
                df = df.drop('New_Column', axis=1) 

                # ===========================================================================================
                '''
                수익률 평가입니다.
                ''' 
                # 투자금액은 처음 Buy 시점의 Close 값으로 설정 
                initial_investment = 10000

                # 누적 수익값 계산
                df['Cum_Profit'] = df['Seed_money'] - initial_investment
                
                # 누적 미실현 수익값 계산
                df['미실현_Cum_Profit'] = df['미실현_Seed_money'] - initial_investment
                
                # 누적 수익률 계산
                # 누적 수익률 컬럼 추가
                df['Cum_PER'] = (df['Seed_money'] / initial_investment) * 100 - 100
                    
                # 누적 미실현 수익률 컬럼 추가
                df['미실현_Cum_PER'] = (df['미실현_Seed_money'] / initial_investment) * 100 - 100
                    
                # ============================================================================================                                   
                # Drawdown을 나타내는 컬럼 생성 
                df['Drawdown'] = 0
                df['미실현_Drawdown'] = 0
                
                max_cum_profit = 0
                max_unrealized_cum_profit = 0

                for index, row in df.iterrows():
                
                    # 포트폴리오 내 Drawdown 변동률 입력
                    if row['Seed_money'] > max_cum_profit:
                        max_cum_profit = row['Seed_money']
                    
                    if row['Seed_money'] != 0:
                        df.at[index, 'Drawdown'] = -(max_cum_profit - row['Seed_money'])

                    # 포트폴리오 내 Drawdown 변동률 입력
                    if row['미실현_Seed_money'] > max_unrealized_cum_profit:
                        max_unrealized_cum_profit = row['미실현_Seed_money']
                    
                    if row['미실현_Seed_money'] != 0:
                        df.at[index, '미실현_Drawdown'] = -(max_unrealized_cum_profit - row['미실현_Seed_money'])
                
                # ============================================================================================
                # Drawdown(return)을 나타내는 컬럼 생성 
                df['Drawdown_return'] = 0
                df['미실현_Drawdown_return'] = 0
                
                max_cum_profit_return = 0
                max_unrealized_cum_profit_return = 0
                
                for index, row in df.iterrows():
                
                    # 포트폴리오 내 Drawdown 변동률 입력
                    if row['Cum_PER'] > max_cum_profit_return:
                        max_cum_profit_return = row['Cum_PER']
                    
                    if row['Cum_PER'] != 0:
                        df.at[index, 'Drawdown_return'] = -(max_cum_profit_return - row['Cum_PER'])

                    # 포트폴리오 내 Drawdown 변동률 입력
                    if row['미실현_Cum_PER'] > max_unrealized_cum_profit_return:
                        max_unrealized_cum_profit_return = row['미실현_Cum_PER']
                    
                    if row['미실현_Cum_PER'] != 0:
                        df.at[index, '미실현_Drawdown_return'] = -(max_unrealized_cum_profit_return - row['미실현_Cum_PER'])
                # ============================================================================================                               
                # Holding 기간을 나타내는 컬럼 생성 
                df['Holding_Period'] = df.groupby((df['action'] != 'Holding').cumsum()).cumcount()                    

                if not os.path.exists(f'./csv/8_Backtesting_Final/YOY_{year}/'):
                        os.makedirs(f'./csv/8_Backtesting_Final/YOY_{year}/')
                        
                df.to_csv(f"./csv/8_Backtesting_Final/YOY_{year}/{ticker}_{model}_{day}.csv", encoding='utf-8-sig')
                
                # ============================================================================================
                # 매매 결과 총 정리 
                df['action'] = df['action'].replace('No action', 0)
                df['action'] = df['action'].replace('Buy', 1)
                df['action'] = df['action'].replace('sell', -1)
                
                # 거래 횟수
                no_trade = len(df[df['Total_Profit'] > 0]) + len(df[df['Total_Profit'] < 0])
                
                # 승률
                winning_ratio = len(df[(df['action'] == -1) & (df['Total_Profit'] > 0)]) / no_trade if no_trade > 0 else 0
                
                # 수익 평균, 손실 평균
                profit_average = df[df['EPS'] > 0]['EPS'].mean()
                loss_average = df[df['EPS'] < 0]['EPS'].mean()
                
                # payoff_ratio, profit_factor
                payoff_ratio = profit_average / -loss_average if loss_average < 0 else 0
                profit_factor = -df[df['EPS'] > 0]['EPS'].sum() / df[df['EPS'] < 0]['EPS'].sum()
                
                # Maximum Drawdown (MDD)
                max_drawdown = df['Drawdown'].min()
                max_portfolio_drawdown = df['미실현_Drawdown'].min()
                
                max_drawdown_return = df['Drawdown_return'].min()
                max_portfolio_drawdown_return = df['미실현_Drawdown_return'].min()
                
                # 가장 긴 Holding 기간의 값을 찾기
                max_holding_period = df[df['action'] == 'Holding']['Holding_Period'].max()
                
                # 평균 Holding 기간의 값을 찾기
                mean_holding_period = df[df['action'] == 'Holding']['Holding_Period'].mean()

                # Maximum profit and maximum loss (최대 실현 수익 & 손실금액 및 비율)
                max_profit = df[df['Total_Profit'] != 0]['Total_Profit'].max()
                max_profit_return = df[df['PER'] != 0]['PER'].max()
                
                max_loss = df[df['Total_Profit'] != 0]['Total_Profit'].min()
                max_loss_return = df[df['PER'] != 0]['PER'].min()
                
                # Maximum profit and maximum loss Ratio (최대 누적수익 & 손실금액 및 비율 + 최종 수익)
                max_cum_profit = df[df['Cum_Profit'] != 0]['Cum_Profit'].max()
                max_cum_profit_return = df[df['Cum_PER'] != 0]['Cum_PER'].max()
                
                max_cum_loss = df[df['Cum_Profit'] != 0]['Cum_Profit'].min()
                max_cum_loss_return = df[df['Cum_PER'] != 0]['Cum_PER'].min()
                
                last_cumulative = df['Cum_Profit'].iloc[-1]
                last_Cum_PER = df['Cum_PER'].iloc[-1]
                
                # Maximum Unrealized profit and maximum loss (최대 & 최소 잔고평가 금액 및 비율)
                max_unrealized_cum_profit = df[df['미실현_Seed_money'] != 0]['미실현_Seed_money'].max()
                max_unrealized_cum_profit_return = df[df['미실현_Cum_PER'] != 0]['미실현_Cum_PER'].max()
                
                max_unrealized_cum_loss = df[df['미실현_Seed_money'] != 0]['미실현_Seed_money'].min()
                max_unrealized_cum_loss_return = df[df['미실현_Cum_PER'] != 0]['미실현_Cum_PER'].min()
                
                data_frame_name = f"{ticker}_{model}_{day}"

                summary_data.append([data_frame_name, no_trade, winning_ratio, profit_average, loss_average, payoff_ratio, profit_factor, 
                                        max_drawdown, max_portfolio_drawdown, max_drawdown_return, max_portfolio_drawdown_return, 
                                        max_holding_period, mean_holding_period, 
                                        max_profit, max_profit_return, max_loss, max_loss_return,
                                        max_cum_profit, max_cum_profit_return, max_cum_loss, max_cum_loss_return, 
                                        last_cumulative, last_Cum_PER,
                                        max_unrealized_cum_profit, max_unrealized_cum_profit_return, max_unrealized_cum_loss, max_unrealized_cum_loss_return])
    
        
        
        summary_df = pd.DataFrame(summary_data, columns=['Data_Frame_Name', 'No_Trade', 'Winning_Ratio', 'Profit_Average', 'Loss_Average', 'Payoff_Ratio', 'Profit_Factor',
                                                         'MDD', 'MDD(portfolio)', 'MDD_return', 'MDD_return(portfolio)','Max_Holding_Period', 'Mean_Holding_Period',                                                 'Max_Profit', 'Max_Profit_Return', 'Max_Loss', 'Max_Loss_Return',
                                                         'Max_Cum_Profit', 'Max_Cum_Profit_Return', 'Max_Cum_Loss', 'Max_Cum_Loss_Return', 
                                                         'Last_Cumulative', 'last_Cum_PER',
                                                         'Max_Portfolio_Profit', 'Max_Portfolio_Profit_Return', 'Max_Portfolio_Loss', 'Max_Portfolio_Loss_Return',])
    
    
    if not os.path.exists(f'./csv/result_summary/individual/'):
                    os.makedirs(f'./csv/result_summary/individual/')

    summary_df.to_csv(f"./csv/result_summary/individual/Final_results_Summary_YOY_{year}.csv")