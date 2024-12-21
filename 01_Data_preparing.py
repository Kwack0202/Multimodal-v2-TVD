from common_imports import *

# 종목 별 이미지 생성 기간과 동일한 인덱스의 메타 데이터 저장
start_day = '2010-06-01'
end_day = '2024-01-10'

stock_codes = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN",
    
    "BRK-B", "LLY", "AVGO", "TSLA", "JPM", 
    
    "WMT", "UNH", "V", "XOM", "MA", 
    
    "PG", "COST", "JNJ", "ORCL", "HD", 
    
    "BAC", "KO", "NFLX", "MRK",  "CVX", 
    
    "CRM", "ADBE", "AMD", "PEP", "TMO"
    ]

if not os.path.exists('./csv/1_origin_data/'):
    os.makedirs('./csv/1_origin_data/')
    
for stock_code in tqdm(stock_codes):
    stock_data = pd.DataFrame(fdr.DataReader(stock_code, start_day, end_day))
    stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].astype(float)
    stock_data = stock_data.reset_index()
            
    stock_data.to_csv(f"./csv/1_origin_data/{stock_code}.csv", encoding='utf-8', index=False)