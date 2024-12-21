from common_imports import *
from data_generate import *

# 경고 메시지 숨기기
plt.rcParams['figure.max_open_warning'] = 0

# 이미지 생성용 파라미터 =================================================
seq_lens = [120, 60, 20, 5]  # 차트 이미지에 포함되는 시계열 인덱스 리스트
window_len = 1  # 차트 이미지 윈도우 이동 단위

# Main processing loop
stock_codes = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN",
    
    "BRK-B", "LLY", "AVGO", "TSLA", "JPM", 
    
    "WMT", "UNH", "V", "XOM", "MA", 
    
    "PG", "COST", "JNJ", "ORCL", "HD", 
    
    "BAC", "KO", "NFLX", "MRK",  "CVX", 
    
    "CRM", "ADBE", "AMD", "PEP", "TMO"
    ]

for stock_code in tqdm(stock_codes):
    stock_data = pd.read_csv(f"./csv/1_origin_data/{stock_code}.csv")
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data = stock_data[stock_data['Date'] <= '2024']

    mask = (stock_data['Date'].dt.year == 2011)
    filtered_stock_data = stock_data[mask]

    if not filtered_stock_data.empty:
        idx = filtered_stock_data.index[0] - max(seq_lens) + 1
        stock_data = stock_data.iloc[idx:].reset_index(drop=True)

    route_new = os.path.join("./candle_img", stock_code)
    print(f"\n캔들스틱 차트 이미지 생성 : [ {stock_code} ]")

    save_candlestick_images(stock_data, seq_lens, route_new)