from common_imports import *

# 이미지 생성용 함수
def plot_candles(pricing, title=None, trend_line = False, volume_bars=False, color_function=None, technicals=None):
    
    def default_color(index, open_price, close_price, low, high):
        return 'b' if open_price[index] > close_price[index] else 'r'
    
    color_function = color_function or default_color
    technicals = technicals or []
    open_price = pricing['Open']
    close_price = pricing['Close']
    low = pricing['Low']
    high = pricing['High']
    oc_min = pd.concat([open_price, close_price], axis=1).min(axis=1)
    oc_max = pd.concat([open_price, close_price], axis=1).max(axis=1)
    
    def plot_trendline(ax, pricing, linewidth=5):
        x = np.arange(len(pricing))
        y = pricing.values
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), 'g--', linewidth = linewidth)
    
    if volume_bars:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3,1]},figsize=(20,10))
    else:
        fig, ax1 = plt.subplots(1, 1)
    if title:
        ax1.set_title(title)
    fig.tight_layout()
    x = np.arange(len(pricing))
    candle_colors = [color_function(i, open_price, close_price, low, high) for i in x]
    candles = ax1.bar(x, oc_max-oc_min, bottom=oc_min, color=candle_colors, linewidth=0)
    lines = ax1.vlines(x , low, high, color=candle_colors, linewidth=1)
    
    # 추세선 생성
    if trend_line:
        plot_trendline(ax1, pricing['Close'])
    
    ax1.xaxis.grid(True)
    ax1.yaxis.grid(True)
    ax1.xaxis.set_tick_params(which='major', length=3.0, direction='in', top='off')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.axis(False)

    for indicator in technicals:
        ax1.plot(x, indicator)
    
    if volume_bars:
        volume = pricing['Volume']
        volume_scale = None
        scaled_volume = volume
        if volume.max() > 1000000:
            volume_scale = 'M'
            scaled_volume = volume / 1000000
        elif volume.max() > 1000:
            volume_scale = 'K'
            scaled_volume = volume / 1000
        ax2.bar(x, scaled_volume, color=candle_colors)
        volume_title = 'Volume'
        if volume_scale:
            volume_title = 'Volume (%s)' % volume_scale
        #ax2.set_title(volume_title)
        ax2.xaxis.grid(True)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        ax2.axis(False)
    return fig


# 이미지 저장용 함수
def save_candlestick_images(stock_data, seq_lens, route_new):
    for seq_len in seq_lens:
        for i in tqdm(range(0, len(stock_data) - max(seq_lens) + 1)):
            if seq_len == max(seq_lens):
                candlestick_data = stock_data.iloc[i:i + seq_len]
            else:
                candlestick_data = stock_data.iloc[i + max(seq_lens) - seq_len:i + max(seq_lens)]
            candlestick_data = candlestick_data.reset_index(drop=True)

            seq_path = os.path.join(route_new, str(seq_len))
            os.makedirs(seq_path, exist_ok=True)

            fig = plot_candles(candlestick_data, trend_line=False, volume_bars=False)
            fig.savefig(os.path.join(seq_path, f'{i}.png'), dpi=150)
            plt.close(fig)


# 기술적 지표 생성 함수
def calculate_indicators(df):
       
    # ADX (Average Directional Movement Index)
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    
    # AROON
    df['AROON_down'], df['AROON_up'] = talib.AROON(df['High'], df['Low'], timeperiod=14)

    # AROONOSC (Aroon Oscillator)
    df['AROONOSC'] = talib.AROONOSC(df['High'], df['Low'], timeperiod=14)

    # BOP (Balance Of Power)
    df['BOP'] = talib.BOP(df['Open'], df['High'], df['Low'], df['Close'])

    # CCI (Commodity Channel Index)
    df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)

    # CMO (Chande Momentum Oscillator)
    df['CMO'] = talib.CMO(df['Close'], timeperiod=14)

    # DX (Directional Movement Index)
    df['DX'] = talib.DX(df['High'], df['Low'], df['Close'], timeperiod=14)

    # MFI (Money Flow Index)
    df['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)

    # MINUS_DI (Minus Directional Indicator)
    df['MINUS_DI'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)

    # MOM (Momentum)
    df['MOM'] = talib.MOM(df['Close'], timeperiod=10)

    # PLUS_DI (Plus Directional Indicator)
    df['PLUS_DI'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)

    # PPO (Percentalibge Price Oscillator)
    df['PPO'] = talib.PPO(df['Close'], fastperiod=12, slowperiod=26, matype=0)

    # ROC (Rate of Change)
    df['ROC'] = talib.ROC(df['Close'], timeperiod=10)

    # ROCR (Rate of Change Ratio)
    df['ROCR'] = talib.ROCR(df['Close'], timeperiod=10)

    # RSI (Relative Strength Index)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)

    # STOCH (Stochastic)
    df['STOCH_slowk'], df['STOCH_slowd'] = talib.STOCH(
        df['High'], df['Low'], df['Close'], 
        fastk_period=5, slowk_period=3, slowk_matype=0, 
        slowd_period=3, slowd_matype=0)

    # STOCHF (Stochastic Fast)
    df['STOCHF_fastk'], df['STOCHF_fastd'] = talib.STOCHF(
        df['High'], df['Low'], df['Close'], 
        fastk_period=5, fastd_period=3, fastd_matype=0)

    # STOCHRSI (Stochastic Relative Strength Index)
    df['STOCHRSI_fastk'], df['STOCHRSI_fastd'] = talib.STOCHRSI(df['Close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)

    # TRIX (1-day Rate of Change of a Triple Smooth EMA)
    df['TRIX'] = talib.TRIX(df['Close'], timeperiod=30)

    # ULTOSC (Ultimate Oscillator)
    df['ULTOSC'] = talib.ULTOSC(df['High'], df['Low'], df['Close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)

    # WILLR (Williams' %R)
    df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
    
    return df


# 라벨 생성 함수
def add_labels(df, num_days = 5):
    df['Signal_origin'] = np.where((df['Close'].shift(-1) - df['Close']) / df['Close'] >= 0.00, 1, 0)
    df['Signal_trend'] = np.where(df['Close'].rolling(window=num_days).mean().shift(-num_days) > df['Close'], 1, 0)
    return df


# 모델링용 원본 데이터 구축 함수
def process_stock_data(stock_code, seq_lens, output_dir):
    file_path = f"./csv/1_origin_data/{stock_code}.csv"
    stock_data = pd.read_csv(file_path)

    stock_data = calculate_indicators(stock_data)
    stock_data = add_labels(stock_data, num_days = 5)

    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    mask = (stock_data['Date'].dt.year == 2011)
    filtered_stock_data = stock_data[mask]
    
    if not filtered_stock_data.empty:
        idx = filtered_stock_data.index[0] - max(seq_lens) + 1
        stock_data = stock_data.iloc[idx:, :]

    stock_data = stock_data[stock_data['Date'].dt.year < 2024]
    stock_data = stock_data.reset_index(drop=True)

    # TSLA 같이 초반에 공백인 경우 0으로 채우기
    stock_data = stock_data.fillna(0)
    
    stock_data.to_csv(f"{output_dir} {stock_code}_update.csv", encoding='utf-8', index=False)