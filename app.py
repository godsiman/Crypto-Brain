import streamlit as st
import pandas as pd
import pandas_ta as ta
import requests
import time
import streamlit.components.v1 as components

# --- 1. Python 後端大腦區 (處理數據與策略) ---

def get_binance_data(symbol="BTCUSDT", interval="5m", limit=100):
    """從幣安抓取 K 線數據"""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'q_vol', 'num_trades', 't_base', 't_quote', 'ignore'])
        df['close'] = pd.to_numeric(df['close'])
        return df
    except:
        return pd.DataFrame()

def calculate_strategy(df):
    """
    這裡是用 Python 寫的策略！
    我們使用 pandas-ta 庫來計算真正的 RSI 指標。
    """
    if df.empty:
        return 0, 0, "No Data", "neutral", 0

    # 計算 RSI (14週期)
    df['rsi'] = ta.rsi(df['close'], length=14)
    
    current_price = df['close'].iloc[-1]
    
    # 處理資料不足導致 RSI 為 NaN 的情況
    if pd.isna(df['rsi'].iloc[-1]):
        current_rsi = 50.0
    else:
        current_rsi = df['rsi'].iloc[-1]
    
    # --- 策略邏輯 (RSI 逆勢策略) ---
    prediction = current_price 
    signal = "觀望 Wait"
    bias = "neutral" 

    if current_rsi > 70:
        prediction = current_price * 0.995 # 預測跌
        signal = "🔴 過熱! 做空 Short"
        bias = "down"
    elif current_rsi < 30:
        prediction = current_price * 1.005 # 預測漲
        signal = "🟢 超賣! 做多 Long"
        bias = "up"
    else:
        # 簡單趨勢跟隨
        sma = df['close'].rolling(20).mean().iloc[-1]
        if not pd.isna(sma) and current_price > sma:
            prediction = current_price * 1.001
            signal = "🌊 趨勢向上 (RSI中性)"
            bias = "up"
        else:
            prediction = current_price * 0.999
            signal = "🌧️ 趨勢向下 (RSI中性)"
            bias = "down"

    return current_price, prediction, signal, bias, current_rsi

# --- 2. 執行運算 ---

st.set_page_config(page_title="Python戰情室", layout="wide")

df = get_binance_data()
price, predict, sig, bias, rsi_val = calculate_strategy(df)

# --- 3. 前端 HTML 介面區 ---

html_code = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: 'Segoe UI', sans-serif;
            background-color: #0e1117;
            color: #fafafa;
            padding: 20px;
        }}
        .card {{
            background-color: #262730;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #363945;
            margin-bottom: 20px;
            text-align: center;
        }}
        .value {{ font-size: 2.5em; font-weight: bold; font-family: monospace; }}
        .label {{ color: #aaa; font-size: 0.9em; }}
        .up {{ color: #00cc96; }}
        .down {{ color: #ef553b; }}
        .neutral {{ color: #bbb; }}
        
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
        
        .signal-box {{
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
            margin-top: 10px;
            background: #333;
        }}
    </style>
</head>
<body>
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <h1>🐍 Python 智能戰情室</h1>
        <div style="color:#888;">Python 運算中 | RSI: {rsi_val:.2f}</div>
    </div>

    <div class="grid">
        <div class="card">
            <div class="label">Binance 現價</div>
            <div class="value">${price:,.2f}</div>
        </div>

        <div class="card">
            <div class="label">AI 預測價格</div>
            <div class="value {bias}">${predict:,.2f}</div>
            <div class="signal-box" style="color: {'#00cc96' if 'Long' in sig or '向上' in sig else '#ef553b' if 'Short' in sig or '向下' in sig else '#bbb'}">
                {sig}
            </div>
        </div>
    </div>

    <hr style="border-color: #333;">
    
    <div style="color: #666; font-size: 0.8em; text-align: center;">
        數據來源: Binance API | 策略引擎: Python Pandas-TA | 刷新頻率: 10秒
    </div>

</body>
</html>
"""

components.html(html_code, height=400)

time.sleep(10)
st.rerun()
