import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import time
import streamlit.components.v1 as components

# --- 1. Python 後端大腦區 (處理數據與策略) ---

def get_crypto_data(symbol="BTC-USD", period="1d", interval="5m"):
    """
    改用 yfinance 抓取數據，解決 Streamlit Cloud 被幣安擋IP的問題
    """
    try:
        # 下載數據
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        
        # yfinance 的欄位通常是多層索引，我們簡化它
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # 重新命名欄位以符合習慣 (Yahoo 的欄位是 Capitalize 的)
        df = df.rename(columns={
            "Open": "open", 
            "High": "high", 
            "Low": "low", 
            "Close": "close", 
            "Volume": "volume"
        })
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def calculate_strategy(df):
    """
    計算 RSI 與 預測邏輯
    """
    if df.empty or len(df) < 20:
        return 0, 0, "等待數據...", "neutral", 0

    # 計算 RSI (14週期)
    df['rsi'] = ta.rsi(df['close'], length=14)
    
    # 取得最新一筆資料
    current_price = float(df['close'].iloc[-1])
    
    # 處理 RSI 為空值的情況
    last_rsi = df['rsi'].iloc[-1]
    if pd.isna(last_rsi):
        current_rsi = 50.0
    else:
        current_rsi = float(last_rsi)
    
    # --- 策略邏輯 ---
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
        # 簡單趨勢跟隨 (SMA 20)
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

# 獲取數據 (使用 BTC-USD)
df = get_crypto_data(symbol="BTC-USD")
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
            <div class="label">BTC-USD 現價 (Yahoo)</div>
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
        數據來源: Yahoo Finance | 策略引擎: Python Pandas-TA | 自動刷新
    </div>

</body>
</html>
"""

components.html(html_code, height=400)

time.sleep(15) # 設定 15 秒刷新一次，避免太頻繁被 Yahoo 擋
st.rerun()
