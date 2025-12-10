import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import time
import streamlit.components.v1 as components

# --- 1. Python 後端設定區 ---

# 定義你想看的幣種清單 (可以在這裡自己加)
COINS = {
    "比特幣 (BTC)": "BTC-USD",
    "以太幣 (ETH)": "ETH-USD",
    "索拉納 (SOL)": "SOL-USD",
    "狗狗幣 (DOGE)": "DOGE-USD",
    "幣安幣 (BNB)": "BNB-USD"
}

def get_crypto_data(symbol, period="1d", interval="5m"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        return df
    except Exception as e:
        return pd.DataFrame()

def calculate_strategy(df):
    if df.empty or len(df) < 20:
        return 0, 0, "等待數據...", "neutral", 0, 0, 0

    # 計算 RSI
    df['rsi'] = ta.rsi(df['close'], length=14)
    
    current_price = float(df['close'].iloc[-1])
    
    # 計算今日高低點 (作為文字版的支撐壓力參考)
    day_high = float(df['high'].max())
    day_low = float(df['low'].min())

    # 處理 RSI
    last_rsi = df['rsi'].iloc[-1]
    current_rsi = 50.0 if pd.isna(last_rsi) else float(last_rsi)
    
    # --- 策略邏輯 ---
    prediction = current_price 
    signal = "觀望 Wait"
    bias = "neutral" 

    if current_rsi > 70:
        prediction = current_price * 0.995
        signal = "🔴 過熱! 做空 Short"
        bias = "down"
    elif current_rsi < 30:
        prediction = current_price * 1.005
        signal = "🟢 超賣! 做多 Long"
        bias = "up"
    else:
        sma = df['close'].rolling(20).mean().iloc[-1]
        if not pd.isna(sma) and current_price > sma:
            prediction = current_price * 1.001
            signal = "🌊 趨勢向上 (RSI中性)"
            bias = "up"
        else:
            prediction = current_price * 0.999
            signal = "🌧️ 趨勢向下 (RSI中性)"
            bias = "down"

    return current_price, prediction, signal, bias, current_rsi, day_high, day_low

# --- 2. 介面控制區 ---

st.set_page_config(page_title="極簡戰情室", layout="wide")

# 側邊欄選單
st.sidebar.title("控制台")
selected_name = st.sidebar.radio("選擇幣種", list(COINS.keys()))
symbol = COINS[selected_name]

# 執行運算
df = get_crypto_data(symbol)
price, predict, sig, bias, rsi_val, high, low = calculate_strategy(df)

# --- 3. HTML 顯示區 (新增高低點資訊) ---

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
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; }}
        
        .card {{
            background-color: #262730;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #363945;
            text-align: center;
        }}
        .value {{ font-size: 2.2em; font-weight: bold; font-family: monospace; margin: 10px 0; }}
        .sub-value {{ font-size: 1.1em; color: #888; margin-top: 5px; }}
        .label {{ color: #aaa; font-size: 0.9em; }}
        
        .up {{ color: #00cc96; }}
        .down {{ color: #ef553b; }}
        .neutral {{ color: #bbb; }}
        
        .signal-box {{
            padding: 8px;
            border-radius: 4px;
            font-weight: bold;
            background: #333;
            margin-top: 10px;
        }}
        
        /* 迷你資訊列 */
        .info-row {{
            display: flex;
            justify-content: space-around;
            margin-top: 15px;
            border-top: 1px solid #444;
            padding-top: 10px;
            font-size: 0.85em;
            color: #ccc;
        }}
    </style>
</head>
<body>
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 20px;">
        <h2 style="margin:0;">⚡ {selected_name} 戰情室</h2>
        <div style="color:#888; font-size:0.9em;">RSI 強度: {rsi_val:.1f}</div>
    </div>

    <div class="grid">
        <div class="card">
            <div class="label">目前價格 (Live)</div>
            <div class="value">${price:,.2f}</div>
            <div class="info-row">
                <span>最高: ${high:,.2f}</span>
                <span>最低: ${low:,.2f}</span>
            </div>
        </div>

        <div class="card">
            <div class="label">AI 預測目標</div>
            <div class="value {bias}">${predict:,.2f}</div>
            <div class="signal-box" style="color: {'#00cc96' if 'Long' in sig or '向上' in sig else '#ef553b' if 'Short' in sig or '向下' in sig else '#bbb'}">
                {sig}
            </div>
        </div>
    </div>
    
    <div style="text-align:center; color:#555; font-size:0.8em; margin-top:30px;">
        每 10 秒自動更新 | 數據源: Yahoo Finance
    </div>
</body>
</html>
"""

components.html(html_code, height=450)

time.sleep(10)
st.rerun()
