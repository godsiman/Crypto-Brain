import streamlit as st
import pandas as pd
import pandas_ta as ta
import ccxt
import numpy as np
import time
import streamlit.components.v1 as components
from datetime import datetime

# ==========================================
# 1. 系統設定與參數
# ==========================================
st.set_page_config(page_title="Crypto God Mode (Binance)", layout="wide")

# 這裡改用交易所的標準代碼 (Symbol)
COINS = {
    "比特幣 (BTC)": "BTC/USDT",
    "以太幣 (ETH)": "ETH/USDT",
    "索拉納 (SOL)": "SOL/USDT",
    "狗狗幣 (DOGE)": "DOGE/USDT",
    "幣安幣 (BNB)": "BNB/USDT",
    "瑞波幣 (XRP)": "XRP/USDT",
    "艾達幣 (ADA)": "ADA/USDT",
    "佩佩蛙 (PEPE)": "PEPE/USDT", # 測試超小幣種
    "柴犬幣 (SHIB)": "SHIB/USDT"
}

PARAMS = {
    'ema_s': 20, 'ema_m': 50, 'ema_l': 200,
    'rsi_len': 14, 
    'bb_len': 20, 'bb_std': 2,
    'atr_len': 14,
    'fib_window': 100 
}

# ==========================================
# 2. 輔助功能：智慧價格顯示
# ==========================================
def format_price(val):
    """
    根據價格大小，自動決定小數位數
    """
    if val is None or val == 0: return "$0.00"
    
    if val < 0.0001:
        return f"${val:.8f}" # 像 PEPE 這種
    elif val < 1.0:
        return f"${val:.4f}" # 像 DOGE, ADA, XRP
    else:
        return f"${val:,.2f}" # 像 BTC, ETH

# ==========================================
# 3. 核心數據抓取 (改用 CCXT 接幣安)
# ==========================================
# 縮短快取時間到 5 秒，因為交易所數據是即時的
@st.cache_data(ttl=5)
def get_binance_data(symbol, timeframe='15m', limit=200):
    try:
        # 初始化交易所 (使用幣安公開 API)
        exchange = ccxt.binance({
            'enableRateLimit': True, # 防止請求太快被鎖
        })
        
        # 抓取 K 線數據 (OHLCV)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        # 轉成 DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # --- 指標計算 (保持原本邏輯) ---
        df['ema20'] = ta.ema(df['close'], length=PARAMS['ema_s'])
        df['ema50'] = ta.ema(df['close'], length=PARAMS['ema_m'])
        df['ema200'] = ta.ema(df['close'], length=PARAMS['ema_l'])
        
        df['rsi'] = ta.rsi(df['close'], length=PARAMS['rsi_len'])
        
        bb = ta.bbands(df['close'], length=PARAMS['bb_len'], std=PARAMS['bb_std'])
        if bb is not None:
            df['bb_u'] = bb.iloc[:, 0]
            df['bb_m'] = bb.iloc[:, 1]
            df['bb_l'] = bb.iloc[:, 2]
            df['bb_w'] = bb.iloc[:, 4]

        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=PARAMS['atr_len'])

        df['struct_h'] = df['high'].rolling(PARAMS['fib_window']).max()
        df['struct_l'] = df['low'].rolling(PARAMS['fib_window']).min()
        
        df['vol_ma'] = df['volume'].rolling(20).mean()

        return df
    except Exception as e:
        print(f"CCXT Error: {e}")
        return pd.DataFrame() # 回傳空表

# ==========================================
# 4. 策略大腦
# ==========================================
def check_candle_pattern(row, prev):
    body = abs(row['close'] - row['open'])
    total_len = row['high'] - row['low']
    if total_len == 0: return False, False, False, False

    upper_wick = row['high'] - max(row['close'], row['open'])
    lower_wick = min(row['close'], row['open']) - row['low']
    
    is_pin_bull = lower_wick > (total_len * 0.6)
    is_pin_bear = upper_wick > (total_len * 0.6)
    
    is_engulf_bull = (row['close'] > row['open']) and (prev['close'] < prev['open']) and (row['close'] > prev['high']) and (row['open'] < prev['low'])
    is_engulf_bear = (row['close'] < row['open']) and (prev['close'] > prev['open']) and (row['close'] < prev['low']) and (row['open'] > prev['high'])
    
    return is_pin_bull, is_pin_bear, is_engulf_bull, is_engulf_bear

def analyze_strategy(df):
    if df is None or df.empty: return None

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    price = curr['close']
    
    # 趨勢
    trend = "盤整 (No Trade)"
    direction = 0
    
    if curr['ema20'] > curr['ema50'] > curr['ema200']:
        trend = "🔥 多頭趨勢 (Long Only)"
        direction = 1
    elif curr['ema20'] < curr['ema50'] < curr['ema200']:
        trend = "❄️ 空頭趨勢 (Short Only)"
        direction = -1

    # 計分
    score = 0
    reasons = []
    
    pin_bull, pin_bear, engulf_bull, engulf_bear = check_candle_pattern(curr, prev)
    
    diff = curr['struct_h'] - curr['struct_l']
    fib_0618_level = curr['struct_h'] - (diff * 0.618)
    fib_05_level = curr['struct_h'] - (diff * 0.5)
    
    if direction == 1:
        # 1. 掃流動性
        recent_low = df['low'].iloc[-10:-1].min()
        if curr['low'] < recent_low and curr['close'] > recent_low:
            score += 1; reasons.append("✅ 掃 Liquidity (下影線洗盤)")
        # 2. RSI
        if prev['rsi'] < 40 and curr['rsi'] > 40:
            score += 1; reasons.append("✅ RSI 低檔反轉")
        # 3. K線
        if pin_bull or engulf_bull:
            score += 1; reasons.append("✅ K線 (PinBar/吞沒)")
        # 4. 布林
        if curr['close'] > curr['bb_m'] and prev['close'] < curr['bb_m']:
            score += 1; reasons.append("✅ 站回布林中軌")
        # 5. 放量
        if curr['volume'] > curr['vol_ma'] * 1.2:
            score += 1; reasons.append("✅ 成交量放大")
        # 6. Fib
        if abs(price - fib_0618_level)/price < 0.005:
            score += 1; reasons.append("✅ 回踩 Fib 0.618")

    elif direction == -1:
        recent_high = df['high'].iloc[-10:-1].max()
        if curr['high'] > recent_high and curr['close'] < recent_high:
            score += 1; reasons.append("✅ 掃 Liquidity (假突破)")
        if prev['rsi'] > 60 and curr['rsi'] < 60:
            score += 1; reasons.append("✅ RSI 高檔回落")
        if pin_bear or engulf_bear:
            score += 1; reasons.append("✅ K線 (倒鎚/吞沒)")
        if curr['close'] < curr['bb_m'] and prev['close'] > curr['bb_m']:
            score += 1; reasons.append("✅ 跌破布林中軌")
        if curr['volume'] > curr['vol_ma'] * 1.2:
            score += 1; reasons.append("✅ 成交量放大")
        if abs(price - fib_0618_level)/price < 0.005:
            score += 1; reasons.append("✅ 反彈至 Fib 0.618")

    # 止盈止損計算
    atr_val = curr['atr']
    sl_price = 0; tp1_price = 0; tp2_price = 0
    
    if direction == 1:
        sl_price = curr['low'] - (2 * atr_val)
        tp1_price = curr['struct_h']
        tp2_price = curr['struct_h'] + (diff * 0.618)
    elif direction == -1:
        sl_price = curr['high'] + (2 * atr_val)
        tp1_price = curr['struct_l']
        tp2_price = curr['struct_l'] - (diff * 0.618)

    return {
        "price": price, "trend": trend, "direction": direction,
        "score": score, "reasons": reasons,
        "sl": sl_price, "tp1": tp1_price, "tp2": tp2_price,
        "rsi": curr['rsi']
    }

# ==========================================
# 5. 前端介面渲染
# ==========================================
st.sidebar.header("🎛️ 幣安實戰控制台")
selected_coin = st.sidebar.radio("監控幣種", list(COINS.keys()))
timeframe = st.sidebar.select_slider("時間級別", options=["5m", "15m", "1h", "4h", "1d"], value="15m")

if st.sidebar.button("🔄 強制刷新 (Binance)"):
    st.cache_data.clear()
    st.rerun()

symbol = COINS[selected_coin]
df = get_binance_data(symbol, timeframe=timeframe)

if df is not None and not df.empty:
    data = analyze_strategy(df)
    
    # 使用新的 format_price 函數來處理顯示
    p_price = format_price(data['price'])
    p_sl = format_price(data['sl'])
    p_tp1 = format_price(data['tp1'])
    p_tp2 = format_price(data['tp2'])

    card_color = "#333"
    signal_text = "⏳ 等待訊號 (Wait)"
    
    if data['score'] >= 3:
        if data['direction'] == 1:
            card_color = "rgba(0, 204, 150, 0.2)"
            signal_text = f"🚀 條件滿足 (Score {data['score']}) - 做多 LONG"
        elif data['direction'] == -1:
            card_color = "rgba(239, 85, 59, 0.2)"
            signal_text = f"🔻 條件滿足 (Score {data['score']}) - 做空 SHORT"
    else:
        signal_text = f"👀 觀察中 (Score {data['score']}/6)"

    reasons_html = ""
    for r in data['reasons']:
        reasons_html += f"<div style='color:#fff; font-size:0.9em; margin-bottom:3px;'>{r}</div>"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background-color: #0e1117; color: #fafafa; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .card {{ background-color: #262730; padding: 20px; border-radius: 12px; border: 1px solid #363945; }}
        .signal-card {{ background-color: {card_color}; border: 1px solid #fff; padding: 20px; border-radius: 12px; }}
        .big-val {{ font-size: 2.2em; font-weight: bold; font-family: monospace; }}
        .label {{ color: #aaa; font-size: 0.9em; margin-bottom: 5px; }}
        .tp-sl-box {{ background: #111; padding: 10px; border-radius: 5px; margin-top: 10px; font-family: monospace; }}
    </style>
    </head>
    <body>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
            <h1>🔥 幣安實戰模板 (Binance API)</h1>
            <div style="text-align:right; color:#888;">{symbol} | {timeframe}</div>
        </div>

        <div class="grid">
            <div class="card">
                <div class="label">Step 1: 趨勢 (Binance)</div>
                <div class="big-val">{p_price}</div>
                <div style="font-size: 1.1em; font-weight:bold; margin-top:10px;">
                    {data['trend']}
                </div>
                <div style="font-size:0.9em; color:#ccc; margin-top:5px;">
                    RSI: <span style="color:{'#ef553b' if data['rsi']>65 else '#00cc96' if data['rsi']<35 else '#ccc'}">{data['rsi']:.1f}</span>
                </div>
            </div>

            <div class="signal-card">
                <div class="label">Step 2: 入場訊號 (需 >= 3 分)</div>
                <div style="font-size: 1.5em; font-weight:bold; margin-bottom:10px;">
                    {signal_text}
                </div>
                {reasons_html if data['reasons'] else "<div style='color:#888;'>等待條件觸發...</div>"}
            </div>

            <div class="card">
                <div class="label">Step 3: 止盈止損 (小數優化版)</div>
                <div class="tp-sl-box">
                    <div style="display:flex; justify-content:space-between; color:#ef553b;">
                        <span>⛔ 止損 (SL):</span>
                        <span>{p_sl}</span>
                    </div>
                </div>
                
                <div class="tp-sl-box">
                    <div style="display:flex; justify-content:space-between; color:#00cc96;">
                        <span>💰 止盈 1:</span>
                        <span>{p_tp1}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; color:#00cc96; margin-top:5px;">
                        <span>💰 止盈 2 (1.618):</span>
                        <span>{p_tp2}</span>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    components.html(html_content, height=550, scrolling=True)

else:
    st.error("⚠️ 無法連線至幣安 (Binance)。")
    st.write("可能原因：Streamlit Cloud 主機位於美國，可能被幣安限制 IP。")
    st.write("💡 建議：此程式碼若在你的本地電腦 (台灣 IP) 執行，將會非常完美且快速。")
