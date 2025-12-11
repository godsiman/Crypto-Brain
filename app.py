import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import numpy as np
import time
import streamlit.components.v1 as components

# ==========================================
# 1. 系統設定與參數
# ==========================================
st.set_page_config(page_title="Crypto God Mode", layout="wide")

COINS = {
    "比特幣 (BTC)": "BTC-USD",
    "以太幣 (ETH)": "ETH-USD",
    "索拉納 (SOL)": "SOL-USD",
    "狗狗幣 (DOGE)": "DOGE-USD",
    "幣安幣 (BNB)": "BNB-USD",
    "瑞波幣 (XRP)": "XRP-USD",
    "艾達幣 (ADA)": "ADA-USD"
}

# 依照你的模板設定參數
PARAMS = {
    'ema_s': 20, 'ema_m': 50, 'ema_l': 200,
    'rsi_len': 14, 
    'bb_len': 20, 'bb_std': 2,
    'atr_len': 14,
    'fib_window': 100  # 用來找近期高低點的時間窗口
}

# ==========================================
# 2. 核心數據抓取 (含快取)
# ==========================================
@st.cache_data(ttl=60)
def get_data(symbol, interval='15m', period='5d'):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        
        # 格式整理
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0)
            except: pass
        df.columns = [c.lower() for c in df.columns]
        
        if len(df) < 200: return pd.DataFrame() # 數據不足

        # --- 指標計算 (依照你的模板) ---
        # 1. EMA 趨勢組
        df['ema20'] = ta.ema(df['close'], length=PARAMS['ema_s'])
        df['ema50'] = ta.ema(df['close'], length=PARAMS['ema_m'])
        df['ema200'] = ta.ema(df['close'], length=PARAMS['ema_l'])

        # 2. RSI & 布林帶
        df['rsi'] = ta.rsi(df['close'], length=PARAMS['rsi_len'])
        bb = ta.bbands(df['close'], length=PARAMS['bb_len'], std=PARAMS['bb_std'])
        if bb is not None:
            df['bb_u'] = bb.iloc[:, 0]
            df['bb_m'] = bb.iloc[:, 1]
            df['bb_l'] = bb.iloc[:, 2]
            df['bb_w'] = bb.iloc[:, 4] # 帶寬

        # 3. ATR (止損用)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=PARAMS['atr_len'])

        # 4. Fibonacci / 結構點 (最近100根K線的高低)
        df['struct_h'] = df['high'].rolling(PARAMS['fib_window']).max()
        df['struct_l'] = df['low'].rolling(PARAMS['fib_window']).min()
        
        # 5. 成交量均線 (判斷放量)
        df['vol_ma'] = df['volume'].rolling(20).mean()

        return df
    except Exception as e:
        return pd.DataFrame()

# ==========================================
# 3. 策略大腦 (依照你的 3 步驟邏輯)
# ==========================================
def check_candle_pattern(row, prev):
    """ 辨識 Pin Bar 與 吞沒 """
    body = abs(row['close'] - row['open'])
    upper_wick = row['high'] - max(row['close'], row['open'])
    lower_wick = min(row['close'], row['open']) - row['low']
    total_len = row['high'] - row['low']
    
    is_pin_bull = lower_wick > (total_len * 0.6) # 長下影
    is_pin_bear = upper_wick > (total_len * 0.6) # 長上影
    
    is_engulf_bull = (row['close'] > row['open']) and (prev['close'] < prev['open']) and (row['close'] > prev['high']) and (row['open'] < prev['low'])
    is_engulf_bear = (row['close'] < row['open']) and (prev['close'] > prev['open']) and (row['close'] < prev['low']) and (row['open'] > prev['high'])
    
    return is_pin_bull, is_pin_bear, is_engulf_bull, is_engulf_bear

def analyze_strategy(df):
    if df is None or df.empty: return None

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    price = curr['close']
    
    # --- 步驟 1: 判斷方向 (EMA 嚴格過濾) ---
    trend = "盤整 (No Trade)"
    direction = 0 # 1=多, -1=空, 0=盤整
    
    # 多頭排列: 20 > 50 > 200
    if curr['ema20'] > curr['ema50'] > curr['ema200']:
        trend = "🔥 多頭趨勢 (Long Only)"
        direction = 1
    # 空頭排列: 20 < 50 < 200
    elif curr['ema20'] < curr['ema50'] < curr['ema200']:
        trend = "❄️ 空頭趨勢 (Short Only)"
        direction = -1

    # --- 步驟 2: 找入場點 (計分制) ---
    score = 0
    reasons = []
    
    # K線型態
    pin_bull, pin_bear, engulf_bull, engulf_bear = check_candle_pattern(curr, prev)
    
    # 計算 Fibonacci
    diff = curr['struct_h'] - curr['struct_l']
    fib_0618_level = curr['struct_h'] - (diff * 0.618)
    fib_05_level = curr['struct_h'] - (diff * 0.5)
    
    # 判斷多單條件 (必須是多頭趨勢)
    if direction == 1:
        # 1. 掃流動性 (破前低收回)
        recent_low = df['low'].iloc[-10:-1].min()
        if curr['low'] < recent_low and curr['close'] > recent_low:
            score += 1; reasons.append("✅ 掃 Liquidity (下影線洗盤)")
            
        # 2. RSI 反轉 (不是超賣買，是 <35 回升)
        if prev['rsi'] < 40 and curr['rsi'] > 40: # 稍微放寬到 40 以適應 15m
            score += 1; reasons.append("✅ RSI 低檔反轉")
            
        # 3. K線型態
        if pin_bull or engulf_bull:
            score += 1; reasons.append("✅ K線 (PinBar/吞沒)")
            
        # 4. 布林帶 (回中軌或擠壓站回)
        if curr['close'] > curr['bb_m'] and prev['close'] < curr['bb_m']:
            score += 1; reasons.append("✅ 站回布林中軌")
            
        # 5. 放量
        if curr['volume'] > curr['vol_ma'] * 1.2:
            score += 1; reasons.append("✅ 成交量放大")
            
        # 6. Fib 回踩
        if abs(price - fib_0618_level)/price < 0.005 or abs(price - fib_05_level)/price < 0.005:
            score += 1; reasons.append("✅ 回踩 Fib 0.5/0.618")

    # 判斷空單條件 (必須是空頭趨勢)
    elif direction == -1:
        # 1. 掃上方流動性
        recent_high = df['high'].iloc[-10:-1].max()
        if curr['high'] > recent_high and curr['close'] < recent_high:
            score += 1; reasons.append("✅ 掃 Liquidity (假突破)")
            
        # 2. RSI 回落
        if prev['rsi'] > 60 and curr['rsi'] < 60:
            score += 1; reasons.append("✅ RSI 高檔回落")
            
        # 3. K線
        if pin_bear or engulf_bear:
            score += 1; reasons.append("✅ K線 (倒鎚/吞沒)")
            
        # 4. 布林中軌壓制
        if curr['close'] < curr['bb_m'] and prev['close'] > curr['bb_m']:
            score += 1; reasons.append("✅ 跌破布林中軌")
            
        # 5. 放量
        if curr['volume'] > curr['vol_ma'] * 1.2:
            score += 1; reasons.append("✅ 成交量放大")
            
        # 6. Fib 反彈空點
        if abs(price - fib_0618_level)/price < 0.005 or abs(price - fib_05_level)/price < 0.005:
            score += 1; reasons.append("✅ 反彈至 Fib 0.5/0.618")

    # --- 步驟 3: 計算止盈止損 (ATR) ---
    atr_val = curr['atr']
    sl_price = 0
    tp1_price = 0
    tp2_price = 0
    
    if direction == 1:
        sl_price = curr['low'] - (2 * atr_val) # 2 ATR 下方
        tp1_price = curr['struct_h'] # 前高
        tp2_price = curr['struct_h'] + (diff * 0.618) # Fib 1.618 延伸
    elif direction == -1:
        sl_price = curr['high'] + (2 * atr_val) # 2 ATR 上方
        tp1_price = curr['struct_l'] # 前低
        tp2_price = curr['struct_l'] - (diff * 0.618) # Fib 1.618 延伸

    return {
        "price": price,
        "trend": trend,
        "direction": direction,
        "score": score,
        "reasons": reasons,
        "sl": sl_price,
        "tp1": tp1_price,
        "tp2": tp2_price,
        "rsi": curr['rsi'],
        "vol_burst": curr['volume'] > curr['vol_ma']
    }

# ==========================================
# 4. 前端介面渲染
# ==========================================
st.sidebar.header("🎛️ 模板控制台")
selected_coin = st.sidebar.radio("監控幣種", list(COINS.keys()))
timeframe = st.sidebar.select_slider("時間級別", options=["15m", "1h", "4h", "1d"], value="15m")

if st.sidebar.button("🔄 刷新數據"):
    st.cache_data.clear()
    st.rerun()

symbol = COINS[selected_coin]
df = get_data(symbol, interval=timeframe)

if df is not None and not df.empty:
    data = analyze_strategy(df)
    
    # 決定卡片顏色與訊號
    card_color = "#333"
    signal_text = "⏳ 等待訊號 (Wait)"
    
    # 只有當分數 >= 3 且趨勢正確時，才給訊號
    if data['score'] >= 3:
        if data['direction'] == 1:
            card_color = "rgba(0, 204, 150, 0.2)" # 綠色背景
            signal_text = f"🚀 條件滿足 (Score {data['score']}) - 做多 LONG"
        elif data['direction'] == -1:
            card_color = "rgba(239, 85, 59, 0.2)" # 紅色背景
            signal_text = f"🔻 條件滿足 (Score {data['score']}) - 做空 SHORT"
    else:
        # 分數不足，顯示目前狀況
        signal_text = f"👀 觀察中 (Score {data['score']}/6)"

    # 生成原因列表 HTML
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
            <h1>🔥 加密幣入場模板 (Pro)</h1>
            <div style="text-align:right; color:#888;">{symbol} | {timeframe}</div>
        </div>

        <div class="grid">
            <div class="card">
                <div class="label">Step 1: 趨勢過濾 (EMA20/50/200)</div>
                <div class="big-val">${data['price']:,.2f}</div>
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
                <div class="label">Step 3: 止盈止損計畫 (ATR + Fib)</div>
                <div class="tp-sl-box">
                    <div style="display:flex; justify-content:space-between; color:#ef553b;">
                        <span>⛔ 止損 (SL):</span>
                        <span>${data['sl']:,.2f}</span>
                    </div>
                    <div style="font-size:0.8em; color:#666; text-align:right;">(前高低點 ± 2 ATR)</div>
                </div>
                
                <div class="tp-sl-box">
                    <div style="display:flex; justify-content:space-between; color:#00cc96;">
                        <span>💰 止盈 1 (前高低):</span>
                        <span>${data['tp1']:,.2f}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; color:#00cc96; margin-top:5px;">
                        <span>💰 止盈 2 (Fib 1.618):</span>
                        <span>${data['tp2']:,.2f}</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div style="margin-top:20px; font-size:0.8em; color:#555; text-align:center;">
            💎 策略核心：方向靠 EMA＋結構｜入場靠 Liquidity 反轉 (>=3分)｜止盈 Fib 1.618
        </div>
    </body>
    </html>
    """
    components.html(html_content, height=550, scrolling=True)

else:
    st.error("⚠️ 無法獲取數據，請稍後再試。")
