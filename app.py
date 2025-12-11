import streamlit as st
import pandas as pd
import pandas_ta as ta
import ccxt
import concurrent.futures
import streamlit.components.v1 as components

# ==========================================
# 1. 系統設定與參數
# ==========================================
st.set_page_config(page_title="Crypto God Mode (Stable)", layout="wide")

# 定義基礎幣種清單
BASE_COINS = {
    "BTC": "BTC/USD",
    "ETH": "ETH/USD",
    "SOL": "SOL/USD",
    "DOGE": "DOGE/USD",
    "XRP": "XRP/USD",
    "ADA": "ADA/USD",
    "PEPE": "PEPE/USD",
    "SHIB": "SHIB/USD"
}

PARAMS = {
    'ema_s': 20, 'ema_m': 50, 'ema_l': 200,
    'rsi_len': 14, 
    'bb_len': 20, 'bb_std': 2,
    'atr_len': 14,
    'fib_window': 100 
}

# ==========================================
# 2. 核心：數據抓取與指標計算 (平行運算版)
# ==========================================
def format_price(val):
    if val is None or val == 0: return "$0.00"
    if val < 0.0001: return f"${val:.8f}"
    elif val < 10.0: return f"${val:.4f}"
    else: return f"${val:,.2f}"

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

def analyze_logic(df):
    if df is None or df.empty: return None

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    price = curr['close']
    
    # 趨勢判斷
    trend = "盤整 (No Trade)"
    direction = 0
    if pd.notna(curr['ema200']):
        if curr['ema20'] > curr['ema50'] > curr['ema200']:
            trend = "🔥 多頭趨勢"
            direction = 1
        elif curr['ema20'] < curr['ema50'] < curr['ema200']:
            trend = "❄️ 空頭趨勢"
            direction = -1

    # 計分
    score = 0
    reasons = [] # 格式: (type, text) -> type: 1(多), -1(空)
    
    pin_bull, pin_bear, engulf_bull, engulf_bear = check_candle_pattern(curr, prev)
    diff = curr['struct_h'] - curr['struct_l']
    fib_0618 = curr['struct_h'] - (diff * 0.618)
    
    if direction == 1: # 多頭條件
        recent_low = df['low'].iloc[-10:-1].min()
        if curr['low'] < recent_low and curr['close'] > recent_low: 
            score += 1; reasons.append((1, "掃 Liquidity (下影線)"))
        if prev['rsi'] < 40 and curr['rsi'] > 40: 
            score += 1; reasons.append((1, "RSI 低檔反轉"))
        if pin_bull or engulf_bull: 
            score += 1; reasons.append((1, "K線 (PinBar/吞沒)"))
        if curr['close'] > curr['bb_m'] and prev['close'] < curr['bb_m']: 
            score += 1; reasons.append((1, "站回布林中軌"))
        if curr['volume'] > curr['vol_ma'] * 1.2: 
            score += 1; reasons.append((1, "成交量放大"))
        if abs(price - fib_0618)/price < 0.005: 
            score += 1; reasons.append((1, "回踩 Fib 0.618"))

    elif direction == -1: # 空頭條件
        recent_high = df['high'].iloc[-10:-1].max()
        if curr['high'] > recent_high and curr['close'] < recent_high: 
            score += 1; reasons.append((-1, "掃 Liquidity (假突破)"))
        if prev['rsi'] > 60 and curr['rsi'] < 60: 
            score += 1; reasons.append((-1, "RSI 高檔回落"))
        if pin_bear or engulf_bear: 
            score += 1; reasons.append((-1, "K線 (倒鎚/吞沒)"))
        if curr['close'] < curr['bb_m'] and prev['close'] > curr['bb_m']: 
            score += 1; reasons.append((-1, "跌破布林中軌"))
        if curr['volume'] > curr['vol_ma'] * 1.2: 
            score += 1; reasons.append((-1, "成交量放大"))
        if abs(price - fib_0618)/price < 0.005: 
            score += 1; reasons.append((-1, "反彈至 Fib 0.618"))

    # 止盈止損
    atr = curr['atr']
    sl = curr['low'] - (2*atr) if direction == 1 else curr['high'] + (2*atr)
    tp1 = curr['struct_h'] if direction == 1 else curr['struct_l']
    tp2 = curr['struct_h'] + (diff * 0.618) if direction == 1 else curr['struct_l'] - (diff * 0.618)

    return {
        "price": price, "trend": trend, "direction": direction,
        "score": score, "reasons": reasons,
        "sl": sl, "tp1": tp1, "tp2": tp2, "rsi": curr['rsi']
    }

# --- 核心優化：單一幣種抓取 ---
def process_single_coin(name, symbol, timeframe):
    try:
        exchange = ccxt.kraken()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # 指標計算
        df['ema20'] = ta.ema(df['close'], length=PARAMS['ema_s'])
        df['ema50'] = ta.ema(df['close'], length=PARAMS['ema_m'])
        df['ema200'] = ta.ema(df['close'], length=PARAMS['ema_l'])
        df['rsi'] = ta.rsi(df['close'], length=PARAMS['rsi_len'])
        
        bb = ta.bbands(df['close'], length=PARAMS['bb_len'], std=PARAMS['bb_std'])
        if bb is not None:
            df['bb_u'] = bb.iloc[:, 0]
            df['bb_m'] = bb.iloc[:, 1]
            df['bb_l'] = bb.iloc[:, 2]

        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=PARAMS['atr_len'])
        df['struct_h'] = df['high'].rolling(PARAMS['fib_window']).max()
        df['struct_l'] = df['low'].rolling(PARAMS['fib_window']).min()
        df['vol_ma'] = df['volume'].rolling(20).mean()

        result = analyze_logic(df)
        return name, result
    except Exception as e:
        return name, None

# --- 快取層 (改為 10 秒 TTL，不需要太短因為是手動刷新) ---
@st.cache_data(ttl=10, show_spinner=False)
def fetch_all_market_data(coins_dict, timeframe):
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_coin = {
            executor.submit(process_single_coin, name, symbol, timeframe): name 
            for name, symbol in coins_dict.items()
        }
        for future in concurrent.futures.as_completed(future_to_coin):
            name, data = future.result()
            results[name] = data
    return results

# ==========================================
# 3. 側邊欄：控制台
# ==========================================
st.sidebar.header("🚀 市場掃描 (Kraken)")
timeframe = st.sidebar.select_slider("時間級別", options=["5m", "15m", "1h", "4h"], value="15m")

# 這裡只會執行一次抓取，之後除非按下刷新，否則不會動
with st.spinner("⚡ 正在掃描市場訊號..."):
    scan_results = fetch_all_market_data(BASE_COINS, timeframe)

# 定義顯示格式函式
def format_func_scanner(option_name):
    data = scan_results.get(option_name)
    if data:
        price_fmt = format_price(data['price'])
        if data['score'] >= 3:
            return f"🟢 {option_name} {price_fmt}" # 綠燈
        elif data['direction'] == 0:
            return f"⚪ {option_name} {price_fmt}" # 灰燈
        else:
            return f"🔴 {option_name} {price_fmt}" # 紅燈
    return f"⚠️ {option_name}"

# 選單
selected_coin_name = st.sidebar.radio(
    "點擊查看詳情：", 
    options=list(BASE_COINS.keys()), 
    format_func=format_func_scanner,
    key="main_coin_selector" # 保持 key 以防跳動
)

if st.sidebar.button("🔄 手動刷新數據"):
    st.cache_data.clear()
    st.rerun()

# ==========================================
# 4. 主畫面渲染
# ==========================================
data = scan_results.get(selected_coin_name)

if data:
    p_price = format_price(data['price'])
    p_sl = format_price(data['sl'])
    p_tp1 = format_price(data['tp1'])
    p_tp2 = format_price(data['tp2'])

    card_color = "#333" 
    signal_text = "⏳ 觀望中 (Wait)"
    
    if data['score'] >= 3:
        if data['direction'] == 1:
            card_color = "rgba(0, 204, 150, 0.2)"
            signal_text = f"🚀 訊號成立 (Score {data['score']}) - 做多 LONG"
        elif data['direction'] == -1:
            card_color = "rgba(239, 85, 59, 0.2)"
            signal_text = f"🔻 訊號成立 (Score {data['score']}) - 做空 SHORT"
    else:
        signal_text = f"👀 條件未滿 (Score {data['score']}/6)"

    reasons_html = ""
    for r_type, r_text in data['reasons']:
        if r_type == 1:
            icon = "<span style='color:#00cc96; font-size:1.2em;'>✔</span>" 
            text_color = "#00cc96"
        else:
            icon = "<span style='color:#ef553b; font-size:1.2em;'>✔</span>"
            text_color = "#ef553b"
            
        reasons_html += f"""
        <div style='color:#fff; font-size:1em; margin-bottom:5px; padding: 5px; background:rgba(255,255,255,0.05); border-radius:4px;'>
            {icon} <span style='color:{text_color}; font-weight:bold;'>{r_text}</span>
        </div>
        """

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
        .strategy-footer {{ margin-top:30px; padding:15px; background:#1c1e24; border-radius:8px; font-size:0.85em; color:#aaa; text-align:center; border:1px dashed #444; }}
    </style>
    </head>
    <body>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
            <h1>🔥 {selected_coin_name} 智能戰情室</h1>
            <div style="text-align:right; color:#888;">Kraken | {timeframe}</div>
        </div>

        <div class="grid">
            <div class="card">
                <div class="label">Step 1: 趨勢與現價</div>
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
                <div style="margin-top:15px;">
                    {reasons_html if data['reasons'] else "<div style='color:#ccc; font-style:italic;'>暫無觸發條件...</div>"}
                </div>
            </div>

            <div class="card">
                <div class="label">Step 3: 止盈止損 (Pro)</div>
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

        <div class="strategy-footer">
            💎 <b>策略核心 (God Mode)</b><br>
            方向靠 EMA20/50/200 排列 ＋ 市場結構<br>
            入場靠 Liquidity 掃單後反轉 (需滿足 3 個以上濾網)<br>
            止盈看 Fib 1.618 延伸位｜止損動態設在 2 ATR 處
        </div>
    </body>
    </html>
    """
    components.html(html_content, height=600, scrolling=True)
else:
    st.error("暫時無法獲取數據，請稍後再試。")
