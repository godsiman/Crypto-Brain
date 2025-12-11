import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import numpy as np
import time
import streamlit.components.v1 as components

# ==========================================
# 1. 參數與設定區
# ==========================================
COINS = {
    "比特幣 (BTC)": "BTC-USD",
    "以太幣 (ETH)": "ETH-USD",
    "索拉納 (SOL)": "SOL-USD",
    "狗狗幣 (DOGE)": "DOGE-USD",
    "幣安幣 (BNB)": "BNB-USD",
    "瑞波幣 (XRP)": "XRP-USD"
}

# 幣圈常用參數
PARAMS = {
    'ema_s': 20, 'ema_m': 50, 'ema_l': 200,
    'rsi_len': 14, 'rsi_ob': 70, 'rsi_os': 30,
    'bb_len': 20, 'bb_std': 2,
    'macd_fast': 12, 'macd_slow': 26, 'macd_sig': 9,
    'atr_len': 14
}

# ==========================================
# 2. 數據抓取與指標計算 (核心大腦)
# ==========================================
def get_data(symbol, interval='15m', period='5d'):
    try:
        # 下載數據
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        
        if len(df) < 200: return pd.DataFrame() # 資料太少不跑

        # --- 1. MA / EMA (趨勢) ---
        df['ema20'] = ta.ema(df['close'], length=PARAMS['ema_s'])
        df['ema50'] = ta.ema(df['close'], length=PARAMS['ema_m'])
        df['ema200'] = ta.ema(df['close'], length=PARAMS['ema_l'])

        # --- 2. RSI (強弱) ---
        df['rsi'] = ta.rsi(df['close'], length=PARAMS['rsi_len'])

        # --- 3. MACD (動能) ---
        macd = ta.macd(df['close'], fast=PARAMS['macd_fast'], slow=PARAMS['macd_slow'], signal=PARAMS['macd_sig'])
        df['macd'] = macd[f'MACD_{PARAMS["macd_fast"]}_{PARAMS["macd_slow"]}_{PARAMS["macd_sig"]}']
        df['macd_hist'] = macd[f'MACDh_{PARAMS["macd_fast"]}_{PARAMS["macd_slow"]}_{PARAMS["macd_sig"]}']

        # --- 4. Bollinger Bands (波動) ---
        bb = ta.bbands(df['close'], length=PARAMS['bb_len'], std=PARAMS['bb_std'])
        df['bb_upper'] = bb[f'BBU_{PARAMS["bb_len"]}_{PARAMS["bb_std"]}']
        df['bb_lower'] = bb[f'BBL_{PARAMS["bb_len"]}_{PARAMS["bb_std"]}']
        df['bb_width'] = bb[f'BBB_{PARAMS["bb_len"]}_{PARAMS["bb_std"]}'] # 帶寬 (收斂用)

        # --- 6. ATR (止損/波動) ---
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=PARAMS['atr_len'])

        # --- 9. Ichimoku (一目均衡表 - 簡化版) ---
        ichi = ta.ichimoku(df['high'], df['low'], df['close'])[0]
        # 修正：檢查 ichimoku 返回的欄位名稱 (pandas_ta 版本差異)
        span_a_col = [c for c in ichi.columns if 'ISA' in c][0]
        span_b_col = [c for c in ichi.columns if 'ISB' in c][0]
        df['ichi_a'] = ichi[span_a_col]
        df['ichi_b'] = ichi[span_b_col]

        # --- 7. Fibonacci (斐波那契 - 近100根K線的高低點) ---
        recent_high = df['high'].rolling(100).max()
        recent_low = df['low'].rolling(100).min()
        diff = recent_high - recent_low
        df['fib_0.618'] = recent_high - (diff * 0.618)
        df['fib_0.382'] = recent_high - (diff * 0.382)

        return df
    except Exception as e:
        print(f"Data Error: {e}")
        return pd.DataFrame()

# ==========================================
# 3. 策略邏輯引擎 (5大核心 + 暴倉模擬)
# ==========================================
def analyze_market(df):
    if df.empty: return None

    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = last['close']
    
    # --- 判斷 1: EMA 趨勢 (核心策略 1) ---
    trend = "盤整 Consolidate"
    trend_color = "neutral"
    if last['ema20'] > last['ema50'] > last['ema200']:
        trend = "強多頭 (Bull Trend)"
        trend_color = "up"
    elif last['ema20'] < last['ema50'] < last['ema200']:
        trend = "強空頭 (Bear Trend)"
        trend_color = "down"

    # --- 判斷 2: 5大核心策略訊號 ---
    strategies = []

    # 策略 1: 趨勢回調 (EMA多排 + 回踩EMA20/50)
    if trend_color == "up" and (last['low'] <= last['ema20'] or last['low'] <= last['ema50']) and price > last['ema50']:
        strategies.append({"name": "EMA 趨勢回踩", "side": "Long", "desc": "多頭回調進場點"})
    
    # 策略 2: SMC 流動性掃描 (簡單模擬：破前高收低)
    # 這裡用前5根K線最高點當作 Liquidity Pool
    recent_high_5 = df['high'].iloc[-7:-2].max() 
    if last['high'] > recent_high_5 and price < recent_high_5:
        strategies.append({"name": "SMC 掃流動性 (空)", "side": "Short", "desc": "假突破掃止損"})
    
    recent_low_5 = df['low'].iloc[-7:-2].min()
    if last['low'] < recent_low_5 and price > recent_low_5:
        strategies.append({"name": "SMC 掃流動性 (多)", "side": "Long", "desc": "假跌破掃止損"})

    # 策略 3: 布林帶突破 (帶寬壓縮 + 放量突破)
    is_squeeze = last['bb_width'] < df['bb_width'].rolling(50).mean().iloc[-1] * 0.8
    if is_squeeze and price > last['bb_upper']:
        strategies.append({"name": "布林壓縮突破", "side": "Long", "desc": "波動率爆發"})
    elif is_squeeze and price < last['bb_lower']:
        strategies.append({"name": "布林壓縮跌破", "side": "Short", "desc": "波動率爆發"})

    # 策略 4: RSI 背離 (簡化版：超買超賣反轉)
    if last['rsi'] > 70 and price < prev['close']: # 超買轉跌
        strategies.append({"name": "RSI 過熱修正", "side": "Short", "desc": "高檔鈍化反轉"})
    if last['rsi'] < 30 and price > prev['close']: # 超賣轉漲
        strategies.append({"name": "RSI 超賣反彈", "side": "Long", "desc": "低檔背離"})

    # 策略 5: Fibonacci 0.618 回踩
    dist_fib = abs(price - last['fib_0.618']) / price
    if dist_fib < 0.003: # 距離 0.618 非常近 (0.3%)
        strategies.append({"name": "Fib 0.618 黃金位", "side": "Watch", "desc": "關鍵支撐/壓力"})

    # --- 判斷 3: 模擬暴倉地圖 (Liquidation Map) ---
    # 邏輯：尋找過去 50 根 K 線的最高/最低點，這些地方是最多止損單聚集的地方
    liq_high = df['high'].rolling(50).max().iloc[-1]
    liq_low = df['low'].rolling(50).min().iloc[-1]
    
    # 計算 ATR 止損建議
    stop_loss_dist = last['atr'] * 2

    return {
        "price": price,
        "trend": trend,
        "trend_color": trend_color,
        "strategies": strategies,
        "rsi": last['rsi'],
        "macd": last['macd_hist'],
        "liq_high": liq_high,
        "liq_low": liq_low,
        "fib618": last['fib_0.618'],
        "sl_dist": stop_loss_dist,
        "volume": last['volume'],
        "vol_ma": df['volume'].rolling(20).mean().iloc[-1]
    }

# ==========================================
# 4. Streamlit 介面渲染
# ==========================================
st.set_page_config(page_title="Crypto Sniper Pro", layout="wide")

# 側邊欄
st.sidebar.header("🎛️ 戰情室控制台")
selected_coin = st.sidebar.radio("監控幣種", list(COINS.keys()))
timeframe = st.sidebar.select_slider("時間級別", options=["5m", "15m", "1h", "4h", "1d"], value="15m")
st.sidebar.info("💡 數據源: Yahoo Finance\n(無即時暴倉數據，以結構模擬)")

# 抓取數據與分析
symbol = COINS[selected_coin]
df = get_data(symbol, interval=timeframe)
data = analyze_market(df)

# HTML 樣式 (極簡暗黑風)
if data:
    # 構建策略卡片 HTML
    strat_html = ""
    if not data['strategies']:
        strat_html = "<div style='color:#666; padding:10px;'>😴 目前無特定策略訊號，建議觀望</div>"
    else:
        for s in data['strategies']:
            color = "#00cc96" if s['side'] == "Long" else "#ef553b" if s['side'] == "Short" else "#ffa500"
            strat_html += f"""
            <div style="background:#333; padding:10px; border-radius:5px; margin-bottom:8px; border-left: 4px solid {color};">
                <div style="font-weight:bold; color:{color};">{s['name']} <span style="font-size:0.8em; color:#fff; background:{color}; padding:2px 6px; border-radius:4px; margin-left:5px;">{s['side']}</span></div>
                <div style="font-size:0.85em; color:#ccc; margin-top:3px;">{s['desc']}</div>
            </div>
            """

    # 構建暴倉地圖條
    pct_pos = ((data['price'] - data['liq_low']) / (data['liq_high'] - data['liq_low'])) * 100
    pct_pos = max(0, min(100, pct_pos)) # 限制在 0-100

    main_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background-color: #0e1117; color: #fafafa; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .card {{ background-color: #262730; padding: 20px; border-radius: 12px; border: 1px solid #363945; }}
        .big-val {{ font-size: 2.5em; font-weight: bold; font-family: monospace; }}
        .label {{ color: #aaa; font-size: 0.9em; margin-bottom: 5px; }}
        .up {{ color: #00cc96; }} .down {{ color: #ef553b; }} .neutral {{ color: #bbb; }}
        
        /* 暴倉地圖樣式 */
        .liq-bar {{ height: 10px; background: #444; border-radius: 5px; position: relative; margin: 20px 0; }}
        .liq-marker {{ 
            width: 14px; height: 14px; background: #fff; border-radius: 50%; 
            position: absolute; top: -2px; left: {pct_pos}%; transform: translateX(-50%);
            box-shadow: 0 0 10px white;
        }}
        .liq-label {{ font-size: 0.8em; color: #ef553b; position: absolute; top: -20px; }}
    </style>
    </head>
    <body>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <h1>🎯 Crypto Sniper Pro</h1>
            <div style="text-align:right;">
                <div style="font-size:0.9em; color:#888;">{symbol} | {timeframe}</div>
                <div style="font-size:0.8em; color:#555;">更新時間: {time.strftime('%H:%M:%S')}</div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <div class="label">當前價格 (Price)</div>
                <div class="big-val">${data['price']:,.2f}</div>
                <div style="margin-top:10px;">
                    趨勢: <span class="{data['trend_color']}" style="font-weight:bold;">{data['trend']}</span>
                </div>
                <div style="font-size:0.9em; color:#888; margin-top:5px;">
                    RSI: <span style="color:{'#ef553b' if data['rsi']>70 else '#00cc96' if data['rsi']<30 else '#ccc'}">{data['rsi']:.1f}</span> | 
                    MACD: <span style="color:{'#00cc96' if data['macd']>0 else '#ef553b'}">{data['macd']:.2f}</span>
                </div>
                <div style="font-size:0.9em; color:#888; margin-top:5px;">
                    Volume: {'🔥 爆量' if data['volume'] > data['vol_ma']*1.5 else '☁️ 縮量'}
                </div>
            </div>

            <div class="card">
                <div class="label">⚡ 5 大核心策略訊號偵測</div>
                {strat_html}
            </div>

            <div class="card">
                <div class="label">☠️ 模擬暴倉/流動性地圖 (Liquidity Map)</div>
                <div style="font-size:0.8em; color:#888; margin-bottom:10px;">
                    當價格接近 <span style="color:#ef553b">紅字</span> 時，容易觸發大量止損/暴倉。
                </div>
                
                <div class="liq-bar">
                    <div class="liq-label" style="left:0;">${data['liq_low']:,.2f} (多頭止損區)</div>
                    <div class="liq-label" style="right:0;">${data['liq_high']:,.2f} (空頭止損區)</div>
                    <div class="liq-marker"></div>
                </div>

                <hr style="border-color:#444; margin:15px 0;">
                
                <div class="label">🛡️ 風控建議 (ATR Setups)</div>
                <div style="display:flex; justify-content:space-between; font-size:0.9em; color:#ccc;">
                    <span>建議止損距離 (2x ATR):</span>
                    <span style="color:#ffa500">${data['sl_dist']:,.2f}</span>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.9em; color:#ccc; margin-top:5px;">
                    <span>Fib 0.618 關鍵位:</span>
                    <span style="color:#00cc96">${data['fib618']:,.2f}</span>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    components.html(main_html, height=550, scrolling=True)
else:
    st.error("無法抓取數據，請稍後再試或切換幣種。")

# 自動刷新
time.sleep(30)
st.rerun()
