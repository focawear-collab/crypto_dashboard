# -*- coding: utf-8 -*-
# =======================================================================
# Crypto Daily Dashboard – Core Portfolio (2–3x Ready)
# - Robustez frente a None/Null en CoinGecko (sin TypeError)
# - Semáforos T1/T2/T3, split 50/50 BTC–USDT, DCA planner
# - Macro rápidos (BTC Dominance, Fear & Greed)
# - Derivados (Funding BTC, Open Interest Binance)
# - Plotly con fallback seguro si no está instalado
# =======================================================================

import streamlit as st
import pandas as pd
import requests
from io import StringIO
from datetime import datetime

# Plotly opcional (fallback a tabla si no está)
try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

st.set_page_config(page_title="Crypto Daily – Core Portfolio", layout="wide")

# ----------------------- CONFIG ----------------------------------------
DEFAULT_TRANCHES = [0.25, 0.35, 0.40]   # T1/T2/T3
DEFAULT_SPLIT_TO_BTC = 0.50             # 50% BTC / 50% USDT
YELLOW_BUFFER_PCT = 10                  # "cerca de" target (±10%)

# Targets por token (ajusta a gusto)
TARGETS = {
    "ETH": [4900, 8000, 10000],
    "LINK": [50, 75, 100],
    "SUI": [8, 12, 15],
    "NEAR": [15, 20, 25],
    "RAY": [10, 15, 20],
    "TAO": [1200, 1500, 2000],
    "VIRTUAL": [2, 3, 5],
    "BEAM": [0.08, 0.12, 0.20],
    "SUPER": [1.0, 1.5, 2.0],
    "SUPRA": [0.008, 0.012, 0.02],
    "TAOBOT": [0.8, 1.2, 2.0],
    "PUMP": [0.004, 0.006, 0.01],
    "BONK": [0.00003, 0.00005, 0.00008],
    "AVAX": [60, 90, 120],
    "ENA": [1.2, 2.0, 3.0],
    "ONDO": [1.5, 2.5, 4.0],
    "USDT": [None, None, None],
    "BTC":  [None, None, None],
}

# CoinGecko IDs
CG_IDS = {
    "BTC":"bitcoin",
    "ETH":"ethereum","SUI":"sui","LINK":"chainlink","RAY":"raydium",
    "NEAR":"near","TAO":"bittensor","ONDO":"ondo-finance","USDT":"tether",
    "AVAX":"avalanche-2","ENA":"ethena","SUPRA":"supra-token",
    "TAOBOT":"tao-bot","PUMP":"pump-2","BONK":"bonk",
    "BEAM":"beam","SUPER":"superverse","VIRTUAL":"virtuals-protocol"
}

# CSV base (puedes sobrescribirlo en la app)
DEFAULT_CSV = """Token,Holdings,AvgCost
ETH,9,245.52
SUI,3300,1.19
LINK,350,13.64
RAY,2595.411,2.37
NEAR,3249.6,2.27
TAO,23.4064,283.58
VIRTUAL,1600,1.182
BEAM,113000,0.02998
SUPER,1111,0.6744
SUPRA,360762,0.003984
TAOBOT,2982,0.4643
PUMP,386077,0.003949
BONK,44617418,0.00002731
AVAX,234.7,11.57
ENA,7000,0.6874
ONDO,7000,0.8559
USDT,5500,1
"""

# ----------------------- HELPERS (ROBUSTOS) -----------------------------
def cg_simple_price(ids_joined: str) -> dict:
    """Wrapper robusto para CoinGecko simple/price."""
    url = (
        "https://api.coingecko.com/api/v3/simple/price"
        f"?ids={ids_joined}&vs_currencies=usd&include_24hr_change=true"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, dict) else {}

def _safe_float(x, default=0.0):
    """Convierte a float tolerando None y strings no numéricas."""
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def fetch_prices(tokens):
    """
    Devuelve {SYM: {"price": float|None, "ch24": float}} robusto a faltantes.
    Divide en lotes por si hay rate-limit y evita TypeError en None.
    """
    ids = [CG_IDS[t] for t in tokens if t in CG_IDS]
    if not ids:
        return {}

    out = {}
    BATCH = 50
    inv = {v: k for k, v in CG_IDS.items()}

    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i+BATCH]
        data = cg_simple_price(",".join(chunk))
        for cid, payload in (data or {}).items():
            sym = inv.get(cid)
            if not sym:
                continue
            price = _safe_float(payload.get("usd"), default=None)   # None => sin precio
            ch_raw = payload.get("usd_24h_change", 0.0)             # puede venir None
            ch24 = _safe_float(ch_raw, default=0.0) / 100.0
            out[sym] = {"price": price, "ch24": ch24}

    return out

def get_btc_price(price_map: dict, manual: float|None = None) -> float|None:
    if manual and manual > 0:
        return manual
    if "BTC" in price_map and price_map["BTC"].get("price"):
        return float(price_map["BTC"]["price"])
    try:
        data = cg_simple_price("bitcoin")
        return _safe_float(data.get("bitcoin", {}).get("usd"), default=None)
    except Exception:
        return None

def compute_mode(vol_map: dict, threshold=0.10):
    chs = [abs(v.get("ch24",0.0)) for v in vol_map.values() if "ch24" in v]
    avg_abs = sum(chs)/len(chs) if chs else 0.0
    return ("DIARIO" if avg_abs >= threshold else "SEMANAL", avg_abs)

def compute_signals(df, prices, targets, btc_price, tranches, split_btc, yellow_pct):
    ybuf = yellow_pct/100.0
    rows = []
    for _, r in df.iterrows():
        sym = r["Token"]; hold = float(r["Holdings"]); avg = float(r["AvgCost"])
        price = prices.get(sym, {}).get("price")
        ch24  = prices.get(sym, {}).get("ch24", 0.0)
        t1,t2,t3 = (targets.get(sym, [None,None,None]) + [None,None,None])[:3]

        val = price*hold if price is not None else None
        cost = avg*hold
        pnl = (val - cost) if (val is not None) else None
        pnl_pct = (pnl/cost) if (cost>0 and pnl is not None) else None

        q1,q2,q3 = [round(hold*x, 6) for x in tranches]

        def state(target):
            if (target is None) or (price is None): return "—"
            if price >= target: return "🟩 EJECUTAR"
            if price >= target*(1 - ybuf): return "🟨 VIGILAR"
            return "⬜ LEJOS"

        s1,s2,s3 = state(t1), state(t2), state(t3)

        def tranche_calc(q, target):
            if (q is None) or (target is None): return (None,None,None)
            usd = q * target
            # si btc_price None, evitamos división
            btc = (usd * split_btc) / btc_price if btc_price else None
            usdt = usd * (1 - split_btc)
            return (usd, usdt, btc)

        usd1,usdt1,btc1 = tranche_calc(q1, t1)
        usd2,usdt2,btc2 = tranche_calc(q2, t2)
        usd3,usdt3,btc3 = tranche_calc(q3, t3)

        rows.append({
            "Token": sym, "Holdings":hold, "AvgCost":avg, "Precio":price, "24h":ch24,
            "ValorUSD":val, "CostoUSD":cost, "PnLUSD":pnl, "PnL%":pnl_pct,
            "T1":t1,"Q1":q1,"SemT1":s1,"USD_T1":usd1,"USDT_T1":usdt1,"BTC_T1":btc1,
            "T2":t2,"Q2":q2,"SemT2":s2,"USD_T2":usd2,"USDT_T2":usdt2,"BTC_T2":btc2,
            "T3":t3,"Q3":q3,"SemT3":s3,"USD_T3":usd3,"USDT_T3":usdt3,"BTC_T3":btc3
        })

    out = pd.DataFrame(rows)
    totals = {
        "Total Valor USD": out["ValorUSD"].sum(skipna=True),
        "Total Costo USD": out["CostoUSD"].sum(skipna=True),
        "Total PnL USD": out["PnLUSD"].sum(skipna=True),
        "BTC T1": out["BTC_T1"].sum(skipna=True),
        "BTC T2": out["BTC_T2"].sum(skipna=True),
        "BTC T3": out["BTC_T3"].sum(skipna=True),
        "USDT T1": out["USDT_T1"].sum(skipna=True),
        "USDT T2": out["USDT_T2"].sum(skipna=True),
        "USDT T3": out["USDT_T3"].sum(skipna=True),
    }
    return out, totals

def dca_table(total_usdt, ranges=(90000, 80000, 70000), weights=(0.3,0.4,0.3)):
    assert abs(sum(weights)-1.0) < 1e-6
    rows = []
    for p,w in zip(ranges,weights):
        usd = total_usdt*w
        btc = usd/p if p else 0
        rows.append({"BTC Price": int(p), "USDT": round(usd,2), "BTC Comprable": btc})
    df = pd.DataFrame(rows)
    df.loc["TOTAL"] = {"BTC Price":"—","USDT":df["USDT"].sum(),"BTC Comprable":df["BTC Comprable"].sum()}
    return df

# -------------------- Macro quick: BTC.D + F&G --------------------------
def fetch_btc_dominance():
    try:
        resp = requests.get("https://api.coingecko.com/api/v3/global", timeout=15).json()
        return float(resp["data"]["market_cap_percentage"]["btc"])
    except Exception:
        return None

def fetch_fear_greed():
    try:
        fg = requests.get("https://api.alternative.me/fng/?limit=1", timeout=15).json()
        v = fg["data"][0]
        return int(v["value"]), v["value_classification"]
    except Exception:
        return None, None

# --------------- Derivados básicos: Funding & Open Interest -------------
def fetch_binance_funding(symbol="BTCUSDT"):
    try:
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1"
        r = requests.get(url, timeout=15).json()
        if r:
            fr = _safe_float(r[0].get("fundingRate"), default=None)
            ts = int(r[0]["fundingTime"])
            return fr, ts
    except Exception:
        pass
    return None, None

def fetch_binance_oi_hist(symbol="BTCUSDT", period="1h", limit=48):
    try:
        url = f"https://fapi.binance.com/futures/data/openInterestHist?symbol={symbol}&period={period}&limit={limit}"
        r = requests.get(url, timeout=15).json()
        df = pd.DataFrame(r)
        if not df.empty:
            df["sumOpenInterest"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
    except Exception:
        pass
    return pd.DataFrame()

# =========================== UI ========================================
st.title("📊 Crypto Daily – Core Portfolio (2–3x Ready)")
st.caption(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Cargar CSV
st.sidebar.header("📥 Portafolio")
up = st.sidebar.file_uploader("Sube CSV (Token,Holdings,AvgCost)", type=["csv"])
csv_text = up.read().decode("utf-8") if up else DEFAULT_CSV
with st.expander("Ver/editar CSV (opcional)"):
    csv_text = st.text_area("CSV", value=csv_text, height=220)
df = pd.read_csv(StringIO(csv_text))
df["Token"] = df["Token"].str.upper().str.strip()
df["Holdings"] = pd.to_numeric(df["Holdings"], errors="coerce").fillna(0.0)
df["AvgCost"] = pd.to_numeric(df["AvgCost"], errors="coerce").fillna(0.0)

# Parámetros
st.sidebar.header("⚙️ Parámetros")
tr1,tr2,tr3 = DEFAULT_TRANCHES
split_btc = st.sidebar.slider("Split a BTC (%)", 0, 100, int(DEFAULT_SPLIT_TO_BTC*100))/100
yb = st.sidebar.slider("Buffer semáforo (±%)", 0, 25, YELLOW_BUFFER_PCT)
btc_manual = st.sidebar.number_input("Precio BTC (manual, opcional)", value=0.0, step=1000.0)

# Precios
tokens = df["Token"].tolist()
tokens_plus = tokens if "BTC" in tokens else tokens + ["BTC"]
prices = fetch_prices(tokens_plus)

btc_price = get_btc_price(prices, btc_manual if btc_manual>0 else None)
if not btc_price:
    st.warning("No hay precio de BTC disponible (API). Uso 1.0 temporalmente para evitar errores en cálculos.")
    btc_price = 1.0

# Modo por volatilidad 24h
mode, avg_abs = compute_mode(prices, threshold=0.10)

# Señales + Totales
signals, totals = compute_signals(df, prices, TARGETS, btc_price, [tr1,tr2,tr3], split_btc, yb)

# ======================= KPIs ==========================================
c1,c2,c3,c4 = st.columns(4)
c1.metric("Valor Portafolio (USD)", f"{totals['Total Valor USD']:,.0f}")
c2.metric("Costo Total (USD)", f"{totals['Total Costo USD']:,.0f}")
c3.metric("PnL (USD)", f"{totals['Total PnL USD']:,.0f}")
c4.metric("Modo actualización", mode, f"Vol. 24h: {avg_abs*100:.1f}%")

c5,c6,c7 = st.columns(3)
c5.metric("BTC por T1 (split)", f"{totals['BTC T1']:.3f}")
c6.metric("BTC por T2 (split)", f"{totals['BTC T2']:.3f}")
c7.metric("BTC por T3 (split)", f"{totals['BTC T3']:.3f}")

# ======================= Macro quick ===================================
btc_dom = fetch_btc_dominance()
fg_val, fg_txt = fetch_fear_greed()
c8,c9 = st.columns(2)
c8.metric("BTC Dominance (%)", f"{btc_dom:.1f}" if btc_dom is not None else "—")
c9.metric("Fear & Greed", f"{fg_val if fg_val is not None else '—'} ({fg_txt or '—'})")

# ======================= Derivados =====================================
st.markdown("### 🔗 Derivados (Binance Futures)")
fr, fr_ts = fetch_binance_funding("BTCUSDT")
oi_df = fetch_binance_oi_hist("BTCUSDT", period="1h", limit=48)

c10,c11 = st.columns(2)
c10.metric("Funding BTC/USDT (último)", f"{(fr*100):.4f}%" if fr is not None else "—")
if not oi_df.empty and PLOTLY_OK:
    fig_oi = px.line(oi_df, x="timestamp", y="sumOpenInterest", title="BTC Open Interest (Binance, 1h)")
    c11.plotly_chart(fig_oi, use_container_width=True)
elif not oi_df.empty:
    c11.dataframe(oi_df[["timestamp","sumOpenInterest"]], use_container_width=True)
else:
    c11.info("Sin datos de Open Interest disponibles ahora.")

# ======================= Tabla principal ===============================
st.markdown("### 📋 Semáforos por Token (T1/T2/T3)")
tbl = signals.copy()
tbl["24h"] = (tbl["24h"]*100).round(2)
tbl["PnL%"] = (tbl["PnL%"]*100).round(2)
cols = [
    "Token","Holdings","AvgCost","Precio","24h","ValorUSD","CostoUSD","PnLUSD","PnL%",
    "T1","Q1","SemT1","USD_T1","USDT_T1","BTC_T1",
    "T2","Q2","SemT2","USD_T2","USDT_T2","BTC_T2",
    "T3","Q3","SemT3","USD_T3","USDT_T3","BTC_T3"
]
st.dataframe(tbl[cols], use_container_width=True, height=520)

# ======================= Alertas sugeridas ==============================
st.markdown("### 🔔 Niveles para crear alertas (TradingView/CoinStats)")
alerts = []
for _, r in signals.iterrows():
    for k in ["T1","T2","T3"]:
        lvl = r[k]
        if pd.notnull(lvl):
            alerts.append({"Token": r["Token"], "Nivel (USD)": lvl})
st.dataframe(pd.DataFrame(alerts), use_container_width=True)

# ======================= DCA Planner ===================================
st.markdown("### 🪙 Plan DCA (Bear) usando USDT de las ventas (split)")
usdt_total = float(signals["USDT_T1"].sum(skipna=True) + signals["USDT_T2"].sum(skipna=True) + signals["USDT_T3"].sum(skipna=True))
c12,c13 = st.columns([1,2])
with c12:
    st.write(f"USDT estimado por ventas T1–T3: **{usdt_total:,.0f}**")
    r1 = st.number_input("Rango 1 (USD)", value=90000, step=5000)
    r2 = st.number_input("Rango 2 (USD)", value=80000, step=5000)
    r3 = st.number_input("Rango 3 (USD)", value=70000, step=5000)
    w1 = st.slider("Peso R1 (%)", 0, 100, 30)/100
    w2 = st.slider("Peso R2 (%)", 0, 100, 40)/100
    w3 = max(0.0, 1 - (w1+w2))
    st.caption(f"Peso R3 auto: {w3:.2f}")
with c13:
    dca_df = dca_table(usdt_total, ranges=(r1,r2,r3), weights=(w1,w2,w3))
    st.dataframe(dca_df, use_container_width=True)
    if "TOTAL" in dca_df.index:
        st.metric("BTC comprable total (DCA)", f"{dca_df.loc['TOTAL','BTC Comprable']:.3f}")

# ======================= Distribución por token =========================
st.markdown("### 🧩 Distribución por token (USD)")
dist = signals[["Token","ValorUSD"]].dropna().sort_values("ValorUSD", ascending=False)
if not dist.empty and PLOTLY_OK:
    fig = None
    try:
        fig = px.bar(dist, x="Token", y="ValorUSD", title="Distribución por Token (USD)")
    except Exception:
        fig = None
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
elif not dist.empty:
    st.dataframe(dist, use_container_width=True)

st.success("Listo. Dashboard robusto activo. Si CoinGecko devuelve null en 24h_change o precios, el panel sigue funcionando sin caerse. 🚀")
