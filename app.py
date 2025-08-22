# ------------------------------------------------------------
# Crypto Dashboard – BTC Accumulation (50/50 plan)  ✨ ONE FILE
# ------------------------------------------------------------
# Cómo correr:
# 1) pip install streamlit pandas requests plotly pyyaml
# 2) streamlit run this_file.py
#
# Qué hace:
# - Lee tu portafolio (Token, Holdings, AvgCost) desde un CSV o usa uno de ejemplo
# - Obtiene precios (CoinGecko)
# - Calcula PnL, semáforos vs T1/T2/T3, cantidades a vender por tramo (25/35/40)
# - Estima BTC/USDT que acumulas por tramo con split 50/50 (editable)
# - Muestra alertas sugeridas y un plan DCA para bear
# - Modo de actualización "SEMANAL/DIARIO" según volatilidad 24h promedio
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from io import StringIO

# =============== CONFIG (EMBEBIDA) ==========================
DEFAULT_CONFIG = {
    "btc_price_manual": None,       # si pones un número, usa ese; si None usa API
    "tranches": [0.25, 0.35, 0.40], # T1/T2/T3
    "split_to_btc": 0.5,            # 50% a BTC, 50% a USDT
    "yellow_buffer_pct": 10,        # semáforo 🟨 cuando precio está a ≤10% de target
    "volatility_daily_threshold": 0.10,  # si |24h| promedio >= 10% => DIARIO
    "targets": {
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
        "USDT": [None, None, None],  # sin targets
        "BTC":  [None, None, None],  # sin targets
    },
    # Mapeo de símbolos a IDs de CoinGecko
    "coingecko_ids": {
        "BTC":"bitcoin",
        "ETH":"ethereum","SUI":"sui","LINK":"chainlink","RAY":"raydium",
        "NEAR":"near","TAO":"bittensor","ONDO":"ondo-finance","USDT":"tether",
        "AVAX":"avalanche-2","ENA":"ethena","SUPRA":"supra-token",
        "TAOBOT":"tao-bot","PUMP":"pump-2","BONK":"bonk",
        "BEAM":"beam","SUPER":"superverse","VIRTUAL":"virtuals-protocol"
    },
}

# CSV de ejemplo (puedes editar en la UI)
DEFAULT_PORTFOLIO_CSV = """Token,Holdings,AvgCost
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

# =============== ESTILO RÁPIDO ===============================
STYLE = """
<style>
.block { padding:12px;border-radius:8px;border:1px solid #2f2f2f; }
.kpi { font-size:1.1rem; font-weight:700; }
.badge { padding:4px 8px;border-radius:6px;font-weight:600; display:inline-block; }
.green { background:#0f5132;color:#d1e7dd; }
.yellow{ background:#664d03;color:#fff3cd; }
.gray  { background:#2c2f33;color:#e0e0e0; }
</style>
"""

# =============== FUNCIONES CORE =============================
def parse_portfolio(csv_text: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(csv_text))
    df["Token"] = df["Token"].str.upper().str.strip()
    df["Holdings"] = pd.to_numeric(df["Holdings"], errors="coerce").fillna(0.0)
    df["AvgCost"] = pd.to_numeric(df["AvgCost"], errors="coerce").fillna(0.0)
    return df

def fetch_simple_price(ids):
    url = (
        "https://api.coingecko.com/api/v3/simple/price"
        f"?ids={ids}&vs_currencies=usd&include_24hr_change=true"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()

def fetch_prices(tokens, cg_map):
    ids = [cg_map[t] for t in tokens if t in cg_map]
    if not ids:
        return {}
    joined = ",".join(ids)
    data = fetch_simple_price(joined)
    out = {}
    inv_cg = {v: k for k, v in cg_map.items()}
    for cid, payload in data.items():
        sym = inv_cg.get(cid)
        if sym:
            out[sym] = {
                "price": float(payload.get("usd", 0.0)),
                "ch24": float(payload.get("usd_24h_change", 0.0))/100.0
            }
    return out

def get_btc_price(cfg, price_map):
    manual = cfg.get("btc_price_manual")
    if manual and float(manual) > 0:
        return float(manual)
    # si ya está en el map, úsalo
    if "BTC" in price_map and price_map["BTC"].get("price"):
        return float(price_map["BTC"]["price"])
    # fallback a una llamada puntual
    try:
        data = fetch_simple_price("bitcoin")
        return float(data["bitcoin"]["usd"])
    except Exception:
        return None

def compute_volatility_mode(price_map, threshold=0.10):
    chs = [abs(v.get("ch24", 0.0)) for v in price_map.values() if "ch24" in v]
    avg_abs = sum(chs)/len(chs) if chs else 0.0
    return ("DIARIO" if avg_abs >= threshold else "SEMANAL", avg_abs)

def compute_signals(df, prices, cfg, btc_price):
    targets = cfg["targets"]
    ybuf = float(cfg["yellow_buffer_pct"]) / 100.0
    tranches = cfg["tranches"]
    split = float(cfg["split_to_btc"])

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
            usdt = usd * (1 - split)
            btc = (usd * split) / btc_price if btc_price else None
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

def dca_plan(total_usdt, ranges=(90000, 80000, 70000), weights=(0.3, 0.4, 0.3)):
    assert abs(sum(weights)-1.0) < 1e-6, "Los pesos del DCA deben sumar 1.0"
    rows = []
    for p, w in zip(ranges, weights):
        usd = total_usdt * w
        btc = usd / p if p else 0.0
        rows.append({"BTC Price": int(p), "USDT": round(usd,2), "BTC Comprable": btc})
    df = pd.DataFrame(rows)
    df.loc["TOTAL"] = {"BTC Price":"—","USDT":df["USDT"].sum(),"BTC Comprable":df["BTC Comprable"].sum()}
    return df

# =============== APP (STREAMLIT) ============================
st.set_page_config(page_title="Crypto Dashboard – BTC Accumulation", layout="wide")
st.markdown(STYLE, unsafe_allow_html=True)
st.title("📊 Crypto Dashboard – BTC Accumulation (50/50)")

cfg = DEFAULT_CONFIG.copy()

# --- Sidebar: carga de CSV y parámetros
st.sidebar.header("📥 Portafolio")
uploaded = st.sidebar.file_uploader("Sube CSV (Token,Holdings,AvgCost)", type=["csv"])
if uploaded:
    csv_text = uploaded.read().decode("utf-8")
else:
    csv_text = DEFAULT_PORTFOLIO_CSV

with st.expander("Ver/Editar CSV actual (opcional)"):
    csv_text = st.text_area("CSV de portafolio", value=csv_text, height=220)

df = parse_portfolio(csv_text)

st.sidebar.header("⚙️ Parámetros")
cfg["split_to_btc"] = st.sidebar.slider("Split a BTC (%)", 0, 100, int(cfg["split_to_btc"]*100)) / 100
cfg["yellow_buffer_pct"] = st.sidebar.slider("Buffer semáforo 🟨 (%)", 0, 25, cfg["yellow_buffer_pct"])
cfg["volatility_daily_threshold"] = st.sidebar.slider("Umbral volatilidad diaria (%)", 5, 30, int(cfg["volatility_daily_threshold"]*100)) / 100
btc_manual = st.sidebar.number_input("Precio BTC (manual, opcional)", value=float(cfg["btc_price_manual"] or 0.0), step=1000.0)
cfg["btc_price_manual"] = None if btc_manual == 0 else btc_manual

# --- Precios
tokens = df["Token"].tolist()
cg_map = cfg["coingecko_ids"]
# aseguramos pedir BTC también
if "BTC" not in tokens:
    tokens_plus_btc = tokens + ["BTC"]
else:
    tokens_plus_btc = tokens

try:
    prices = fetch_prices(tokens_plus_btc, cg_map)
except Exception as e:
    st.warning(f"No pude obtener precios de la API. Motivo: {e}")
    prices = {}

btc_price = get_btc_price(cfg, prices)

# --- Modo (semanal/diario) por volatilidad
mode, avg_abs = compute_volatility_mode(prices, cfg["volatility_daily_threshold"])

# --- Cálculo de señales y totales
signals, totals = compute_signals(df, prices, cfg, btc_price or 1.0)

# =================== KPIs ============================
c1,c2,c3,c4 = st.columns(4)
c1.metric("Valor Portafolio (USD)", f"{totals['Total Valor USD']:,.0f}")
c2.metric("Costo Total (USD)", f"{totals['Total Costo USD']:,.0f}")
c3.metric("PnL (USD)", f"{totals['Total PnL USD']:,.0f}")
c4.metric("Modo actualización", mode, f"Vol. 24h: {avg_abs*100:.1f}%")

c5,c6,c7 = st.columns(3)
c5.metric("BTC por T1 (split)", f"{totals['BTC T1']:.3f}")
c6.metric("BTC por T2 (split)", f"{totals['BTC T2']:.3f}")
c7.metric("BTC por T3 (split)", f"{totals['BTC T3']:.3f}")

# =================== TABLA PRINCIPAL ==================
st.markdown("### 📋 Señales por token (semáforos y tramos)")
tbl = signals.copy()
tbl["24h"] = (tbl["24h"]*100).round(2)
tbl["PnL%"] = (tbl["PnL%"]*100).round(2)
show_cols = [
    "Token","Holdings","AvgCost","Precio","24h","ValorUSD","CostoUSD","PnLUSD","PnL%",
    "T1","Q1","SemT1","USD_T1","USDT_T1","BTC_T1",
    "T2","Q2","SemT2","USD_T2","USDT_T2","BTC_T2",
    "T3","Q3","SemT3","USD_T3","USDT_T3","BTC_T3"
]
st.dataframe(tbl[show_cols], use_container_width=True, height=520)

# =================== ALERTAS ==========================
st.markdown("### 🔔 Alertas sugeridas (crea 3 por token: T1/T2/T3)")
alert_rows = []
for _, r in signals.iterrows():
    for k in ["T1","T2","T3"]:
        lvl = r[k]
        if pd.notnull(lvl):
            alert_rows.append({"Token": r["Token"], "Nivel (USD)": lvl})
st.dataframe(pd.DataFrame(alert_rows), use_container_width=True)

# =================== DCA PLANNER ======================
st.markdown("### 🪙 Plan DCA (Bear) con USDT acumulados por ventas (split)")
usdt_total = float(signals["USDT_T1"].sum(skipna=True) + signals["USDT_T2"].sum(skipna=True) + signals["USDT_T3"].sum(skipna=True))
c8,c9 = st.columns([1,2])
with c8:
    st.write(f"USDT estimado disponible (T1–T3): **{usdt_total:,.0f}**")
    r1 = st.number_input("Rango 1 (USD)", value=90000, step=5000)
    r2 = st.number_input("Rango 2 (USD)", value=80000, step=5000)
    r3 = st.number_input("Rango 3 (USD)", value=70000, step=5000)
    w1 = st.slider("Peso R1 (%)", 0, 100, 30)/100
    w2 = st.slider("Peso R2 (%)", 0, 100, 40)/100
    w3 = max(0.0, 1 - (w1+w2))
    st.caption(f"Peso R3 auto: {w3:.2f}")
with c9:
    dca_df = dca_plan(usdt_total, ranges=(r1,r2,r3), weights=(w1,w2,w3))
    st.dataframe(dca_df, use_container_width=True)
    if "TOTAL" in dca_df.index:
        st.metric("BTC comprable total (DCA)", f"{dca_df.loc['TOTAL','BTC Comprable']:.3f}")

# =================== DISTRIBUCIÓN ======================
st.markdown("### 🧩 Distribución por token (USD)")
dist = signals[["Token","ValorUSD"]].dropna().sort_values("ValorUSD", ascending=False)
if not dist.empty:
    fig = px.bar(dist, x="Token", y="ValorUSD")
    st.plotly_chart(fig, use_container_width=True)

st.success("Dashboard listo. Configura alertas, ejecuta tramos sin emociones y rota 50/50 hacia tu meta de 5 BTC. 🚀")
