# --- Reemplazo seguro ---

def cg_simple_price(ids_joined: str) -> dict:
    """Wrapper robusto para CoinGecko simple/price"""
    url = (
        "https://api.coingecko.com/api/v3/simple/price"
        f"?ids={ids_joined}&vs_currencies=usd&include_24hr_change=true"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    # Aseguramos estructura dict
    return data if isinstance(data, dict) else {}

def _safe_float(x, default=0.0):
    try:
        # Solo castea si es int/float/str numérico; None -> default
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def fetch_prices(tokens):
    """
    Devuelve {SYM: {"price": float|None, "ch24": float|None}}
    con tolerancia a campos faltantes o null en CoinGecko.
    """
    # Mapea tus símbolos a IDs de CG (usa tu dict CG_IDS)
    ids = [CG_IDS[t] for t in tokens if t in CG_IDS]
    if not ids:
        return {}

    # CoinGecko puede rate‑limitear; separa en lotes de 50 por seguridad
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
            # Algunos tokens pueden no traer "usd" o venir null
            price = _safe_float(payload.get("usd"), default=None)
            # usd_24h_change puede venir null → usa 0.0 por defecto
            ch_raw = payload.get("usd_24h_change", 0.0)
            ch24 = _safe_float(ch_raw, default=0.0) / 100.0

            out[sym] = {"price": price, "ch24": ch24}

    return out
