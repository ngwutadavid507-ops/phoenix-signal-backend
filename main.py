import os
import asyncio
import time
import json
import hmac
import hashlib
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from groq import Groq

app = FastAPI(title="Phoenix Signal Backend v6")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: str
    history: list = []

GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "").strip()
BYBIT_API_KEY   = os.getenv("BYBIT_API_KEY", "").strip()
BYBIT_API_SECRET= os.getenv("BYBIT_API_SECRET", "").strip()

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
print("🚀 Groq ready."        if groq_client    else "⚠️ GROQ_API_KEY missing.")
print("🔑 Bybit API ready."   if BYBIT_API_KEY  else "⚠️ BYBIT_API_KEY missing.")

BINANCE = "https://api.binance.com/api/v3"
BYBIT   = "https://api.bybit.com/v5/market"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

EXCLUDED_BASE = {
    "USDT","USDC","BUSD","DAI","TUSD","FDUSD","FRAX",
    "USDP","USDD","UST","LUSD","MIM","USDJ","GUSD",
    "HUSD","SUSD","CUSD","USDX","USDK","AEUR",
    "EUR","GBP","AUD","BRL","RUB","TRY","NGN","JPY",
    "KRW","CHF","CAD","SGD","HKD","INR","MXN","ZAR",
    "BIDR","BVND","IDRT","VAI","UAH",
    "WBTC","WETH","WBNB","STETH","BETH","RETH","CBETH","WSTETH","SBTC"
}
FIAT_STRINGS = ["EUR","GBP","AUD","BRL","TRY","NGN","JPY","KRW","CHF","CAD","ZAR"]

_signal_cache = {"data": [], "ts": 0}
_pairs_cache  = {"data": [], "ts": 0}
_bybit_cache  = {"symbols": set(), "ts": 0}
_poly_cache   = {"data": {}, "ts": 0}
CACHE_TTL     = 300
POLY_TTL      = 600

# ─── BYBIT AUTH HELPER ────────────────────────────────────

def bybit_headers(params: dict = {}) -> dict:
    """Generate authenticated Bybit headers."""
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        return HEADERS
    ts        = str(int(time.time() * 1000))
    recv_wind = "5000"
    param_str = ts + BYBIT_API_KEY + recv_wind + "&".join(f"{k}={v}" for k,v in sorted(params.items()))
    signature = hmac.new(BYBIT_API_SECRET.encode(), param_str.encode(), hashlib.sha256).hexdigest()
    return {
        **HEADERS,
        "X-BAPI-API-KEY":       BYBIT_API_KEY,
        "X-BAPI-TIMESTAMP":     ts,
        "X-BAPI-SIGN":          signature,
        "X-BAPI-RECV-WINDOW":   recv_wind,
    }

def safe_float(val, default=0.0) -> float:
    """Safely convert any value to float."""
    try:
        if val is None or val == "" or val == "N/A":
            return default
        return float(val)
    except (ValueError, TypeError):
        return default

# ─── BYBIT FUTURES SYMBOLS ────────────────────────────────

async def get_bybit_symbols() -> set:
    now = time.time()
    if now - _bybit_cache["ts"] < CACHE_TTL and _bybit_cache["symbols"]:
        return _bybit_cache["symbols"]

    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(
                f"{BYBIT}/instruments-info",
                params={"category": "linear", "limit": 1000},
                headers=HEADERS,
                timeout=15.0
            )
            if r.status_code == 200:
                items = r.json().get("result", {}).get("list", [])
                syms  = {
                    i["symbol"] for i in items
                    if i.get("status") == "Trading"
                    and i.get("symbol","").endswith("USDT")
                }
                _bybit_cache["symbols"] = syms
                _bybit_cache["ts"]      = time.time()
                print(f"✅ Bybit: {len(syms)} active futures loaded")
                return syms
        except Exception as e:
            print(f"Bybit instruments error: {e}")
    return _bybit_cache["symbols"] or set()

# ─── TECHNICAL ANALYSIS ───────────────────────────────────

def compute_rsi(prices, period=14):
    if len(prices) < period + 1: return 50.0
    gains  = [max(prices[i]-prices[i-1], 0) for i in range(len(prices)-period, len(prices))]
    losses = [max(prices[i-1]-prices[i], 0) for i in range(len(prices)-period, len(prices))]
    ag = sum(gains)/period; al = sum(losses)/period
    return round(100-(100/(1+ag/al)), 2) if al > 0 else 100.0

def compute_ema(prices, period):
    if not prices: return 0
    if len(prices) < period: return sum(prices)/len(prices)
    k = 2/(period+1)
    ema = sum(prices[:period])/period
    for p in prices[period:]: ema = p*k + ema*(1-k)
    return round(ema, 8)

def compute_macd(prices):
    if len(prices) < 26: return 0, 0
    return (round(compute_ema(prices,12)-compute_ema(prices,26), 8),
            round(compute_ema(prices[-9:], min(9,len(prices[-9:]))), 8))

def compute_bollinger(prices, period=20):
    if len(prices) < period:
        p = prices[-1] if prices else 0; return p,p,p
    w = prices[-period:]; mid = sum(w)/period
    std = (sum((p-mid)**2 for p in w)/period)**0.5
    return round(mid+2*std,8), round(mid,8), round(mid-2*std,8)

def analyse_coin(closes, volumes, symbol, price):
    if len(closes) < 30 or price <= 0: return None

    rsi                   = compute_rsi(closes)
    ema20                 = compute_ema(closes, 20)
    ema50                 = compute_ema(closes, 50) if len(closes)>=50 else ema20
    macd_l, sig_l         = compute_macd(closes)
    bb_up, bb_mid, bb_lo  = compute_bollinger(closes)
    avg_vol   = sum(volumes[-14:])/14 if len(volumes)>=14 else (volumes[-1] if volumes else 1)
    vol_spike = (volumes[-1] if volumes else 0) > avg_vol * 1.4

    bull, bear, reasons = 0, 0, []

    if rsi < 35:    bull+=1; reasons.append(f"RSI oversold at {rsi} — reversal likely")
    elif rsi > 65:  bear+=1; reasons.append(f"RSI overbought at {rsi} — pullback likely")
    elif rsi < 45:  bull+=0.5
    elif rsi > 55:  bear+=0.5

    if ema20 > ema50*1.001:    bull+=1; reasons.append("EMA20 above EMA50 — uptrend confirmed")
    elif ema20 < ema50*0.999:  bear+=1; reasons.append("EMA20 below EMA50 — downtrend confirmed")

    if macd_l > sig_l:    bull+=1; reasons.append("MACD bullish crossover" + (" above zero" if macd_l>0 else ""))
    elif macd_l < sig_l:  bear+=1; reasons.append("MACD bearish crossover" + (" below zero" if macd_l<0 else ""))

    if price <= bb_lo*1.008:    bull+=1; reasons.append("Price at lower Bollinger Band — bounce setup")
    elif price >= bb_up*0.992:  bear+=1; reasons.append("Price at upper Bollinger Band — rejection setup")
    elif price > bb_mid:        bull+=0.5; reasons.append("Price above BB midline with momentum")
    else:                       bear+=0.5; reasons.append("Price below BB midline losing support")

    if len(closes) >= 5:
        mom = (closes[-1]-closes[-5])/closes[-5]*100
        if mom > 2:    bull+=1; reasons.append(f"Strong 5-period momentum +{mom:.1f}%")
        elif mom < -2: bear+=1; reasons.append(f"Negative 5-period momentum {mom:.1f}%")

    if vol_spike:
        if bull>=bear: bull+=1; reasons.append("Volume spike confirms buying pressure")
        else:          bear+=1; reasons.append("Volume spike confirms selling pressure")

    direction = "LONG" if bull > bear else "SHORT"
    score     = bull if direction=="LONG" else bear
    if score < 3: return None

    rsi_b  = max(0,(50-rsi)/4)   if direction=="LONG"  else max(0,(rsi-50)/4)
    macd_b = 3 if (direction=="LONG" and macd_l>0) or (direction=="SHORT" and macd_l<0) else 0
    conf   = min(96, int(64+(score/6)*28+rsi_b+macd_b))
    if conf < 68: return None

    p = price
    if direction=="LONG":
        sl,tp1,tp2,tp3 = round(p*0.99,8),round(p*1.015,8),round(p*1.026,8),round(p*1.042,8)
    else:
        sl,tp1,tp2,tp3 = round(p*1.01,8),round(p*0.985,8),round(p*0.974,8),round(p*0.958,8)

    rr = round(abs(tp2-p)/abs(p-sl),1) if abs(p-sl)>0 else 0

    def fmt(v):
        if v>=1000: return f"{v:,.2f}"
        if v>=1:    return f"{v:,.3f}"
        return f"{v:.6f}"

    return {
        "pair":f"{symbol}/USDT","direction":direction,
        "entry":fmt(p),"sl":fmt(sl),"tp1":fmt(tp1),"tp2":fmt(tp2),"tp3":fmt(tp3),
        "confidence":f"{conf}%","rr":f"1:{rr}","leverage":"20x",
        "reasons":[r for r in reasons if r][:4],"rsi":rsi,"score":f"{int(score)}/6",
        "bybit_link":f"https://www.bybit.com/trade/usdt/{symbol}USDT"
    }

# ─── OHLCV ────────────────────────────────────────────────

async def fetch_binance_ohlcv(symbol, client):
    try:
        r = await client.get(f"{BINANCE}/klines",
            params={"symbol":symbol,"interval":"1d","limit":90},
            timeout=12.0, headers=HEADERS)
        if r.status_code == 200:
            c = r.json()
            return [safe_float(x[4]) for x in c], [safe_float(x[5]) for x in c]
    except Exception as e:
        print(f"Binance OHLCV {symbol}: {e}")
    return [], []

async def fetch_bybit_ohlcv(symbol, client):
    try:
        params = {"category":"linear","symbol":symbol,"interval":"D","limit":90}
        r = await client.get(f"{BYBIT}/kline",
            params=params, timeout=12.0,
            headers=bybit_headers(params))
        if r.status_code == 200:
            d = r.json().get("result",{}).get("list",[])
            closes  = [safe_float(c[4]) for c in reversed(d)]
            volumes = [safe_float(c[5]) for c in reversed(d)]
            return closes, volumes
    except Exception as e:
        print(f"Bybit OHLCV {symbol}: {e}")
    return [], []

# ─── SIGNALS ──────────────────────────────────────────────

async def generate_signals():
    now = time.time()
    if now-_signal_cache["ts"]<CACHE_TTL and _signal_cache["data"]:
        return _signal_cache["data"]

    bybit_syms = await get_bybit_symbols()

    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{BINANCE}/ticker/24hr", timeout=15.0, headers=HEADERS)
            tickers = r.json() if r.status_code==200 else []
        except Exception as e:
            print(f"Binance ticker: {e}"); return _signal_cache["data"] or []

        candidates = []
        for t in tickers:
            sym   = t.get("symbol","")
            if not sym.endswith("USDT"): continue
            base  = sym[:-4]
            if base in EXCLUDED_BASE: continue
            if any(f in base for f in FIAT_STRINGS): continue
            if bybit_syms and sym not in bybit_syms: continue
            vol   = safe_float(t.get("quoteVolume"))
            price = safe_float(t.get("lastPrice"))
            if vol < 2_000_000 or price <= 0: continue
            candidates.append({"symbol":sym,"base":base,"price":price,"volume":vol})

        candidates.sort(key=lambda x: x["volume"], reverse=True)
        candidates = candidates[:30]
        print(f"Signal candidates: {len(candidates)}")

        signals = []
        for coin in candidates:
            sym,base,price = coin["symbol"],coin["base"],coin["price"]
            closes,volumes = await fetch_binance_ohlcv(sym, client)
            if len(closes) < 30:
                closes,volumes = await fetch_bybit_ohlcv(sym, client)
            if len(closes) < 30:
                await asyncio.sleep(0.2); continue
            result = analyse_coin(closes, volumes, base, price)
            if result:
                signals.append(result)
                print(f"✅ Signal: {base} {result['direction']} {result['confidence']}")
            await asyncio.sleep(0.15)
            if len(signals) >= 10: break

        signals.sort(key=lambda x:int(x["confidence"].replace("%","")), reverse=True)
        _signal_cache["data"] = signals
        _signal_cache["ts"]   = time.time()
        print(f"Signals generated: {len(signals)}")
        return signals

# ─── LIVE PAIRS ───────────────────────────────────────────

async def fetch_live_pairs(search: str = ""):
    now = time.time()
    if not search and now-_pairs_cache["ts"]<CACHE_TTL and _pairs_cache["data"]:
        return _pairs_cache["data"]

    async with httpx.AsyncClient() as client:
        try:
            # Fetch with auth if available
            params = {"category": "linear"}
            r = await client.get(
                f"{BYBIT}/tickers",
                params=params,
                headers=bybit_headers(params),
                timeout=15.0
            )
            print(f"Bybit tickers status: {r.status_code}")

            if r.status_code != 200:
                raise Exception(f"Bybit returned {r.status_code}: {r.text[:200]}")

            raw     = r.json()
            tickers = raw.get("result", {}).get("list", [])
            print(f"Bybit raw tickers count: {len(tickers)}")

            output = []
            for t in tickers:
                sym = t.get("symbol","")
                if not sym.endswith("USDT"): continue
                base = sym[:-4]
                if base in EXCLUDED_BASE: continue
                if any(f in base for f in FIAT_STRINGS): continue

                # Use safe_float for ALL fields — Bybit returns empty strings
                price   = safe_float(t.get("lastPrice"))
                change  = safe_float(t.get("price24hPcnt")) * 100
                vol     = safe_float(t.get("turnover24h"))
                high    = safe_float(t.get("highPrice24h"))
                low     = safe_float(t.get("lowPrice24h"))
                funding = t.get("fundingRate","")
                oi      = safe_float(t.get("openInterest"))
                mark    = safe_float(t.get("markPrice"))
                bid     = safe_float(t.get("bid1Price"))
                ask     = safe_float(t.get("ask1Price"))

                # Only skip if truly no price at all
                if price <= 0 and mark <= 0:
                    continue

                # Use mark price as fallback if lastPrice is 0
                display_price = price if price > 0 else mark

                if search and search.upper() not in base.upper():
                    continue

                def fp(v):
                    if v <= 0:      return "—"
                    if v < 0.0001:  return f"{v:.8f}"
                    if v < 0.01:    return f"{v:.6f}"
                    if v < 1:       return f"{v:.4f}"
                    if v < 1000:    return f"{v:,.3f}"
                    return f"{v:,.2f}"

                fund_str = "—"
                if funding and funding != "":
                    try: fund_str = f"{float(funding)*100:.4f}%"
                    except: pass

                output.append({
                    "symbol":       base,
                    "pair":         sym,
                    "price":        fp(display_price),
                    "change":       f"{change:+.2f}%",
                    "volume":       f"${vol/1_000_000:.1f}M" if vol>=1_000_000 else (f"${vol:,.0f}" if vol>0 else "—"),
                    "high_24h":     fp(high),
                    "low_24h":      fp(low),
                    "funding_rate": fund_str,
                    "open_interest":f"${oi/1_000_000:.1f}M" if oi>=1_000_000 else (f"{oi:,.0f}" if oi>0 else "—"),
                    "mark_price":   fp(mark),
                    "bid":          fp(bid),
                    "ask":          fp(ask),
                    "vol_raw":      vol
                })

            output.sort(key=lambda x: x["vol_raw"], reverse=True)
            result = output if search else output[:80]
            print(f"Pairs output: {len(result)} (search='{search}')")

            if not search:
                _pairs_cache["data"] = result
                _pairs_cache["ts"]   = time.time()

            return result

        except Exception as e:
            print(f"Pairs fetch error: {e}")
            import traceback; traceback.print_exc()

    return _pairs_cache["data"] or []

# ─── TRENDING (Bybit top gainers) ─────────────────────────

async def fetch_trending():
    async with httpx.AsyncClient() as client:
        try:
            params = {"category": "linear"}
            r = await client.get(
                f"{BYBIT}/tickers",
                params=params,
                headers=bybit_headers(params),
                timeout=12.0
            )
            tickers = r.json().get("result",{}).get("list",[]) if r.status_code==200 else []

            gainers = []
            for t in tickers:
                sym = t.get("symbol","")
                if not sym.endswith("USDT"): continue
                base   = sym[:-4]
                if base in EXCLUDED_BASE: continue
                if any(f in base for f in FIAT_STRINGS): continue
                change = safe_float(t.get("price24hPcnt")) * 100
                price  = safe_float(t.get("lastPrice"))
                vol    = safe_float(t.get("turnover24h"))
                if vol < 1_000_000 or (price <= 0 and safe_float(t.get("markPrice")) <= 0):
                    continue
                gainers.append({
                    "name":   base,
                    "symbol": base,
                    "change": f"{change:+.2f}%",
                    "volume": f"${vol/1_000_000:.1f}M",
                    "price":  f"{price:.4f}" if 0<price<1 else f"{price:,.2f}",
                    "rank":   "—",
                    "thumb":  "",
                    "change_raw": change
                })

            gainers.sort(key=lambda x: x["change_raw"], reverse=True)
            for g in gainers: g.pop("change_raw", None)
            return {"trending": gainers[:10]}

        except Exception as e:
            print(f"Trending error: {e}")
    return {"trending": []}

# ─── POLYMARKET ───────────────────────────────────────────

POLY_CATEGORIES = {
    "crypto":      ["bitcoin","ethereum","crypto","btc","eth","blockchain","defi","altcoin","token","sol","bnb","xrp"],
    "geopolitics": ["election","war","ceasefire","nato","trade","fed","rate","recession","president","government","policy","sanctions","china","russia","ukraine","iran","trump","biden","congress"],
    "sports":      ["world cup","nba","nfl","champion","league","tournament","fifa","sport","soccer","basketball","football","tennis","golf","olympics","win","match","game"],
    "other":       ["ai","artificial","robot","spacex","moon","inflation","ipo","social media","tech","elon","climate","energy","apple","google","microsoft"]
}

async def fetch_polymarket_live() -> dict:
    categorised = {"crypto":[],"geopolitics":[],"sports":[],"other":[]}

    async with httpx.AsyncClient(headers=HEADERS) as client:
        try:
            # Try gamma API with active flag and high volume sort
            r = await client.get(
                "https://gamma-api.polymarket.com/markets",
                params={
                    "active":   "true",
                    "closed":   "false",
                    "limit":    300,
                    "order":    "volume24hr",
                    "ascending":"false"
                },
                timeout=20.0
            )
            print(f"Polymarket status: {r.status_code}")

            markets = []
            if r.status_code == 200:
                data = r.json()
                markets = data if isinstance(data, list) else data.get("data", data.get("markets", []))

            print(f"Polymarket raw markets: {len(markets)}")

            # Current year filter — skip anything clearly expired
            current_year = 2026
            skip_years   = ["2023","2024-01","2024-02","2024-03","2024-04","2024-05","2024-06"]

            for m in markets:
                question = (m.get("question","") or m.get("title","") or "").strip()
                if not question or len(question) < 8:
                    continue

                # Skip old markets
                end_date = str(m.get("endDate","") or m.get("end_date","") or "")
                if any(y in end_date for y in skip_years):
                    continue

                # Skip zero volume old markets
                vol = safe_float(m.get("volume24hr") or m.get("volume") or m.get("volumeNum"))
                if vol == 0:
    
