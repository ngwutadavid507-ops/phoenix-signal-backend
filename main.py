import os
import asyncio
import time
import json
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from groq import Groq

app = FastAPI(title="Phoenix Signal Backend v5")

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

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
groq_client  = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
print("🚀 Groq ready." if groq_client else "⚠️ GROQ_API_KEY missing.")

# ─── APIS ─────────────────────────────────────────────────

BINANCE    = "https://api.binance.com/api/v3"
BYBIT      = "https://api.bybit.com/v5/market"
POLYMARKET = "https://clob.polymarket.com"
HEADERS    = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

EXCLUDED_BASE = {
    "USDT","USDC","BUSD","DAI","TUSD","FDUSD","FRAX",
    "USDP","USDD","UST","LUSD","MIM","USDJ","GUSD",
    "HUSD","SUSD","CUSD","CEUR","USDX","USDK","AEUR",
    "EUR","GBP","AUD","BRL","RUB","TRY","NGN","JPY",
    "KRW","CHF","CAD","SGD","HKD","INR","MXN","ZAR",
    "BIDR","BVND","IDRT","VAI","UAH",
    "WBTC","WETH","WBNB","STETH","BETH","RETH","CBETH","WSTETH","SBTC"
}

FIAT_STRINGS = ["EUR","GBP","AUD","BRL","TRY","NGN","JPY","KRW","CHF","CAD","ZAR"]

_signal_cache  = {"data": [], "ts": 0}
_pairs_cache   = {"data": [], "ts": 0}
_bybit_cache   = {"symbols": set(), "ts": 0}
_poly_cache    = {"data": {}, "ts": 0}
CACHE_TTL      = 300
POLY_CACHE_TTL = 600

# ─── BYBIT FUTURES VERIFICATION ───────────────────────────

async def get_bybit_futures_symbols() -> set:
    """Fetch all tradeable linear perpetual symbols from Bybit."""
    now = time.time()
    if now - _bybit_cache["ts"] < CACHE_TTL and _bybit_cache["symbols"]:
        return _bybit_cache["symbols"]

    async with httpx.AsyncClient(headers=HEADERS) as client:
        try:
            r = await client.get(
                f"{BYBIT}/instruments-info",
                params={"category": "linear", "limit": 1000},
                timeout=12.0
            )
            if r.status_code == 200:
                items   = r.json().get("result", {}).get("list", [])
                symbols = {
                    item["symbol"] for item in items
                    if item.get("status") == "Trading"
                    and item.get("symbol", "").endswith("USDT")
                }
                _bybit_cache["symbols"] = symbols
                _bybit_cache["ts"]      = time.time()
                print(f"✅ Bybit futures: {len(symbols)} tradeable pairs loaded")
                return symbols
        except Exception as e:
            print(f"Bybit instruments error: {e}")
    return _bybit_cache["symbols"] or set()

# ─── TECHNICAL ANALYSIS ───────────────────────────────────

def compute_rsi(prices: list, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(len(prices) - period, len(prices)):
        diff = prices[i] - prices[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    ag = sum(gains) / period
    al = sum(losses) / period
    if al == 0:
        return 100.0
    return round(100 - (100 / (1 + ag / al)), 2)

def compute_ema(prices: list, period: int) -> float:
    if not prices:
        return 0
    if len(prices) < period:
        return sum(prices) / len(prices)
    k   = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    for p in prices[period:]:
        ema = p * k + ema * (1 - k)
    return round(ema, 8)

def compute_macd(prices: list):
    if len(prices) < 26:
        return 0, 0
    ema12 = compute_ema(prices, 12)
    ema26 = compute_ema(prices, 26)
    macd  = ema12 - ema26
    sig   = compute_ema(prices[-9:], min(9, len(prices[-9:])))
    return round(macd, 8), round(sig, 8)

def compute_bollinger(prices: list, period: int = 20):
    if len(prices) < period:
        p = prices[-1] if prices else 0
        return p, p, p
    w   = prices[-period:]
    mid = sum(w) / period
    std = (sum((p - mid) ** 2 for p in w) / period) ** 0.5
    return round(mid + 2 * std, 8), round(mid, 8), round(mid - 2 * std, 8)

def analyse_coin(closes: list, volumes: list, symbol: str, price: float):
    if len(closes) < 30:
        return None

    rsi                  = compute_rsi(closes)
    ema20                = compute_ema(closes, 20)
    ema50                = compute_ema(closes, 50) if len(closes) >= 50 else ema20
    macd_l, sig_l        = compute_macd(closes)
    bb_up, bb_mid, bb_lo = compute_bollinger(closes)

    avg_vol   = sum(volumes[-14:]) / 14 if len(volumes) >= 14 else (volumes[-1] if volumes else 1)
    last_vol  = volumes[-1] if volumes else 0
    vol_spike = last_vol > avg_vol * 1.4

    bull, bear = 0, 0
    reasons    = []

    if rsi < 35:
        bull += 1
        reasons.append(f"RSI oversold at {rsi} — reversal likely")
    elif rsi > 65:
        bear += 1
        reasons.append(f"RSI overbought at {rsi} — pullback likely")
    elif rsi < 45:
        bull += 0.5
    elif rsi > 55:
        bear += 0.5

    if ema20 > ema50 * 1.001:
        bull += 1
        reasons.append("EMA20 above EMA50 — uptrend confirmed")
    elif ema20 < ema50 * 0.999:
        bear += 1
        reasons.append("EMA20 below EMA50 — downtrend confirmed")

    if macd_l > sig_l:
        bull += 1
        reasons.append("MACD bullish crossover above zero" if macd_l > 0 else "MACD bullish crossover")
    elif macd_l < sig_l:
        bear += 1
        reasons.append("MACD bearish crossover below zero" if macd_l < 0 else "MACD bearish crossover")

    if price <= bb_lo * 1.008:
        bull += 1
        reasons.append("Price at lower Bollinger Band — bounce setup")
    elif price >= bb_up * 0.992:
        bear += 1
        reasons.append("Price at upper Bollinger Band — rejection setup")
    elif price > bb_mid:
        bull += 0.5
        reasons.append("Price above BB midline with momentum")
    else:
        bear += 0.5
        reasons.append("Price below BB midline losing support")

    if len(closes) >= 5:
        mom = (closes[-1] - closes[-5]) / closes[-5] * 100
        if mom > 2:
            bull += 1
            reasons.append(f"Strong 5-period momentum +{mom:.1f}%")
        elif mom < -2:
            bear += 1
            reasons.append(f"Negative 5-period momentum {mom:.1f}%")

    if vol_spike:
        if bull >= bear:
            bull += 1
            reasons.append("Volume spike confirms buying pressure")
        else:
            bear += 1
            reasons.append("Volume spike confirms selling pressure")

    direction = "LONG" if bull > bear else "SHORT"
    score     = bull if direction == "LONG" else bear

    if score < 3:
        return None

    rsi_bonus  = max(0, (50 - rsi) / 4) if direction == "LONG" else max(0, (rsi - 50) / 4)
    macd_bonus = 3 if (direction == "LONG" and macd_l > 0) or (direction == "SHORT" and macd_l < 0) else 0
    confidence = min(96, int(64 + (score / 6) * 28 + rsi_bonus + macd_bonus))

    if confidence < 68:
        return None

    p = price
    if direction == "LONG":
        sl  = round(p * 0.9900, 8)
        tp1 = round(p * 1.0150, 8)
        tp2 = round(p * 1.0260, 8)
        tp3 = round(p * 1.0420, 8)
    else:
        sl  = round(p * 1.0100, 8)
        tp1 = round(p * 0.9850, 8)
        tp2 = round(p * 0.9740, 8)
        tp3 = round(p * 0.9580, 8)

    risk   = abs(p - sl)
    reward = abs(tp2 - p)
    rr     = round(reward / risk, 1) if risk > 0 else 0

    def fmt(v):
        if v >= 1000: return f"{v:,.2f}"
        if v >= 1:    return f"{v:,.3f}"
        return f"{v:.6f}"

    return {
        "pair":       f"{symbol}/USDT",
        "direction":  direction,
        "entry":      fmt(p),
        "sl":         fmt(sl),
        "tp1":        fmt(tp1),
        "tp2":        fmt(tp2),
        "tp3":        fmt(tp3),
        "confidence": f"{confidence}%",
        "rr":         f"1:{rr}",
        "leverage":   "20x",
        "reasons":    [r for r in reasons if r][:4],
        "rsi":        rsi,
        "score":      f"{int(score)}/6"
    }

# ─── OHLCV ────────────────────────────────────────────────

async def fetch_binance_ohlcv(symbol: str, client: httpx.AsyncClient):
    try:
        r = await client.get(
            f"{BINANCE}/klines",
            params={"symbol": symbol, "interval": "1d", "limit": 90},
            timeout=10.0,
            headers=HEADERS
        )
        if r.status_code == 200:
            candles = r.json()
            return [float(c[4]) for c in candles], [float(c[5]) for c in candles]
    except Exception as e:
        print(f"Binance OHLCV error {symbol}: {e}")
    return [], []

async def fetch_bybit_ohlcv(symbol: str, client: httpx.AsyncClient):
    try:
        r = await client.get(
            f"{BYBIT}/kline",
            params={"category": "linear", "symbol": symbol, "interval": "D", "limit": 90},
            timeout=10.0,
            headers=HEADERS
        )
        if r.status_code == 200:
            data = r.json().get("result", {}).get("list", [])
            closes  = [float(c[4]) for c in reversed(data)]
            volumes = [float(c[5]) for c in reversed(data)]
            return closes, volumes
    except Exception as e:
        print(f"Bybit OHLCV error {symbol}: {e}")
    return [], []

# ─── SIGNAL PIPELINE ──────────────────────────────────────

async def generate_signals():
    now = time.time()
    if now - _signal_cache["ts"] < CACHE_TTL and _signal_cache["data"]:
        return _signal_cache["data"]

    # Load Bybit futures symbols first
    bybit_symbols = await get_bybit_futures_symbols()

    async with httpx.AsyncClient(headers=HEADERS) as client:
        try:
            r = await client.get(f"{BINANCE}/ticker/24hr", timeout=12.0)
            all_tickers = r.json() if r.status_code == 200 else []
        except Exception as e:
            print(f"Binance ticker error: {e}")
            return _signal_cache["data"] or []

        candidates = []
        for t in all_tickers:
            sym  = t.get("symbol", "")
            if not sym.endswith("USDT"):
                continue
            base = sym[:-4]
            if base in EXCLUDED_BASE:
                continue
            if any(f in base for f in FIAT_STRINGS):
                continue
            # ── ONLY include if tradeable on Bybit futures ──
            if bybit_symbols and sym not in bybit_symbols:
                continue
            vol   = float(t.get("quoteVolume", 0) or 0)
            price = float(t.get("lastPrice", 0) or 0)
            if vol < 2_000_000 or price <= 0:
                continue
            candidates.append({
                "symbol": sym,
                "base":   base,
                "price":  price,
                "volume": vol,
            })

        candidates.sort(key=lambda x: x["volume"], reverse=True)
        candidates = candidates[:30]

        signals = []
        for coin in candidates:
            sym   = coin["symbol"]
            base  = coin["base"]
            price = coin["price"]

            closes, volumes = await fetch_binance_ohlcv(sym, client)
            if len(closes) < 30:
                closes, volumes = await fetch_bybit_ohlcv(sym, client)
            if len(closes) < 30:
                await asyncio.sleep(0.2)
                continue

            result = analyse_coin(closes, volumes, base, price)
            if result:
                # Add Bybit trading link
                result["bybit_link"] = f"https://www.bybit.com/trade/usdt/{base}USDT"
                signals.append(result)

            await asyncio.sleep(0.15)
            if len(signals) >= 10:
                break

        signals.sort(key=lambda x: int(x["confidence"].replace("%", "")), reverse=True)
        _signal_cache["data"] = signals
        _signal_cache["ts"]   = time.time()
        return signals

# ─── LIVE PAIRS (Bybit linear tickers) ───────────────────

async def fetch_live_pairs(search: str = ""):
    now = time.time()
    if not search and now - _pairs_cache["ts"] < CACHE_TTL and _pairs_cache["data"]:
        return _pairs_cache["data"]

    async with httpx.AsyncClient(headers=HEADERS) as client:
        try:
            # Use Bybit linear tickers — real futures data
            r = await client.get(
                f"{BYBIT}/tickers",
                params={"category": "linear"},
                timeout=12.0
            )
            tickers = r.json().get("result", {}).get("list", []) if r.status_code == 200 else []

            output = []
            for t in tickers:
                sym = t.get("symbol", "")
                if not sym.endswith("USDT"):
                    continue
                base = sym[:-4]
                if base in EXCLUDED_BASE:
                    continue
                if any(f in base for f in FIAT_STRINGS):
                    continue

                price        = float(t.get("lastPrice", 0) or 0)
                change       = float(t.get("price24hPcnt", 0) or 0) * 100
                vol          = float(t.get("turnover24h", 0) or 0)
                high         = float(t.get("highPrice24h", 0) or 0)
                low          = float(t.get("lowPrice24h", 0) or 0)
                funding      = t.get("fundingRate", "")
                open_int     = float(t.get("openInterest", 0) or 0)
                mark_price   = float(t.get("markPrice", 0) or 0)
                bid          = float(t.get("bid1Price", 0) or 0)
                ask          = float(t.get("ask1Price", 0) or 0)

                if price <= 0 or vol < 500_000:
                    continue

                if search:
                    q = search.upper()
                    if q not in base.upper():
                        continue

                def fp(v):
                    if v <= 0:     return "N/A"
                    if v < 0.0001: return f"{v:.8f}"
                    if v < 0.01:   return f"{v:.6f}"
                    if v < 1:      return f"{v:.4f}"
                    if v < 1000:   return f"{v:,.3f}"
                    return f"{v:,.2f}"

                output.append({
                    "symbol":       base,
                    "pair":         sym,
                    "price":        fp(price),
                    "change":       f"{change:+.2f}%",
                    "volume":       f"${vol/1_000_000:.1f}M" if vol >= 1_000_000 else f"${vol:,.0f}",
                    "high_24h":     fp(high),
                    "low_24h":      fp(low),
                    "funding_rate": f"{float(funding)*100:.4f}%" if funding else "N/A",
                    "open_interest":f"${open_int/1_000_000:.1f}M" if open_int >= 1_000_000 else f"{open_int:,.0f}",
                    "mark_price":   fp(mark_price),
                    "bid":          fp(bid),
                    "ask":          fp(ask),
                    "vol_raw":      vol
                })

            output.sort(key=lambda x: x["vol_raw"], reverse=True)
            result = output if search else output[:80]

            if not search:
                _pairs_cache["data"] = result
                _pairs_cache["ts"]   = time.time()

            return result

        except Exception as e:
            print(f"Pairs error: {e}")
    return _pairs_cache["data"] or []

# ─── POLYMARKET LIVE ──────────────────────────────────────

POLY_CATEGORIES = {
    "crypto":      ["bitcoin","ethereum","crypto","btc","eth","blockchain","defi","altcoin"],
    "geopolitics": ["election","war","ceasefire","nato","trade","fed","rate","recession","government","policy"],
    "sports":      ["world cup","nba","nfl","champion","league","tournament","fifa","sport","soccer","basketball"],
    "other":       ["ai","artificial","robot","spacex","moon","inflation","ipo","social media","tech"]
}

async def fetch_polymarket_live() -> dict:
    """Fetch live markets from Polymarket CLOB API."""
    now = time.time()
    if now - _poly_cache["ts"] < POLY_CACHE_TTL and _poly_cache["data"]:
        return _poly_cache["data"]

    async with httpx.AsyncClient(headers=HEADERS) as client:
        try:
            # Fetch active markets sorted by volume
            r = await client.get(
                f"{POLYMARKET}/markets",
                params={
                    "active":    "true",
                    "closed":    "false",
                    "limit":     100,
                    "order":     "volume",
                    "ascending": "false"
                },
                timeout=15.0
            )

            if r.status_code != 200:
                raise Exception(f"Polymarket returned {r.status_code}")

            all_markets = r.json()
            if isinstance(all_markets, dict):
                all_markets = all_markets.get("data", []) or all_markets.get("markets", [])

            categorised = {"crypto": [], "geopolitics": [], "sports": [], "other": []}

            for m in all_markets:
                question = (m.get("question", "") or m.get("title", "") or "").lower()
                if not question:
                    continue

                # Get best odds
                outcomes  = m.get("outcomes", []) or []
                yes_price = 0
                if outcomes:
                    for o in outcomes:
                        if isinstance(o, dict):
                            name  = str(o.get("name", "")).upper()
                            price = float(o.get("price", 0) or 0)
                            if name == "YES" and price > 0:
                                yes_price = price
                                break

                if yes_price == 0:
                    # Try tokens field
                    tokens = m.get("tokens", []) or []
                    for tk in tokens:
                        if isinstance(tk, dict) and str(tk.get("outcome","")).upper() == "YES":
                            yes_price = float(tk.get("price", 0) or 0)
                            break

                vol = float(m.get("volume", 0) or m.get("volumeNum", 0) or 0)

                market_obj = {
                    "title":  m.get("question", m.get("title", "Unknown market")),
                    "odds":   f"{int(yes_price * 100)}%" if yes_price > 0 else "N/A",
                    "volume": f"${vol/1_000_000:.1f}M" if vol >= 1_000_000 else f"${vol:,.0f}",
                    "url":    f"https://polymarket.com/event/{m.get('conditionId','')}"
                }

                # Categorise by keywords
                assigned = False
                for cat, keywords in POLY_CATEGORIES.items():
                    if any(kw in question for kw in keywords):
                        if len(categorised[cat]) < 5:
                            categorised[cat].append(market_obj)
                            assigned = True
                            break

                if not assigned and len(categorised["other"]) < 5:
                    categorised["other"].append(market_obj)

            # Fill any empty categories with fallback
            fallbacks = {
                "crypto": [
                    {"title": "Will Bitcoin exceed $120,000 before end of July 2026?",    "odds": "68%", "volume": "$4.2M", "url": "https://polymarket.com"},
                    {"title": "Will Ethereum surpass $4,000 this month?",                  "odds": "54%", "volume": "$2.1M", "url": "https://polymarket.com"},
                    {"title": "Will total crypto market cap exceed $4T by August 2026?",   "odds": "61%", "volume": "$3.8M", "url": "https://polymarket.com"},
                    {"title": "Will a new altcoin enter top 10 by market cap in Q3 2026?", "odds": "72%", "volume": "$1.9M", "url": "https://polymarket.com"},
                    {"title": "Will BTC dominance fall below 50% before September 2026?",  "odds": "44%
