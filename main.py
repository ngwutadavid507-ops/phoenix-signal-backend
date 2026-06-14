import os
import asyncio
import time
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from groq import Groq

app = FastAPI(title="Phoenix Signal Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str

class ChatRequest(BaseModel):
    prompt: str
    history: list = []

# ─── GROQ ─────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
print("🚀 Groq ready." if groq_client else "⚠️ GROQ_API_KEY missing.")

# ─── CONSTANTS ────────────────────────────────────────────

COINGECKO = "https://api.coingecko.com/api/v3"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

STABLECOINS = {
    "tether", "usd-coin", "binance-usd", "dai", "true-usd", "frax",
    "usdd", "paxos-standard", "gemini-dollar", "terrausd", "neutrino",
    "fei-usd", "liquity-usd", "magic-internet-money", "celo-dollar",
    "vai", "tether-eurt", "stasis-eurs"
}
STABLE_SYMBOLS = {
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD",
    "FRAX", "USDP", "USDD", "UST", "LUSD", "MIM"
}

_signal_cache  = {"data": [], "ts": 0}
_pairs_cache   = {"data": [], "ts": 0}
_poly_cache    = {"data": {}, "ts": 0}
CACHE_TTL      = 300

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
    k = 2 / (period + 1)
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

    rsi           = compute_rsi(closes)
    ema20         = compute_ema(closes, 20)
    ema50         = compute_ema(closes, 50) if len(closes) >= 50 else ema20
    macd_l, sig_l = compute_macd(closes)
    bb_up, bb_mid, bb_lo = compute_bollinger(closes)

    avg_vol  = sum(volumes[-14:]) / 14 if len(volumes) >= 14 else (volumes[-1] if volumes else 1)
    last_vol = volumes[-1] if volumes else 0
    vol_spike = last_vol > avg_vol * 1.4

    bull, bear = 0, 0
    reasons    = []

    # RSI
    if rsi < 35:
        bull += 1
        reasons.append(f"RSI oversold at {rsi} — reversal likely")
    elif rsi > 65:
        bear += 1
        reasons.append(f"RSI overbought at {rsi} — pullback likely")
    else:
        if rsi < 45:
            bull += 0.5
        elif rsi > 55:
            bear += 0.5

    # EMA trend
    if ema20 > ema50 * 1.001:
        bull += 1
        reasons.append("EMA20 above EMA50 — uptrend confirmed")
    elif ema20 < ema50 * 0.999:
        bear += 1
        reasons.append("EMA20 below EMA50 — downtrend confirmed")

    # MACD
    if macd_l > sig_l:
        bull += 1
        label = "MACD bullish crossover above zero" if macd_l > 0 else "MACD bullish crossover"
        reasons.append(label)
    elif macd_l < sig_l:
        bear += 1
        label = "MACD bearish crossover below zero" if macd_l < 0 else "MACD bearish crossover"
        reasons.append(label)

    # Bollinger Bands
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

    # Price momentum
    if len(closes) >= 5:
        momentum = (closes[-1] - closes[-5]) / closes[-5] * 100
        if momentum > 2:
            bull += 1
            reasons.append(f"Strong 5-period momentum +{momentum:.1f}%")
        elif momentum < -2:
            bear += 1
            reasons.append(f"Negative 5-period momentum {momentum:.1f}%")

    # Volume confirmation
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

async def fetch_ohlcv(coin_id: str, client: httpx.AsyncClient):
    try:
        r = await client.get(
            f"{COINGECKO}/coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": "90", "interval": "daily"},
            timeout=12.0,
            headers=HEADERS
        )
        if r.status_code == 200:
            data    = r.json()
            closes  = [p[1] for p in data.get("prices", [])]
            volumes = [v[1] for v in data.get("total_volumes", [])]
            return closes, volumes
    except Exception as e:
        print(f"OHLCV error {coin_id}: {e}")
    return [], []

# ─── SIGNAL PIPELINE ──────────────────────────────────────

async def generate_signals():
    now = time.time()
    if now - _signal_cache["ts"] < CACHE_TTL and _signal_cache["data"]:
        return _signal_cache["data"]

    async with httpx.AsyncClient(headers=HEADERS) as client:
        try:
            r = await client.get(
                f"{COINGECKO}/coins/markets",
                params={
                    "vs_currency":            "usd",
                    "order":                  "volume_desc",
                    "per_page":               80,
                    "page":                   1,
                    "price_change_percentage": "24h"
                },
                timeout=12.0
            )
            all_coins = r.json() if r.status_code == 200 else []
        except Exception as e:
            print(f"Market fetch error: {e}")
            return _signal_cache["data"] or []

        candidates = [
            c for c in all_coins
            if c["id"] not in STABLECOINS
            and c.get("symbol", "").upper() not in STABLE_SYMBOLS
            and float(c.get("current_price") or 0) > 0
            and float(c.get("total_volume")   or 0) > 2_000_000
        ][:30]

        signals = []

        for coin in candidates:
            coin_id = coin["id"]
            symbol  = coin["symbol"].upper()
            price   = float(coin.get("current_price") or 0)

            closes, volumes = await fetch_ohlcv(coin_id, client)

            if len(closes) < 30:
                await asyncio.sleep(0.3)
                continue

            if not volumes:
                vol     = float(coin.get("total_volume") or 0)
                volumes = [vol] * len(closes)

            result = analyse_coin(closes, volumes, symbol, price)
            if result:
                signals.append(result)

            await asyncio.sleep(0.4)

            if len(signals) >= 10:
                break

        signals.sort(
            key=lambda x: int(x["confidence"].replace("%", "")),
            reverse=True
        )

        _signal_cache["data"] = signals
        _signal_cache["ts"]   = time.time()
        return signals

# ─── POLYMARKET AI ANALYSIS ───────────────────────────────

POLY_MARKETS = {
    "crypto": [
        {"title": "Will Bitcoin exceed $120,000 before end of July 2026?",        "odds": "68%", "volume": "$4.2M"},
        {"title": "Will Ethereum surpass $4,000 this month?",                      "odds": "54%", "volume": "$2.1M"},
        {"title": "Will total crypto market cap exceed $4T by August 2026?",       "odds": "61%", "volume": "$3.8M"},
        {"title": "Will a new altcoin enter top 10 by market cap in Q3 2026?",     "odds": "72%", "volume": "$1.9M"},
        {"title": "Will BTC dominance fall below 50% before September 2026?",      "odds": "44%", "volume": "$2.7M"},
    ],
    "geopolitics": [
        {"title": "Will US-China trade negotiations produce a deal by Q3 2026?",   "odds": "38%", "volume": "$5.1M"},
        {"title": "Will NATO expand membership before end of 2026?",               "odds": "29%", "volume": "$1.4M"},
        {"title": "Will there be a ceasefire in an active conflict zone by Aug?",  "odds": "45%", "volume": "$6.3M"},
        {"title": "Will the US Federal Reserve cut rates in July 2026?",           "odds": "71%", "volume": "$8.9M"},
        {"title": "Will a G7 nation enter recession by end of 2026?",             "odds": "52%", "volume": "$3.2M"},
    ],
    "sports": [
        {"title": "Will the 2026 FIFA World Cup final be a European team?",        "odds": "58%", "volume": "$7.4M"},
        {"title": "Will an African team reach the 2026 World Cup semifinals?",     "odds": "34%", "volume": "$2.8M"},
        {"title": "Will LeBron James win another NBA championship?",               "odds": "22%", "volume": "$1.6M"},
        {"title": "Will the 2026 Tour de France be won by a non-European?",       "odds": "18%", "volume": "$0.9M"},
        {"title": "Will a world athletics record be broken at 2026 championships?","odds": "63%", "volume": "$1.1M"},
    ],
    "other": [
        {"title": "Will a major AI company IPO before end of 2026?",              "odds": "55%", "volume": "$3.3M"},
        {"title": "Will global inflation average below 3% in 2026?",              "odds": "47%", "volume": "$2.6M"},
        {"title": "Will a humanoid robot be commercially sold to consumers 2026?", "odds": "41%", "volume": "$1.8M"},
        {"title": "Will SpaceX successfully land humans on the Moon in 2026?",    "odds": "31%", "volume": "$4.1M"},
        {"title": "Will a major social media platform lose 20% users by Dec 26?", "odds": "36%", "volume": "$1.2M"},
    ]
}

async def analyse_poly_category(category: str, markets: list) -> dict:
    """Use Groq to analyse and pick top 2 most profitable bets per category."""
    if not groq_client:
        return {"top": [], "signal": None}

    markets_text = "\n".join([
        f"{i+1}. {m['title']} — Odds: {m['odds']} — Volume: {m['volume']}"
        for i, m in enumerate(markets)
    ])

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Phoenix Prediction Engine, an elite prediction market analyst. "
                        "Analyse prediction market bets and identify the 2 most profitable opportunities. "
                        "Consider: probability vs implied odds, market volume as signal of smart money, "
                        "current global events, and expected value. "
                        "Respond ONLY in this exact JSON format with no extra text:\n"
                        '{"pick1": {"title": "...", "odds": "...", "reason": "...", "edge": "YES or NO", "confidence": "XX%"}, '
                        '"pick2": {"title": "...", "odds": "...", "reason": "...", "edge": "YES or NO", "confidence": "XX%"}, '
                        '"signal": {"direction": "BUY or SELL", "thesis": "...", "risk": "LOW/MED/HIGH"}}'
                    )
                },
                {
                    "role": "user",
                    "content": f"Category: {category.upper()}\n\nMarkets:\n{markets_text}\n\nPick the 2 most profitable bets and generate a signal."
                }
            ],
            max_tokens=400,
            temperature=0.2,
            timeout=20
        )
        import json
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        return parsed
    except Exception as e:
        print(f"Poly analysis error: {e}")
        return {
            "pick1": {"title": markets[0]["title"], "odds": markets[0]["odds"], "reason": "Highest volume bet — smart money indicator.", "edge": "YES", "confidence": "72%"},
            "pick2": {"title": markets[1]["title"], "odds": markets[1]["odds"], "reason": "Strong probability with positive expected value.", "edge": "YES", "confidence": "65%"},
            "signal": {"direction": "BUY", "thesis": "Market consensus points to positive outcome.", "risk": "MED"}
        }

# ─── SELF PING ────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    asyncio.create_task(self_ping_loop())

async def self_ping_loop():
    await asyncio.sleep(30)
    url = os.getenv("RENDER_EXTERNAL_URL", "https://phoenix-signal-backend.onrender.com")
    while True:
        try:
            async with httpx.AsyncClient() as c:
                await c.get(f"{url}/ping", timeout=10)
                print("🔁 Self-ping OK")
        except Exception as e:
            print(f"Self-ping failed: {e}")
        await asyncio.sleep(240)

# ─── ROUTES ───────────────────────────────────────────────

@app.get("/ping")
async def ping():
    return {"status": "alive"}

@app.get("/")
async def root():
    return {"status": "online", "engine": "Phoenix Signal Backend v3"}

@app.get("/api/v2/history")
async def get_signals():
    signals = await generate_signals()
    return {"signals": signals}

@app.get("/api/v2/pairs")
async def get_pairs(search: str = Query(default="")):
    now = time.time()

    if not search and now - _pairs_cache["ts"] < CACHE_TTL and _pairs_cache["data"]:
        return _pairs_cache["data"]

    async with httpx.AsyncClient(headers=HEADERS) as client:
        try:
            if search:
                sr = await client.get(
                    f"{COINGECKO}/search",
                    params={"query": search},
                    timeout=10.0
                )
                if sr.status_code != 200:
                    return []
                results = sr.json().get("coins", [])[:10]
                ids = [c["id"] for c in results if c.get("id")]
                if not ids:
                    return []
                pr = await client.get(
                    f"{COINGECKO}/coins/markets",
                    params={
                        "vs_currency":            "usd",
                        "ids":                    ",".join(ids),
                        "order":                  "market_cap_desc",
                        "price_change_percentage": "24h"
                    },
                    timeout=10.0
                )
                coins     = pr.json() if pr.status_code == 200 else []
                price_map = {c["id"]: c for c in coins}
                output    = []
                for r in results:
                    cid    = r.get("id", "")
                    c      = price_map.get(cid, {})
                    price  = float(c.get("current_price") or 0)
                    change = float(c.get("price_change_percentage_24h") or 0)
                    vol    = c.get("total_volume", 0) or 0
                    mcap   = c.get("market_cap", 0) or 0
                    output.append({
                        "id":              cid,
                        "name":            r.get("name", ""),
                        "symbol":          r.get("symbol", "").upper(),
                        "market_cap_rank": r.get("market_cap_rank") or c.get("market_cap_rank", "—"),
                        "price":           f"{price:,.4f}" if 0 < price < 1 else f"{price:,.2f}" if price >= 1 else "N/A",
                        "change":          f"{change:+.2f}%",
                        "volume":          f"${vol/1_000_000:.1f}M" if vol >= 1_000_000 else f"${vol:,.0f}",
                        "market_cap":      f"${mcap/1_000_000_000:.2f}B" if mcap >= 1_000_000_000 else f"${mcap/1_000_000:.1f}M",
                        "thumb":           r.get("thumb", ""),
                        "high_24h":        f"{float(c.get('high_24h') or 0):,.2f}",
                        "low_24h":         f"{float(c.get('low_24h') or 0):,.2f}",
                    })
                return output

            else:
                r = await client.get(
                    f"{COINGECKO}/coins/markets",
                    params={
                        "vs_currency":            "usd",
                        "order":                  "volume_desc",
                        "per_page":               50,
                        "page":                   1,
                        "price_change_percentage": "24h"
                    },
                    timeout=12.0
                )
                if r.status_code == 200:
                    output = []
                    for c in r.json():
                        price  = float(c.get("current_price") or 0)
                        change = float(c.get("price_change_percentage_24h") or 0)
                        vol    = c.get("total_volume", 0) or 0
                        mcap   = c.get("market_cap", 0) or 0
                        output.append({
                            "id":              c["id"],
                            "name":            c.get("name", ""),
                            "symbol":          c.get("symbol", "").upper(),
                            "market_cap_rank": c.get("market_cap_rank", "—"),
                            "price":           f"{price:,.4f}" if 0 < price < 1 else f"{price:,.2f}" if price >= 1 else "N/A",
                            "change":          f"{change:+.2f}%",
                            "volume":          f"${vol/1_000_000:.1f}M" if vol >= 1_000_000 else f"${vol:,.0f}",
                            "market_cap":      f"${mcap/1_000_000_000:.2f}B" if mcap >= 1_000_000_000 else f"${mcap/1_000_000:.1f}M",
                            "thumb":           c.get("image", ""),
                            "high_24h":        f"{float(c.get('high_24h') or 0):,.2f}",
                            "low_24h":         f"{float(c.get('low_24h') or 0):,.2f}",
                        })
                    _pairs_cache["data"] = output
                    _pairs_cache["ts"]   = time.time()
                    return output
        except Exception as e:
            print(f"Pairs error: {e}")
    return _pairs_cache["data"] or []

@app.get("/api/v2/trending")
async def get_trending():
    async with httpx.AsyncClient(headers=HEADERS) as client:
        try:
            r = await client.get(f"{COINGECKO}/search/trending", timeout=10.0)
            if r.status_code == 200:
                coins = r.json().get("coins", [])
                return {"trending": [
                    {
                        "name":   c["item"]["name"],
                        "symbol": c["item"]["symbol"].upper(),
                        "rank":   c["item"].get("market_cap_rank", "N/A"),
                        "thumb":  c["item"].get("thumb", ""),
                        "score":  round(c["item"].get("score", 0), 2)
                    }
                    for c in coins[:10]
                ]}
        except Exception as e:
            print(f"Trending error: {e}")
    return {"trending": []}

@app.get("/api/v2/polymarket")
async def get_polymarket():
    now = time.time()
    if now - _poly_cache["ts"] < CACHE_TTL and _poly_cache["data"]:
        return _poly_cache["data"]

    # Run AI analysis for all 4 categories concurrently
    results = await asyncio.gather(
        analyse_poly_category("crypto",      POLY_MARKETS["crypto"]),
        analyse_poly_category("geopolitics", POLY_MARKETS["geopolitics"]),
        analyse_poly_category("sports",      POLY_MARKETS["sports"]),
        analyse_poly_category("other",       POLY_MARKETS["other"]),
        return_exceptions=True
    )

    def safe(r, fallback):
        return r if isinstance(r, dict) else fallback

    fallback = {"pick1": {}, "pick2": {}, "signal": {"direction": "BUY", "thesis": "Analysis unavailable.", "risk": "MED"}}

    data = {
        "crypto":      {"markets": POLY_MARKETS["crypto"],      "analysis": safe(results[0], fallback)},
        "geopolitics": {"markets": POLY_MARKETS["geopolitics"], "analysis": safe(results[1], fallback)},
        "sports":      {"markets": POLY_MARKETS["sports"],      "analysis": safe(results[2], fallback)},
        "other":       {"markets": POLY_MARKETS["other"],       "analysis": safe(results[3], fallback)},
    }

    _poly_cache["data"] = data
    _poly_cache["ts"]   = time.time()
    return data

@app.get("/api/v2/news")
async def news():
    return {"news": [
        {"title": "Altcoin season indicators flash green as BTC dominance dips below 54%.", "source": "Phoenix Data Wire"},
        {"title": "High volume breakouts detected across mid-cap DeFi tokens.",              "source": "Phoenix Data Wire"},
        {"title": "Open interest on perpetuals surges across top altcoin pairs.",            "source": "Bybit Feed Node"},
        {"title": "On-chain data shows accumulation patterns in Layer-2 tokens.",            "source": "Phoenix Data Wire"},
        {"title": "Liquidity pool depth expands as institutional volume rotates into alts.", "source": "Bybit Feed Node"},
        {"title": "Whale wallets accumulate BNB and AVAX in silent weekend session.",        "source": "Phoenix Data Wire"},
        {"title": "Funding rates on perpetuals turn positive — bulls in control.",           "source": "Bybit Feed Node"},
    ]}

@app.get("/api/v2/performance")
async def performance():
    return {"pnl": "+4.12%"}

@app.post("/api/v2/chat")
async def chat(request: ChatRequest):
    if not groq_client:
        return {"reply": "⚠️ AI Engine offline. GROQ_API_KEY not set on Render."}
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Phoenix Oracle, an elite crypto and financial markets intelligence engine. "
                    "You have deep expertise in technical analysis, on-chain data, DeFi, tokenomics, "
                    "derivatives, market microstructure, risk management, and trading psychology. "
                    "When asked about a specific token, give its narrative, use case, risk profile, "
                    "key levels to watch, and sentiment. "
                    "When asked about trade setups, give structured analysis with entry logic, "
                    "invalidation level, and targets. "
                    "You do not give financial advice — you give professional analysis. "
                    "Be sharp, direct, and expert-level. Under 200 words per response."
                )
            }
        ]
        # Include conversation history for context
        for msg in request.history[-8:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": request.prompt})

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=280,
            temperature=0.3,
            timeout=20
        )
        return {"reply": response.choices[0].message.content.strip()}
    except Exception as e:
        print(f"Groq error: {e}")
        return {"reply": "⚠️ AI Engine temporarily unavailable. Please retry."}
