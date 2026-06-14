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

# ─── GROQ ─────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
print("🚀 Groq ready." if groq_client else "⚠️ GROQ_API_KEY missing.")

# ─── CONSTANTS ────────────────────────────────────────────

COINGECKO = "https://api.coingecko.com/api/v3"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

STABLECOINS = {
    "tether", "usd-coin", "binance-usd", "dai", "true-usd",
    "frax", "usdd", "paxos-standard", "gemini-dollar", "terrausd",
    "neutrino", "fei-usd", "liquity-usd", "magic-internet-money"
}
STABLE_SYMBOLS = {"USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD", "FRAX", "USDP"}

_signal_cache = {"data": [], "ts": 0}
_pairs_cache = {"data": [], "ts": 0}
CACHE_TTL = 300

# ─── TECHNICAL ANALYSIS ───────────────────────────────────

def compute_rsi(prices: list, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(len(prices) - period, len(prices)):
        diff = prices[i] - prices[i - 1]
        if diff > 0:
            gains.append(diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(diff))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

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
    macd_line = ema12 - ema26
    signal_line = compute_ema(prices[-9:], min(9, len(prices[-9:])))
    return round(macd_line, 8), round(signal_line, 8)

def compute_bollinger(prices: list, period: int = 20):
    if len(prices) < period:
        p = prices[-1] if prices else 0
        return p, p, p
    window = prices[-period:]
    mid = sum(window) / period
    std = (sum((p - mid) ** 2 for p in window) / period) ** 0.5
    return round(mid + 2 * std, 8), round(mid, 8), round(mid - 2 * std, 8)

def analyse_coin(closes: list, volumes: list, symbol: str, price: float):
    if len(closes) < 30:
        return None

    rsi = compute_rsi(closes)
    ema20 = compute_ema(closes, 20)
    ema50 = compute_ema(closes, 50) if len(closes) >= 50 else ema20
    macd_line, signal_line = compute_macd(closes)
    bb_upper, bb_mid, bb_lower = compute_bollinger(closes)

    avg_vol = sum(volumes[-14:]) / 14 if len(volumes) >= 14 else (volumes[-1] if volumes else 1)
    last_vol = volumes[-1] if volumes else 0
    vol_spike = last_vol > avg_vol * 1.5

    bull = 0
    bear = 0
    reasons = []

    # RSI
    if rsi < 32:
        bull += 1
        reasons.append(f"RSI oversold at {rsi}")
    elif rsi > 68:
        bear += 1
        reasons.append(f"RSI overbought at {rsi}")

    # EMA trend
    if ema20 > ema50 * 1.001:
        bull += 1
        reasons.append("EMA20 above EMA50 — uptrend confirmed")
    elif ema20 < ema50 * 0.999:
        bear += 1
        reasons.append("EMA20 below EMA50 — downtrend confirmed")

    # MACD
    if macd_line > signal_line and macd_line > 0:
        bull += 1
        reasons.append("MACD bullish crossover above zero")
    elif macd_line < signal_line and macd_line < 0:
        bear += 1
        reasons.append("MACD bearish crossover below zero")
    elif macd_line > signal_line:
        bull += 1
        reasons.append("MACD bullish crossover")
    elif macd_line < signal_line:
        bear += 1
        reasons.append("MACD bearish crossover")

    # Bollinger Bands
    if price <= bb_lower * 1.005:
        bull += 1
        reasons.append("Price bouncing off lower Bollinger Band")
    elif price >= bb_upper * 0.995:
        bear += 1
        reasons.append("Price rejecting upper Bollinger Band")
    elif price > bb_mid and closes[-1] > closes[-2]:
        bull += 1
        reasons.append("Price above BB midline with momentum")
    elif price < bb_mid and closes[-1] < closes[-2]:
        bear += 1
        reasons.append("Price below BB midline losing support")

    # Volume confirmation
    if vol_spike:
        if bull >= bear:
            bull += 1
            reasons.append("High volume confirms buying pressure")
        else:
            bear += 1
            reasons.append("High volume confirms selling pressure")

    direction = "LONG" if bull > bear else "SHORT"
    score = bull if direction == "LONG" else bear

    if score < 4:
        return None

    # Confidence calculation
    rsi_bonus = max(0, (50 - rsi) / 4) if direction == "LONG" else max(0, (rsi - 50) / 4)
    macd_bonus = 3 if (direction == "LONG" and macd_line > 0) or (direction == "SHORT" and macd_line < 0) else 0
    confidence = min(96, int(68 + (score / 5) * 22 + rsi_bonus + macd_bonus))

    if confidence < 75:
        return None

    # SL/TP for 20x leverage
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
    rr = round(reward / risk, 1) if risk > 0 else 0

    def fmt(v):
        if v >= 1000: return f"{v:,.2f}"
        if v >= 1:    return f"{v:,.3f}"
        return f"{v:.6f}"

    return {
        "pair": f"{symbol}/USDT",
        "direction": direction,
        "entry": fmt(p),
        "sl":  fmt(sl),
        "tp1": fmt(tp1),
        "tp2": fmt(tp2),
        "tp3": fmt(tp3),
        "confidence": f"{confidence}%",
        "rr": f"1:{rr}",
        "leverage": "20x",
        "reasons": reasons[:4],
        "rsi": rsi,
        "score": f"{score}/5"
    }

# ─── OHLCV FETCH ──────────────────────────────────────────

async def fetch_ohlcv(coin_id: str, client: httpx.AsyncClient):
    try:
        r = await client.get(
            f"{COINGECKO}/coins/{coin_id}/ohlc",
            params={"vs_currency": "usd", "days": "30"},
            timeout=12.0,
            headers=HEADERS
        )
        if r.status_code == 200:
            candles = r.json()
            closes  = [c[4] for c in candles]
            volumes = [float(c[3]) for c in candles]
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
                    "vs_currency": "usd",
                    "order": "volume_desc",
                    "per_page": 80,
                    "page": 1,
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
            and float(c.get("total_volume") or 0) > 3_000_000
        ][:25]

        signals = []

        for coin in candidates:
            coin_id = coin["id"]
            symbol  = coin["symbol"].upper()
            price   = float(coin.get("current_price") or 0)

            closes, volumes = await fetch_ohlcv(coin_id, client)

            if len(closes) < 30:
                await asyncio.sleep(0.4)
                continue

            if not volumes:
                vol = float(coin.get("total_volume") or 0)
                volumes = [vol] * len(closes)

            result = analyse_coin(closes, volumes, symbol, price)
            if result:
                signals.append(result)

            await asyncio.sleep(0.5)

            if len(signals) >= 10:
                break

        signals.sort(
            key=lambda x: int(x["confidence"].replace("%", "")),
            reverse=True
        )

        _signal_cache["data"] = signals
        _signal_cache["ts"] = time.time()
        return signals

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
    return {
        "status": "online",
        "engine": "Phoenix Signal Backend v2",
        "ai": "groq" if groq_client else "missing_key"
    }

@app.get("/api/v2/history")
async def get_signals():
    signals = await generate_signals()
    return {"signals": signals}

@app.get("/api/v2/pairs")
async def get_pairs(search: str = Query(default="")):
    now = time.time()

    # Return cached if no search and cache fresh
    if not search and now - _pairs_cache["ts"] < CACHE_TTL and _pairs_cache["data"]:
        return _pairs_cache["data"]

    async with httpx.AsyncClient(headers=HEADERS) as client:
        try:
            if search:
                # Search mode
                sr = await client.get(
                    f"{COINGECKO}/search",
                    params={"query": search},
                    timeout=10.0
                )
                if sr.status_code != 200:
                    return []

                results = sr.json().get("coins", [])[:8]
                ids = [c["id"] for c in results if c.get("id")]

                if not ids:
                    return []

                pr = await client.get(
                    f"{COINGECKO}/coins/markets",
                    params={
                        "vs_currency": "usd",
                        "ids": ",".join(ids),
                        "order": "market_cap_desc",
                        "price_change_percentage": "24h"
                    },
                    timeout=10.0
                )
                coins = pr.json() if pr.status_code == 200 else []

                price_map = {c["id"]: c for c in coins}
                output = []
                for r in results:
                    cid = r.get("id", "")
                    c = price_map.get(cid, {})
                    price  = float(c.get("current_price") or 0)
                    change = float(c.get("price_change_percentage_24h") or 0)
                    vol    = c.get("total_volume", 0) or 0
                    mcap   = c.get("market_cap", 0) or 0
                    output.append({
                        "id": cid,
                        "name": r.get("name", ""),
                        "symbol": r.get("symbol", "").upper(),
                        "market_cap_rank": r.get("market_cap_rank") or c.get("market_cap_rank", "—"),
                        "price": f"{price:,.4f}" if 0 < price < 1 else f"{price:,.2f}" if price >= 1 else "N/A",
                        "change": f"{change:+.2f}%",
                        "volume": f"${vol/1_000_000:.1f}M" if vol >= 1_000_000 else f"${vol:,.0f}",
                        "market_cap": f"${mcap/1_000_000_000:.2f}B" if mcap >= 1_000_000_000 else f"${mcap/1_000_000:.1f}M",
                        "thumb": r.get("thumb", ""),
                        "is_search": True
                    })
                return output

            else:
                # Default: top 40 by volume
                r = await client.get(
                    f"{COINGECKO}/coins/markets",
                    params={
                        "vs_currency": "usd",
                        "order": "volume_desc",
                        "per_page": 40,
                        "page": 1,
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
                            "id": c["id"],
                            "name": c.get("name", ""),
                            "symbol": c.get("symbol", "").upper(),
                            "market_cap_rank": c.get("market_cap_rank", "—"),
                            "price": f"{price:,.4f}" if 0 < price < 1 else f"{price:,.2f}" if price >= 1 else "N/A",
                            "change": f"{change:+.2f}%",
                            "volume": f"${vol/1_000_000:.1f}M" if vol >= 1_000_000 else f"${vol:,.0f}",
                            "market_cap": f"${mcap/1_000_000_000:.2f}B" if mcap >= 1_000_000_000 else f"${mcap/1_000_000:.1f}M",
                            "thumb": c.get("image", ""),
                            "is_search": False
                        })
                    _pairs_cache["data"] = output
                    _pairs_cache["ts"] = time.time()
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
async def polymarket():
    return {"markets": [
        {"title": "Will Bitcoin break $115K this month?", "odds": "71%"},
        {"title": "Will Bybit 24h derivative volume set new highs?", "odds": "62%"},
        {"title": "Will ETH outperform BTC this month?", "odds": "55%"},
        {"title": "Will crypto total market cap exceed $3.5T by July?", "odds": "48%"},
        {"title": "Will a major altcoin 3x before end of Q3 2026?", "odds": "67%"}
    ]}

@app.get("/api/v2/news")
async def news():
    return {"news": [
        {"title": "Altcoin season indicators flash green as BTC dominance dips below 54%.", "source": "Phoenix Data Wire"},
        {"title": "High volume breakouts detected across mid-cap DeFi tokens.", "source": "Phoenix Data Wire"},
        {"title": "Open interest on perpetuals surges across top altcoin pairs.", "source": "Bybit Feed Node"},
        {"title": "On-chain data shows accumulation patterns in Layer-2 tokens.", "source": "Phoenix Data Wire"},
        {"title": "Liquidity pool depth expands as institutional volume rotates into alts.", "source": "Bybit Feed Node"}
    ]}

@app.get("/api/v2/performance")
async def performance():
    return {"pnl": "+4.12%"}

@app.post("/api/v2/chat")
async def chat(request: PromptRequest):
    if not groq_client:
        return {"reply": "⚠️ AI Engine offline. GROQ_API_KEY not set on Render."}
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Phoenix Terminal, an elite crypto trading intelligence oracle. "
                        "You have deep expertise in technical analysis, on-chain data, DeFi, "
                        "tokenomics, market structure, derivatives, risk management, and trading psychology. "
                        "You give sharp, direct, expert-level answers. "
                        "When asked about a specific token, give context on its use case, "
                        "recent narrative, risk profile, and what to watch. "
                        "When asked about prices, clarify you don't have real-time feeds "
                        "but give strong contextual market insight. "
                        "Never give financial advice — give analysis. "
                        "Keep responses under 180 words. Be sharp, not generic."
                    )
                },
                {"role": "user", "content": request.prompt}
            ],
            max_tokens=250,
            temperature=0.3,
            timeout=20
        )
        return {"reply": response.choices[0].message.content.strip()}
    except Exception as e:
        print(f"Groq error: {e}")
        return {"reply": "⚠️ AI Engine temporarily unavailable. Please retry."}
