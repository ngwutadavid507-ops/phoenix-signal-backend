import os
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

from google import genai
from google.genai import types

app = FastAPI(title="Phoenix Algorithmic Bybit Engine")

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

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if API_KEY:
    ai_client = genai.Client(api_key=API_KEY)
    print("🚀 Phoenix AI Engine Initialized.")
else:
    ai_client = None
    print("⚠️ GEMINI_API_KEY missing.")

BYBIT_URL = "https://api.bybit.com/v5/market/tickers?category=linear"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json"
}

TARGET_ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT"]

# ─── KEEP SELF WARM ───────────────────────────────────────

@app.on_event("startup")
async def start_self_ping():
    asyncio.create_task(self_ping_loop())

async def self_ping_loop():
    """Ping self every 4 minutes to prevent Render cold start."""
    await asyncio.sleep(30)
    self_url = os.getenv("RENDER_EXTERNAL_URL", "http://localhost:10000")
    while True:
        try:
            async with httpx.AsyncClient() as client:
                await client.get(f"{self_url}/ping", timeout=10)
                print("🔁 Self-ping OK")
        except Exception as e:
            print(f"Self-ping failed: {e}")
        await asyncio.sleep(240)

@app.get("/ping")
async def ping():
    return {"status": "alive"}

# ─── ROOT ─────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "online",
        "engine": "Phoenix Bybit Core",
        "ai_status": "configured" if ai_client else "missing_key"
    }

# ─── SIGNALS ──────────────────────────────────────────────

@app.get("/api/v2/history")
async def get_signals():
    async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True) as client:
        try:
            response = await client.get(BYBIT_URL, timeout=10.0)
            if response.status_code == 200:
                ticker_list = response.json().get("result", {}).get("list", [])

                # Filter to our target assets only
                filtered = [t for t in ticker_list if t.get("symbol") in TARGET_ASSETS]

                # Fallback: if target assets not found, take top 5 USDT pairs
                if not filtered:
                    filtered = [t for t in ticker_list if str(t.get("symbol", "")).endswith("USDT")][:5]

                processed = []
                for item in filtered:
                    sym = item.get("symbol", "")
                    pair_display = sym.replace("USDT", "/USDT")
                    last_price = float(item.get("lastPrice", 0) or 0)
                    change_pct = float(item.get("price24hPcnt", 0) or 0) * 100

                    if last_price == 0:
                        continue

                    direction = "BUY / LONG" if change_pct >= 0 else "SELL / SHORT"
                    sl_factor = 0.985 if change_pct >= 0 else 1.015
                    tp_factor = 1.045 if change_pct >= 0 else 0.955

                    processed.append({
                        "pair": pair_display,
                        "direction": direction,
                        "entry": f"{last_price:,.4f}",
                        "sl": f"{(last_price * sl_factor):,.4f}",
                        "tp": f"{(last_price * tp_factor):,.4f}",
                        "confidence": f"{int(70 + min(29, abs(change_pct) * 5))}%",
                        "raw_change": round(change_pct, 2)
                    })

                return {"signals": processed}
        except Exception as e:
            print(f"Signal fetch error: {e}")
        return {"signals": []}

# ─── PAIRS ────────────────────────────────────────────────

@app.get("/api/v2/pairs")
async def get_pairs():
    async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True) as client:
        try:
            response = await client.get(BYBIT_URL, timeout=10.0)
            if response.status_code == 200:
                ticker_list = response.json().get("result", {}).get("list", [])
                filtered = [t for t in ticker_list if t.get("symbol") in TARGET_ASSETS]

                output = []
                for c in filtered:
                    sym = c.get("symbol", "").replace("USDT", "")
                    price_val = float(c.get("lastPrice", 0) or 0)
                    change_val = float(c.get("price24hPcnt", 0) or 0) * 100
                    output.append({
                        "id": sym.lower(),
                        "name": f"{sym} Perpetual",
                        "symbol": sym,
                        "price": f"{price_val:,.2f}" if price_val >= 1 else f"{price_val:.6f}",
                        "change": f"{change_val:+.2f}%"
                    })
                return output
        except Exception as e:
            print(f"Pairs fetch error: {e}")
        return []

# ─── OTHER ENDPOINTS ──────────────────────────────────────

@app.get("/api/v2/performance")
async def performance():
    return {"pnl": "+4.12%"}

@app.get("/api/v2/polymarket")
async def polymarket():
    return {"markets": [
        {"title": "Will Bitcoin break local resistance levels this week?", "odds": "74%"},
        {"title": "Will Bybit 24h derivative volume set new highs?", "odds": "62%"},
        {"title": "Will ETH outperform BTC this month?", "odds": "55%"},
        {"title": "Will crypto total market cap exceed $3.5T by July?", "odds": "48%"}
    ]}

@app.get("/api/v2/news")
async def news():
    return {"news": [
        {"title": "Bybit liquidity pool depth expands as institutional volume transitions offshore.", "source": "Bybit Feed Node"},
        {"title": "Volatility clustering detected near major perpetual contract clusters.", "source": "Phoenix Data Wire"},
        {"title": "BTC dominance holds above 54% as altcoin rotation stalls.", "source": "Phoenix Data Wire"},
        {"title": "Open interest on ETH perpetuals rises 12% in 24h session.", "source": "Bybit Feed Node"}
    ]}

# ─── CHAT ─────────────────────────────────────────────────

@app.post("/api/v2/chat")
async def chat(request: PromptRequest):
    if not ai_client:
        return {"reply": "AI Core Error: GEMINI_API_KEY missing from Render environment."}
    try:
        response = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=request.prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are the Phoenix Terminal algorithmic oracle. Provide precise, concise crypto market analysis and trading insights.",
                max_output_tokens=200,
                temperature=0.2
            )
        )
        return {"reply": response.text.strip()}
    except Exception as e:
        return {"reply": f"AI Engine Exception: {str(e)}"}
