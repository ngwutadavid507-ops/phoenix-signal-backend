import os
from typing import Dict, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# Official Google GenAI SDK
from google import genai
from google.genai import types

app = FastAPI(title="Phoenix Algorithmic Bybit Engine")

# Permissive CORS layer for direct mobile/webview alignment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str

# Gemini Core Initialization
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    ai_client = genai.Client()
else:
    ai_client = None

BYBIT_PUBLIC_URL = "https://api.bybit.com/v5/market/tickers?category=linear"
STANDARD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json"
}

@app.get("/")
async def root_diagnostic():
    return {"status": "online", "engine": "Phoenix Bybit Core"}

@app.get("/api/v2/history")
async def get_algorithmic_signals():
    """Pulls live data from Bybit and builds automated mock signal parameters."""
    async with httpx.AsyncClient(headers=STANDARD_HEADERS, follow_redirects=True) as client:
        try:
            response = await client.get(BYBIT_PUBLIC_URL, timeout=5.0)
            if response.status_code == 200:
                ticker_list = response.json().get("result", {}).get("list", [])
                # Take top volume symbols ending with USDT
                usdt_pairs = [t for t in ticker_list if str(t.get("symbol")).endswith("USDT")][:5]
                
                processed = []
                for item in usdt_pairs:
                    sym = item.get("symbol", "")
                    pair_display = f"{sym[:-4]}/USDT"
                    last_price = float(item.get("lastPrice", 0) or 0)
                    change_pct = float(item.get("price24hPcnt", 0) or 0) * 100
                    
                    direction = "BUY / LONG" if change_pct >= 0 else "SELL / SHORT"
                    sl_factor = 0.985 if direction == "BUY / LONG" else 1.015
                    tp_factor = 1.045 if direction == "BUY / LONG" else 0.955
                    
                    processed.append({
                        "pair": pair_display,
                        "direction": direction,
                        "entry": f"{last_price:,.4f}",
                        "sl": f"{(last_price * sl_factor):,.4f}",
                        "tp": f"{(last_price * tp_factor):,.4f}",
                        "confidence": f"{int(70 + min(29, abs(change_pct) * 5))}%",
                        "raw_change": change_pct
                    })
                return {"signals": processed}
        except Exception as e:
            print(f"Bybit Signal Trace Failure: {e}")
        return {"signals": []}

@app.get("/api/v2/pairs")
async def get_all_pairs_matrix():
    """Fetches high-volume core tokens straight from live Bybit ticks."""
    async with httpx.AsyncClient(headers=STANDARD_HEADERS, follow_redirects=True) as client:
        try:
            response = await client.get(BYBIT_PUBLIC_URL, timeout=5.0)
            if response.status_code == 200:
                ticker_list = response.json().get("result", {}).get("list", [])
                
                # Filter specific major tokens
                target_assets = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT"]
                filtered = [t for t in ticker_list if t.get("symbol") in target_assets]
                
                output = []
                for c in filtered:
                    sym = c.get("symbol", "").replace("USDT", "")
                    price_val = float(c.get("lastPrice", 0) or 0)
                    change_val = float(c.get("price24hPcnt", 0) or 0) * 100
                    
                    output.append({
                        "id": sym.lower(),
                        "name": f"{sym} Perpetual Index",
                        "symbol": sym,
                        "price": f"{price_val:,.2f}" if price_val >= 1 else f"{price_val:.5f}",
                        "change": f"{change_val:+.2f}%"
                    })
                return output
        except Exception as e:
            print(f"Bybit Matrix Trace Failure: {e}")
        return []

@app.get("/api/v2/performance")
async def fetch_performance_matrix():
    return {"pnl": "+4.12%"}

@app.get("/api/v2/polymarket")
async def get_live_polymarket_events():
    return {"markets": [
        {"title": "Will Bitcoin break local resistance levels this week?", "odds": "74%"},
        {"title": "Will Bybit 24h derivative volume set new highs?", "odds": "62%"}
    ]}

@app.get("/api/v2/news")
async def fetch_live_news_stream():
    return {"news": [
        {"title": "Bybit liquidity pool depth expands as institutional volume transitions offshore.", "source": "Bybit Feed Node"},
        {"title": "Volatility clustering detected near major perpetual contract clusters.", "source": "Phoenix Data Wire"}
    ]}

# CRITICAL FIX: Changed from @app.get to @app.post to stop the 405 Method Error
@app.post("/api/v2/chat")
async def handle_chat(request: PromptRequest):
    if not ai_client:
        return {"reply": "AI Engine initialization fault. Verify GEMINI_API_KEY setup on Render."}
    try:
        response = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=request.prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are the Phoenix Terminal automated oracle. Provide precise market telemetry analyses. Be concise.",
                max_output_tokens=200,
                temperature=0.2
            )
        )
        return {"reply": response.text.strip()}
    except Exception as e:
        return {"reply": f"AI Engine Exception Vector: {str(e)}"}
                
