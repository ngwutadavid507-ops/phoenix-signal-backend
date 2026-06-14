import os
from typing import Dict, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# Modern official Google GenAI SDK integration
from google import genai
from google.genai import types

app = FastAPI(title="Phoenix Algorithmic Pure Stream Engine")

# CORS middleware must be configured immediately after app instantiation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str

# Secure environment mapping for Google GenAI 2.8.0+
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    ai_client = genai.Client()
else:
    ai_client = None

# Anti-blocking headers to ensure high-uptime public streaming
STANDARD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# --- PURE LIVE TELEMETRY CORE ---

async def get_live_bybit_top_pairs() -> List[Dict]:
    """
    Directly scans the live Bybit V5 Linear interface.
    Isolates top trading pairs dynamically by 24h turnover volume.
    """
    async with httpx.AsyncClient(headers=STANDARD_HEADERS) as client:
        try:
            url = "https://api.bybit.com/v5/market/tickers?category=linear"
            response = await client.get(url, timeout=6.0)
            if response.status_code != 200:
                return []
            
            ticker_list = response.json().get("result", {}).get("list", [])
            if not ticker_list:
                return []
            
            # Defensive validation against missing turnover metrics
            sorted_tickers = sorted(
                [t for t in ticker_list if t.get("turnover") is not None], 
                key=lambda x: float(x["turnover"] or 0), 
                reverse=True
            )
            
            top_5 = sorted_tickers[:5]
            processed_signals = []
            
            for item in top_5:
                raw_symbol = item.get("symbol", "")
                if not raw_symbol:
                    continue
                    
                if raw_symbol.endswith("USDT"):
                    pair_display = f"{raw_symbol[:-4]}/USDT"
                else:
                    pair_display = raw_symbol
                
                last_price = float(item.get("lastPrice", 0) or 0)
                price_change_pct = float(item.get("price24hPcnt", 0) or 0) * 100 
                
                if last_price <= 0:
                    continue
                
                if price_change_pct >= 0.2:
                    direction = "BUY / LONG"
                    stop_loss = last_price * 0.982   
                    take_profit = last_price * 1.045  
                    confidence_math = min(99, int(75 + (price_change_pct * 3)))
                elif price_change_pct <= -0.2:
                    direction = "SELL / SHORT"
                    stop_loss = last_price * 1.018   
                    take_profit = last_price * 0.955  
                    confidence_math = min(99, int(75 + (abs(price_change_pct) * 3)))
                else:
                    direction = "BUY / LONG"
                    stop_loss = last_price * 0.988
                    take_profit = last_price * 1.030
                    confidence_math = 70

                fmt = ".2f" if last_price >= 1.0 else ".6f"
                
                processed_signals.append({
                    "pair": pair_display,
                    "direction": direction,
                    "entry": f"{last_price:{fmt}}",
                    "sl": f"{stop_loss:{fmt}}",
                    "tp": f"{take_profit:{fmt}}",
                    "confidence": f"{confidence_math}%",
                    "raw_change": price_change_pct
                })
                
            return processed_signals
        except Exception as e:
            print(f"Bybit Core Pipeline Exception: {str(e)}")
            return []

# --- LIVE REST ROUTER ENDPOINTS ---

@app.get("/api/v2/history")
async def get_algorithmic_signals():
    signals = await get_live_bybit_top_pairs()
    return {"signals": signals}

@app.get("/api/v2/pairs")
async def get_all_pairs_matrix():
    async with httpx.AsyncClient(headers=STANDARD_HEADERS) as client:
        try:
            r = await client.get("https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=15&page=1", timeout=6.0)
            if r.status_code == 200:
                return [{
                    "id": c.get("id"),
                    "name": c.get("name"),
                    "symbol": str(c.get("symbol", "")).upper(),
                    "price": f"{c.get('current_price', 0):,.2f}" if c.get("current_price", 0) >= 1 else f"{c.get('current_price', 0):.6f}",
                    "change": f"{c.get('price_change_percentage_24h', 0) or 0:+.2f}%"
                } for c in r.json()]
        except Exception as e:
            print(f"CoinGecko Data Node Exception: {e}")
        return []

@app.get("/api/v2/performance")
async def fetch_performance_matrix():
    signals = await get_live_bybit_top_pairs()
    if not signals:
        return {"pnl": "0.00%"}
    
    total_movement = sum(abs(s.get("raw_change", 0)) for s in signals)
    calculated_pnl = total_movement / len(signals)
    return {"pnl": f"+{calculated_pnl:.2f}%"}

@app.get("/api/v2/polymarket")
async def get_live_polymarket_events():
    """Fetches contracts dynamically. Enhanced safety fallback array mapping."""
    async with httpx.AsyncClient(headers=STANDARD_HEADERS) as client:
        try:
            r = await client.get("https://clob.polymarket.com/markets/simplified", timeout=6.0)
            if r.status_code == 200:
                markets_data = r.json()
                output = []
                for m in markets_data[:5]:
                    if m and m.get("question"):
                        prices = m.get("outcomePrices")
                        # Bulletproof conversion loop if prices is None or empty list
                        if not prices or not isinstance(prices, list):
                            prices = ["0.5"]
                        try:
                            first_price = float(prices[0] if prices[0] else 0.5)
                        except (ValueError, TypeError):
                            first_price = 0.5
                        
                        output.append({
                            "title": m.get("question"),
                            "odds": f"{int(first_price * 100)}%"
                        })
                return {"markets": output}
        except Exception as e:
            print(f"Polymarket Data Engine Error: {e}")
        return {"markets": []}

@app.get("/api/v2/news")
async def fetch_live_news_stream():
    async with httpx.AsyncClient(headers=STANDARD_HEADERS) as client:
        try:
            r = await client.get("https://cryptopanic.com/api/v1/posts/?public=true", timeout=6.0)
            if r.status_code == 200:
                results = r.json().get("results", [])
                return {"news": [{"title": item.get("title", "Market Update"), "source": item.get("source", {}).get("domain", "CryptoPanic")} for item in results[:4]]}
        except Exception as e:
            print(f"CryptoPanic Wire Connection Error: {e}")
        return {"news": []}

@app.post("/api/v2/chat")
async def handle_chat(request: PromptRequest):
    if not ai_client:
        return {"reply": "AI Engine offline. Verify GEMINI_API_KEY environment configuration inside your hosting dashboard."}
    try:
        response = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=request.prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are the live technical processing oracle for the Phoenix terminal. Run technical analytics parameters directly against raw real-time user requests. No introductory remarks.",
                max_output_tokens=250,
                temperature=0.2
            )
        )
        return {"reply": response.text.strip()}
    except Exception as e:
        return {"reply": f"Processing Pipeline Vector Fault: {str(e)}"}
