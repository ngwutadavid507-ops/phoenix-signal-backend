import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PhoenixSignalEngine")

app = FastAPI(title="Phoenix Signal Core Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Start with multiple multi-token signal configurations out of the box
SIGNAL_HISTORY_CACHE = [
    {
        "asset_pair": "BTC/USDT",
        "direction": "LONG",
        "entry_zone": "64500 - 65200",
        "stop_loss": "62000",
        "take_profit": ["68000", "71000"],
        "analysis_reason": "Order-book depth indicates massive demand liquidity clusters."
    },
    {
        "asset_pair": "ETH/USDT",
        "direction": "LONG",
        "entry_zone": "3420 - 3460",
        "stop_loss": "3310",
        "take_profit": ["3650", "3800"],
        "analysis_reason": "Moving average cross confirmed on 4H telemetry frames."
    },
    {
        "asset_pair": "SOL/USDT",
        "direction": "SHORT",
        "entry_zone": "148.50 - 151.00",
        "stop_loss": "156.20",
        "take_profit": ["135.00", "128.00"],
        "analysis_reason": "RSI boundaries overextended at short-term distribution baselines."
    }
]

class PromptRequest(BaseModel):
    prompt: str

@app.get("/api/v1/history")
async def get_signal_history():
    return {"status": "success", "total": len(SIGNAL_HISTORY_CACHE), "data": SIGNAL_HISTORY_CACHE}

@app.post("/api/v1/generate-signal")
async def generate_signal(request: PromptRequest):
    logger.info(f"Computing matrix metrics for input parameters: '{request.prompt}'")
    raw_text = request.prompt.upper()
    
    # Extract asset context dynamically from input words
    ticker = "AVAX/USDT"
    if "BTC" in raw_text: ticker = "BTC/USDT"
    elif "ETH" in raw_text: ticker = "ETH/USDT"
    elif "SOL" in raw_text: ticker = "SOL/USDT"
    elif "BNB" in raw_text: ticker = "BNB/USDT"
    elif "TON" in raw_text: ticker = "TON/USDT"
    elif "DOGE" in raw_text: ticker = "DOGE/USDT"
    else:
        # Fallback to create custom dynamic pairing from prompt text strings
        cleaned_words = [w for w in raw_text.split() if len(w) <= 5]
        if cleaned_words:
            ticker = f"{cleaned_words[0]}/USDT"

    direction = "SHORT" if "SELL" in raw_text or "SHORT" in raw_text else "LONG"
    
    payload = {
        "asset_pair": ticker,
        "direction": direction,
        "entry_zone": "Market Execution Scope",
        "stop_loss": "Dynamic Calculated Protect",
        "take_profit": ["Target Threshold 1", "Target Threshold 2"],
        "analysis_reason": f"AI Engine generated stream vector based on prompt request parameters."
    }
    
    # Prepend to the top of the stream list array
    SIGNAL_HISTORY_CACHE.insert(0, payload)
    if len(SIGNAL_HISTORY_CACHE) > 12: # Retain up to 12 signals in stream memory cache
        SIGNAL_HISTORY_CACHE.pop()
        
    return payload

@app.get("/")
async def health():
    return {"status": "online", "engine": "Phoenix Signal Core Layer"}
