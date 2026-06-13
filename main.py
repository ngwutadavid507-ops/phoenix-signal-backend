import os
import logging
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uuid

# Configure logging with request context
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PhoenixSignalEngine")

app = FastAPI(
    title="Phoenix Signal Core Backend",
    version="1.0.0",
    description="Production-grade crypto trading signal generation engine"
)

# CORS configuration - restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Data Models ====================

class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500, description="Trading analysis prompt")

class SignalResponse(BaseModel):
    signal_id: str
    asset_pair: str
    direction: str
    entry_zone: str
    stop_loss: str
    take_profit: List[str]
    analysis_reason: str
    confidence_score: float = Field(..., ge=0, le=1)
    generated_at: str
    status: str = "OPEN"

class HistoryResponse(BaseModel):
    status: str
    total: int
    data: List[SignalResponse]

# ==================== Constants ====================

SUPPORTED_TICKERS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "BNB": "BNB/USDT",
    "TON": "TON/USDT",
    "DOGE": "DOGE/USDT",
    "AVAX": "AVAX/USDT",
    "XRP": "XRP/USDT",
    "ADA": "ADA/USDT",
    "LINK": "LINK/USDT",
}

DEFAULT_TICKER = "AVAX/USDT"
MAX_CACHE_SIZE = 50
MAX_PROMPT_LENGTH = 500

# In-memory signal cache (in production, use PostgreSQL/MongoDB)
SIGNAL_HISTORY_CACHE: List[SignalResponse] = [
    SignalResponse(
        signal_id=str(uuid.uuid4()),
        asset_pair="BTC/USDT",
        direction="LONG",
        entry_zone="64500 - 65200",
        stop_loss="62000",
        take_profit=["68000", "71000"],
        analysis_reason="Order-book depth indicates massive demand liquidity clusters.",
        confidence_score=0.78,
        generated_at=datetime.utcnow().isoformat()
    ),
    SignalResponse(
        signal_id=str(uuid.uuid4()),
        asset_pair="ETH/USDT",
        direction="LONG",
        entry_zone="3420 - 3460",
        stop_loss="3310",
        take_profit=["3650", "3800"],
        analysis_reason="Moving average cross confirmed on 4H telemetry frames.",
        confidence_score=0.71,
        generated_at=datetime.utcnow().isoformat()
    ),
    SignalResponse(
        signal_id=str(uuid.uuid4()),
        asset_pair="SOL/USDT",
        direction="SHORT",
        entry_zone="148.50 - 151.00",
        stop_loss="156.20",
        take_profit=["135.00", "128.00"],
        analysis_reason="RSI boundaries overextended at short-term distribution baselines.",
        confidence_score=0.65,
        generated_at=datetime.utcnow().isoformat()
    )
]

# ==================== Signal Generation Logic ====================

def extract_ticker_from_prompt(prompt: str) -> str:
    """Extract asset ticker from prompt with validation."""
    raw_text = prompt.upper().strip()
    
    # Check for known tickers
    for key, value in SUPPORTED_TICKERS.items():
        if key in raw_text:
            logger.info(f"Detected ticker: {value}")
            return value
    
    # Fallback: attempt to extract first 3-5 char word as ticker
    cleaned_words = [w.replace(".", "").replace("/", "") for w in raw_text.split() if 2 <= len(w) <= 5]
    if cleaned_words:
        fallback_ticker = f"{cleaned_words[0]}/USDT"
        logger.warning(f"Using fallback ticker: {fallback_ticker}")
        return fallback_ticker
    
    logger.info(f"No ticker detected, using default: {DEFAULT_TICKER}")
    return DEFAULT_TICKER

def determine_direction(prompt: str) -> str:
    """Determine trading direction (LONG/SHORT) from prompt."""
    raw_text = prompt.upper()
    
    if any(word in raw_text for word in ["SELL", "SHORT", "BEARISH", "DOWN", "DECLINE", "DUMP"]):
        return "SHORT"
    elif any(word in raw_text for word in ["BUY", "LONG", "BULLISH", "UP", "MOON", "PUMP"]):
        return "LONG"
    
    # Default to LONG if neutral
    return "LONG"

def calculate_confidence_score(prompt: str, direction: str) -> float:
    """Calculate confidence score based on prompt strength indicators."""
    confidence = 0.5  # Base confidence
    
    # Adjust based on confidence indicators in prompt
    strong_indicators = ["strong", "confirmed", "bullish", "bearish", "technical", "volume", "consolidation"]
    for indicator in strong_indicators:
        if indicator in prompt.lower():
            confidence += 0.05
    
    # Boost for specific analysis reasons
    if any(word in prompt.lower() for word in ["rsi", "macd", "moving average", "support", "resistance"]):
        confidence += 0.1
    
    # Cap at 0.95
    return min(confidence, 0.95)

def generate_smart_levels(ticker: str, direction: str) -> tuple:
    """Generate realistic entry, stop loss, and take profit levels based on ticker."""
    # Simplified level generation - in production, use real price data
    price_ranges = {
        "BTC/USDT": {"price": 67000, "atr": 1500},
        "ETH/USDT": {"price": 3450, "atr": 100},
        "SOL/USDT": {"price": 150, "atr": 5},
        "BNB/USDT": {"price": 620, "atr": 20},
        "AVAX/USDT": {"price": 45, "atr": 2},
    }
    
    base_data = price_ranges.get(ticker, {"price": 100, "atr": 5})
    price = base_data["price"]
    atr = base_data["atr"]
    
    if direction == "LONG":
        entry_low = price - (atr * 0.5)
        entry_high = price
        stop_loss = price - (atr * 2)
        tp1 = price + (atr * 2)
        tp2 = price + (atr * 4)
    else:  # SHORT
        entry_low = price
        entry_high = price + (atr * 0.5)
        stop_loss = price + (atr * 2)
        tp1 = price - (atr * 2)
        tp2 = price - (atr * 4)
    
    entry_zone = f"{entry_low:.2f} - {entry_high:.2f}"
    stop_loss_str = f"{stop_loss:.2f}"
    take_profit = [f"{tp1:.2f}", f"{tp2:.2f}"]
    
    return entry_zone, stop_loss_str, take_profit

# ==================== API Endpoints ====================

@app.get("/", tags=["Health"])
async def health():
    """Health check endpoint."""
    return {
        "status": "online",
        "engine": "Phoenix Signal Core Layer",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/v1/health", tags=["Health"])
async def health_detailed():
    """Detailed health check with system info."""
    return {
        "status": "online",
        "engine": "Phoenix Signal Core Layer",
        "version": "1.0.0",
        "cache_size": len(SIGNAL_HISTORY_CACHE),
        "max_cache_size": MAX_CACHE_SIZE,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/v1/history", response_model=HistoryResponse, tags=["Signals"])
async def get_signal_history(limit: int = Query(12, ge=1, le=50)):
    """Retrieve signal history with optional limit."""
    try:
        truncated_cache = SIGNAL_HISTORY_CACHE[:limit]
        return HistoryResponse(
            status="success",
            total=len(truncated_cache),
            data=truncated_cache
        )
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve signal history")

@app.post("/api/v1/generate-signal", response_model=SignalResponse, tags=["Signals"])
async def generate_signal(request: PromptRequest):
    """Generate a new trading signal based on prompt."""
    try:
        if not request.prompt or len(request.prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        if len(request.prompt) > MAX_PROMPT_LENGTH:
            raise HTTPException(status_code=400, detail=f"Prompt exceeds maximum length of {MAX_PROMPT_LENGTH}")
        
        logger.info(f"Generating signal for prompt: '{request.prompt}'")
        
        # Extract trading parameters
        ticker = extract_ticker_from_prompt(request.prompt)
        direction = determine_direction(request.prompt)
        entry_zone, stop_loss, take_profit = generate_smart_levels(ticker, direction)
        confidence = calculate_confidence_score(request.prompt, direction)
        
        # Create signal response
        signal = SignalResponse(
            signal_id=str(uuid.uuid4()),
            asset_pair=ticker,
            direction=direction,
            entry_zone=entry_zone,
            stop_loss=stop_loss,
            take_profit=take_profit,
            analysis_reason=f"AI Engine analyzed: {request.prompt[:100]}...",
            confidence_score=round(confidence, 2),
            generated_at=datetime.utcnow().isoformat(),
            status="OPEN"
        )
        
        # Add to cache
        SIGNAL_HISTORY_CACHE.insert(0, signal)
        if len(SIGNAL_HISTORY_CACHE) > MAX_CACHE_SIZE:
            SIGNAL_HISTORY_CACHE.pop()
        
        logger.info(f"Signal generated: {signal.signal_id} for {ticker}")
        return signal
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating signal: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate signal")

@app.get("/api/v1/signal/{signal_id}", response_model=SignalResponse, tags=["Signals"])
async def get_signal_by_id(signal_id: str):
    """Retrieve a specific signal by ID."""
    try:
        for signal in SIGNAL_HISTORY_CACHE:
            if signal.signal_id == signal_id:
                return signal
        raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving signal: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve signal")

@app.get("/api/v1/signals/pair/{pair}", tags=["Signals"])
async def get_signals_by_pair(pair: str, limit: int = Query(10, ge=1, le=50)):
    """Retrieve signals for a specific trading pair."""
    try:
        pair_upper = pair.upper()
        filtered_signals = [s for s in SIGNAL_HISTORY_CACHE if s.asset_pair == pair_upper][:limit]
        
        return {
            "status": "success",
            "pair": pair_upper,
            "total": len(filtered_signals),
            "data": filtered_signals
        }
    except Exception as e:
        logger.error(f"Error filtering signals: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to filter signals")

@app.get("/api/v1/supported-tickers", tags=["Config"])
async def get_supported_tickers():
    """Get list of supported trading pairs."""
    return {
        "status": "success",
        "supported_tickers": list(SUPPORTED_TICKERS.values()),
        "default_ticker": DEFAULT_TICKER
    }

# ==================== Error Handlers ====================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    logger.error(f"Validation error: {str(exc)}")
    return {"status": "error", "detail": str(exc)}

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {str(exc)}")
    return {"status": "error", "detail": "Internal server error"}

# ==================== Startup Event ====================

@app.on_event("startup")
async def startup_event():
    logger.info("Phoenix Signal Core Backend started successfully")
    logger.info(f"Cache size: {len(SIGNAL_HISTORY_CACHE)}/{MAX_CACHE_SIZE}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
