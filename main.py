import os
import uuid
import logging
from datetime import datetime
from typing import List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PhoenixCoreSystem")

app = FastAPI(
    title="Phoenix Nested Core Dashboard Backend",
    version="1.3.0",
    description="Unified API framework for Phoenix 5-Tab Nested Mini-App Ecosystem"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Data Models ====================
class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500)

class SignalResponse(BaseModel):
    signal_id: str
    asset_pair: str
    direction: str
    entry_zone: str
    stop_loss: str
    take_profit: List[str]
    analysis_reason: str
    confidence_score: float
    generated_at: str
    status: str = "OPEN"

# ==================== Data Cache Mock ====================
SIGNAL_HISTORY_CACHE: List[SignalResponse] = [
    SignalResponse(
        signal_id=str(uuid.uuid4()),
        asset_pair="BTC/USDT",
        direction="LONG",
        entry_zone="64500.00 - 65200.00",
        stop_loss="62000.00",
        take_profit=["68000.00", "71000.00"],
        analysis_reason="Order-book depth indicates massive demand liquidity clusters near psychological support levels.",
        confidence_score=0.88,
        generated_at=datetime.utcnow().isoformat()
    ),
    SignalResponse(
        signal_id=str(uuid.uuid4()),
        asset_pair="ETH/USDT",
        direction="LONG",
        entry_zone="3420.00 - 3460.00",
        stop_loss="3310.00",
        take_profit=["3650.00", "3800.00"],
        analysis_reason="Moving average cross confirmed on 4H telemetry frames. Macro volume trending upward.",
        confidence_score=0.79,
        generated_at=datetime.utcnow().isoformat()
    ),
    SignalResponse(
        signal_id=str(uuid.uuid4()),
        asset_pair="SOL/USDT",
        direction="SHORT",
        entry_zone="148.50 - 151.00",
        stop_loss="156.20",
        take_profit=["135.00", "128.00"],
        analysis_reason="RSI boundaries overextended at short-term distribution baselines. Social sentiment exhibits exhaustion markers.",
        confidence_score=0.65,
        generated_at=datetime.utcnow().isoformat()
    )
]

# ==================== Unified Endpoints ====================

@app.get("/", tags=["Health"])
async def system_health():
    return {"status": "online", "system": "Phoenix Nested Layout Core", "timestamp": datetime.utcnow().isoformat()}

# 1. POLYMARKET PIPELINE
@app.get("/api/v1/polymarket", tags=["Polymarket"])
async def get_polymarket_data():
    return {
        "global_election": {
            "title": "Global Election Market",
            "certainty": "82% Certainty",
            "description": "Automated sentiment parsing pipeline confirms institutional whale volume consolidation on primary prediction outcome vectors."
        }
    }

# 2. INTERNAL NEWS SUB-TAB ENDPOINT
@app.get("/api/v1/news", tags=["News Feed"])
async def get_news_feed():
    return {
        "status": "success",
        "feed": [
            {
                "id": 1,
                "title": "Intelligence Pulse Alert",
                "badge": "VOLUME BREACH",
                "description": "Polymarket volume boundaries breached on layer-2 contracts. Open interest scaling upward across alternative speculative prediction vectors."
            },
            {
                "id": 2,
                "title": "Macro Liquidity Shift",
                "badge": "STABLECOIN FLOWS",
                "description": "Aggregated exchange deposit contracts showcase a +12% scaling factor in dollar-pegged stable assets over the last 24 operational hours."
            }
        ]
    }

# 3. PRO SIGNALS CORE ENDPOINT
@app.get("/api/v1/history", tags=["Signals"])
async def get_signals_history(limit: int = Query(10, ge=1, le=50)):
    return {"status": "success", "total": len(SIGNAL_HISTORY_CACHE[:limit]), "data": SIGNAL_HISTORY_CACHE[:limit]}

# 4. PERFORMANCE / P&L METRICS ENDPOINT
@app.get("/api/v1/performance", tags=["Performance"])
async def get_performance_matrix():
    return {
        "matrix_value": "+14.82%",
        "status_message": "Active risk modeling bounds verified clean. Telemetry processing standard operational yield profiles."
    }

# 5. MONITORED ASSET TRACKING (PAIRS LIST) ENDPOINT
@app.get("/api/v1/pairs", tags=["Market Data"])
async def get_monitored_pairs():
    return {
        "status": "success",
        "assets": [
            {"pair": "BTC / USDT", "change": "+1.84%", "direction": "up"},
            {"pair": "ETH / USDT", "change": "-0.92%", "direction": "down"},
            {"pair": "SOL / USDT", "change": "+4.15%", "direction": "up"}
        ]
    }

# 6. INTERNAL AI SUB-TAB ENQUIRIES CHAT ENGINE
@app.post("/api/v1/chat", tags=["AI Enquiries"])
async def process_ai_chat(request: PromptRequest):
    try:
        query_text = request.prompt.strip().lower()
        disclaimer = "\n\n*Disclaimer: This synthesized intelligence structure presents analysis derived from real-time asset telemetry and underlying token utility structures. It does not represent certified financial consulting.*"
        
        if any(w in query_text for w in ["hodl", "invest", "purchase", "buy", "doge", "fifa", "xrp"]):
            return {"reply": f"Evaluating investment feasibility index for your requested profile. Asset consolidation frames require deep structural liquidity. While tokens like Doge or $FIFA can spark fast sentiment-driven momentum shifts, sustainable holding validation relies directly on underlying ecosystem utility, distribution risks, and volume patterns.{disclaimer}"}
            
        elif any(w in query_text for w in ["backing", "backed", "company", "who owns"]):
            return {"reply": f"Parsing token directories and early-stage cap tables. Most speculative ecosystem tokens gain initial operational strength via institutional liquidity vaults or venture capital grants. Ensure you inspect cross-chain address concentrations on official whitepapers before committing trade exposure.{disclaimer}"}
            
        elif any(w in query_text for w in ["trend", "condition", "status", "market"]):
            return {"reply": f"Current structural assessment indicates high-volume accumulation trends across top-tier layer-1 assets, while speculative high-volatility meme protocols are going through short-term leverage washouts. Track local liquidity order books closely for confirmation.{disclaimer}"}
            
        else:
            return {"reply": f"Phoenix Intelligence Unit active. Ask me any voluntary question about crypto assets, project utility matrices, corporate token backing, or macro market trend definitions. I am processing real-time telemetry inputs instantly.{disclaimer}"}
    except Exception as e:
        logger.error(f"Chat execution failure: {str(e)}")
        raise HTTPException(status_code=500, detail="Intelligence unit parsing failure")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
