from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict
import json
import time
from loguru import logger
from ..core.main import MemeCoinBot

app = FastAPI()
bot = None

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    global bot
    bot = MemeCoinBot()
    await bot.start()

@app.on_event("shutdown")
async def shutdown_event():
    if bot:
        await bot.stop()

@app.get("/status")
async def get_status() -> Dict:
    """Get bot status"""
    try:
        if not bot:
            raise HTTPException(status_code=503, detail="Bot not initialized")
            
        return {
            "status": "running",
            "active_trades": bot.trade_execution.get_active_trades(),
            "daily_loss": bot.daily_loss,
            "last_trade_time": bot.last_trade_time,
            "feature_weights": bot.feature_weights
        }
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trades")
async def get_trades() -> Dict:
    """Get recent trades"""
    try:
        with open('trades.json', 'r') as f:
            trades = json.load(f)
            
        # Get last 100 trades
        recent_trades = sorted(
            trades,
            key=lambda x: x['timestamp'],
            reverse=True
        )[:100]
        
        return {
            "total_trades": len(trades),
            "recent_trades": recent_trades
        }
    except Exception as e:
        logger.error(f"Get trades error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics() -> Dict:
    """Get trading metrics"""
    try:
        with open('trades.json', 'r') as f:
            trades = json.load(f)
            
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_profit": 0,
                "avg_profit": 0
            }
            
        profitable_trades = sum(1 for t in trades if t['profit'] > 0)
        total_profit = sum(t['profit'] for t in trades)
        
        return {
            "total_trades": len(trades),
            "win_rate": (profitable_trades/len(trades))*100 if trades else 0,
            "total_profit": total_profit,
            "avg_profit": total_profit/len(trades) if trades else 0
        }
    except Exception as e:
        logger.error(f"Get metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080) 