from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, List
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

@app.post("/webhook")
async def helius_webhook(request: Request):
    """Receive Helius webhook notifications for real-time transaction data"""
    try:
        # Verify the request came from Helius (optional security)
        auth_header = request.headers.get("api-key")
        expected_key = "193ececa-6e42-4d84-b9bd-765c4813816d"  # Your Helius API key
        
        if auth_header != expected_key:
            logger.warning(f"Webhook received with invalid API key: {auth_header}")
            # Still process it for now, but log the warning
        
        # Get the webhook payload
        webhook_data = await request.json()
        logger.info(f"📡 Received Helius webhook with {len(webhook_data)} transactions")
        
        # Process each transaction in the webhook
        processed_count = 0
        for transaction in webhook_data:
            try:
                await process_webhook_transaction(transaction)
                processed_count += 1
            except Exception as e:
                logger.error(f"Error processing webhook transaction: {str(e)}")
                continue
        
        logger.info(f"✅ Processed {processed_count}/{len(webhook_data)} webhook transactions")
        
        return {
            "status": "success",
            "processed": processed_count,
            "total": len(webhook_data),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Webhook processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {str(e)}")

# Load monitored wallets from file
MONITORED_WALLETS = {}  # Changed to dict to store metadata

def load_monitored_wallets():
    """Load the list of wallets to monitor from JSON file"""
    global MONITORED_WALLETS
    try:
        with open('/app/config/monitored_wallets.txt', 'r') as f:
            wallet_data = json.load(f)
            
        # Convert to dict for faster lookups with metadata
        MONITORED_WALLETS = {}
        active_alerts = 0
        
        for wallet in wallet_data:
            address = wallet.get('trackedWalletAddress', '')
            if address:
                MONITORED_WALLETS[address] = {
                    'name': wallet.get('name', 'Unknown'),
                    'emoji': wallet.get('emoji', '👤'),
                    'alertsOn': wallet.get('alertsOn', False)
                }
                if wallet.get('alertsOn', False):
                    active_alerts += 1
        
        logger.info(f"📋 Loaded {len(MONITORED_WALLETS)} monitored wallets ({active_alerts} with alerts enabled)")
        
    except FileNotFoundError:
        logger.warning("⚠️ No monitored_wallets.txt found")
        MONITORED_WALLETS = {}
    except json.JSONDecodeError as e:
        logger.error(f"❌ Error parsing wallet JSON: {str(e)}")
        MONITORED_WALLETS = {}

# Load wallets on startup
load_monitored_wallets()

async def process_webhook_transaction(transaction_data: Dict):
    """Process individual transaction from Helius webhook"""
    try:
        # Extract key information from the transaction
        signature = transaction_data.get("signature", "")
        account_data = transaction_data.get("accountData", [])
        native_transfers = transaction_data.get("nativeTransfers", [])
        token_transfers = transaction_data.get("tokenTransfers", [])
        transaction_error = transaction_data.get("transactionError")
        fee_payer = transaction_data.get("feePayer", "")
        
        # Skip failed transactions
        if transaction_error:
            logger.debug(f"Skipping failed transaction: {signature}")
            return
        
        # Check if this transaction involves any monitored wallets
        involved_wallets = set()
        if fee_payer:
            involved_wallets.add(fee_payer)
        
        # Add wallets from transfers
        for transfer in native_transfers + token_transfers:
            if transfer.get("fromUserAccount"):
                involved_wallets.add(transfer.get("fromUserAccount"))
            if transfer.get("toUserAccount"):
                involved_wallets.add(transfer.get("toUserAccount"))
        
        # Check if any monitored wallets are involved
        monitored_involved = involved_wallets.intersection(set(MONITORED_WALLETS.keys()))
        
        if monitored_involved:
            for wallet in monitored_involved:
                wallet_info = MONITORED_WALLETS[wallet]
                name = wallet_info['name']
                emoji = wallet_info['emoji']
                alerts_on = wallet_info['alertsOn']
                
                # Log with emoji and name for easy identification
                if alerts_on:
                    logger.warning(f"🚨 PRIORITY ALERT: {emoji} [{wallet[:8]}...] ({name}) - ACTIVITY DETECTED!")
                else:
                    logger.info(f"🎯 TRACKED WALLET: {emoji} [{wallet[:8]}...] ({name}) - Activity detected")
                
                logger.info(f"   📄 Transaction: {signature}")
                
                # Store wallet activity for analysis with metadata
                if bot:
                    bot.market_data_cache[f"wallet_{wallet}_activity"] = {
                        "signature": signature,
                        "timestamp": time.time(),
                        "wallet_address": wallet,  # Always store actual address
                        "wallet_name": name,
                        "wallet_emoji": emoji,
                        "alerts_on": alerts_on,
                        "native_transfers": native_transfers,
                        "token_transfers": token_transfers,
                        "fee_payer": fee_payer
                    }
        
        # Look for interesting token transfers
        for transfer in token_transfers:
            token_address = transfer.get("mint", "")
            amount = transfer.get("tokenAmount", 0)
            from_wallet = transfer.get("fromUserAccount", "")
            to_wallet = transfer.get("toUserAccount", "")
            
            # Check if this is a potentially interesting token
            if token_address and amount > 0:
                logger.debug(f"💰 Token transfer: {token_address} (Amount: {amount})")
                
                # Special handling for monitored wallets
                monitored_from = MONITORED_WALLETS.get(from_wallet)
                monitored_to = MONITORED_WALLETS.get(to_wallet)
                
                if monitored_from or monitored_to:
                    wallet_info = monitored_from or monitored_to
                    wallet_addr = from_wallet if monitored_from else to_wallet
                    action = "SOLD" if monitored_from else "BOUGHT"
                    
                    logger.info(f"🔥 {wallet_info['emoji']} [{wallet_addr[:8]}...] {action} TOKEN! (labeled: {wallet_info['name']})")
                    logger.info(f"   🪙 Token: {token_address}")
                    logger.info(f"   📊 Amount: {amount:,.0f}")
                    
                    # Priority analysis for high-alert wallets or large amounts
                    if bot and (wallet_info['alertsOn'] or amount > 10000):
                        priority_level = "HIGH" if wallet_info['alertsOn'] else "MEDIUM"
                        logger.info(f"🚀 Triggering {priority_level} priority analysis for {token_address}")
                        
                        bot.market_data_cache[f"{token_address}_priority_analysis"] = {
                            "trigger": "monitored_wallet_activity",
                            "priority": priority_level,
                            "wallet": wallet_addr,
                            "wallet_name": wallet_info['name'],  # Keep but treat as label
                            "wallet_emoji": wallet_info['emoji'],
                            "action": action,
                            "amount": amount,
                            "timestamp": time.time(),
                            "note": "wallet_name may be unreliable - verify wallet behavior independently"
                        }
                
                # Store general token activity
                if bot:
                    bot.market_data_cache[f"{token_address}_webhook_activity"] = {
                        "transfer_amount": amount,
                        "from_wallet": from_wallet,
                        "to_wallet": to_wallet,
                        "signature": signature,
                        "timestamp": time.time()
                    }
        
        # Look for native SOL transfers (whale activity)
        for transfer in native_transfers:
            amount = transfer.get("amount", 0) / 1e9  # Convert lamports to SOL
            from_wallet = transfer.get("fromUserAccount", "")
            to_wallet = transfer.get("toUserAccount", "")
            
            # Log large SOL transfers
            if amount > 100:  # 100+ SOL transfers
                logger.info(f"🐋 Large SOL transfer: {amount:.2f} SOL")
                
                # Special alert for monitored wallets
                monitored_from = MONITORED_WALLETS.get(from_wallet)
                monitored_to = MONITORED_WALLETS.get(to_wallet)
                
                if monitored_from or monitored_to:
                    wallet_info = monitored_from or monitored_to
                    action = "SENT" if monitored_from else "RECEIVED"
                    wallet_addr = from_wallet if monitored_from else to_wallet
                    logger.warning(f"🚨 {wallet_info['emoji']} [{wallet_addr[:8]}...] {action} {amount:.2f} SOL! (labeled: {wallet_info['name']})")
                
    except Exception as e:
        logger.error(f"Error processing webhook transaction: {str(e)}")
        raise

@app.get("/monitored-wallets")
async def get_monitored_wallets():
    """Get list of currently monitored wallets with metadata"""
    wallets_with_alerts = sum(1 for w in MONITORED_WALLETS.values() if w['alertsOn'])
    
    return {
        "total_wallets": len(MONITORED_WALLETS),
        "wallets_with_alerts": wallets_with_alerts,
        "wallets": [
            {
                "address": addr,
                "name": info['name'],
                "emoji": info['emoji'],
                "alertsOn": info['alertsOn']
            }
            for addr, info in MONITORED_WALLETS.items()
        ]
    }

@app.post("/reload-wallets")
async def reload_monitored_wallets():
    """Reload the monitored wallets list"""
    load_monitored_wallets()
    wallets_with_alerts = sum(1 for w in MONITORED_WALLETS.values() if w['alertsOn'])
    
    return {
        "status": "reloaded",
        "total_wallets": len(MONITORED_WALLETS),
        "wallets_with_alerts": wallets_with_alerts
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080) 