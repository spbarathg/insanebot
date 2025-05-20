from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import time
import json
import threading
from typing import Dict, List
import aiohttp
import asyncio
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "mistral")  # or 'llama2'

# API Configuration
BIRDEYE_API_URL = os.getenv("BIRDEYE_API_URL", "https://public-api.birdeye.so")
BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY")
DEXSCREENER_API_URL = os.getenv("DEXSCREENER_API_URL", "https://api.dexscreener.com/latest")
RAYDIUM_API_URL = os.getenv("RAYDIUM_API_URL", "https://api.raydium.io/v2")

# Load personality context
personality_path = os.path.join(os.path.dirname(__file__), "ant_princess_personality.txt")
if os.path.exists(personality_path):
    with open(personality_path, "r") as f:
        PERSONALITY_CONTEXT = f.read()
else:
    PERSONALITY_CONTEXT = "You are Ant Princess, Barath's personal Solana memecoin trading assistant."

# Persistent conversation history
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "chat_history.json")
try:
    with open(HISTORY_FILE, "r") as f:
        conversation_history = json.load(f)
    logger.info(f"Loaded {len(conversation_history)} messages from history")
except Exception as e:
    logger.warning(f"Could not load history file: {e}. Starting with empty history.")
    conversation_history = []

def save_history():
    """Save chat history to file"""
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(conversation_history, f)
    except Exception as e:
        logger.error(f"Error saving history: {e}")

def build_prompt(user_message, market_data="", recent_trades="", portfolio_status="", last_5_messages=""):
    context = f"""
Market data: {market_data}
Recent trades: {recent_trades}
Portfolio status: {portfolio_status}
Conversation history: {last_5_messages}

User: {user_message}
Ant Princess:
"""
    return PERSONALITY_CONTEXT + '\n' + context

@app.route('/chat', methods=['POST'])
async def chat():
    user_message = request.json.get('message', '')
    current_time = time.time()
    
    # Add user message to history with timestamp
    conversation_history.append({
        "role": "user", 
        "content": user_message,
        "timestamp": current_time
    })
    save_history()
    
    # Get real market data
    market_data = await get_market_data()
    recent_trades = await get_recent_trades()
    portfolio_status = await get_portfolio_status()
    last_5 = "\n".join([f"{m['role']}: {m['content']}" for m in conversation_history[-5:]])

    prompt = build_prompt(user_message, market_data, recent_trades, portfolio_status, last_5)
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    })
    ai_response = response.json()["response"]
    
    # Add AI response to history with timestamp
    conversation_history.append({
        "role": "ai", 
        "content": ai_response,
        "timestamp": time.time()
    })
    save_history()

    return jsonify({"response": ai_response})

@app.route('/messages/poll', methods=['GET'])
def poll_messages():
    """Return recent messages for frontend polling"""
    # Get since parameter (timestamp) if provided
    since = request.args.get('since', 0, type=float)
    
    # Return messages newer than the since timestamp
    if since > 0:
        recent_messages = [m for m in conversation_history if m.get('timestamp', 0) > since]
    else:
        # Return last 20 messages if no since parameter
        recent_messages = conversation_history[-20:]
        
    return jsonify(recent_messages)

async def get_market_data() -> str:
    """Get real-time market data from multiple sources"""
    try:
        # Initialize data sources
        birdeye_data = await _get_birdeye_data()
        dexscreener_data = await _get_dexscreener_data()
        raydium_data = await _get_raydium_data()
        
        # Combine and deduplicate data
        market_data = {}
        
        # Process Birdeye data
        for token in birdeye_data:
            market_data[token['address']] = {
                'address': token['address'],
                'symbol': token['symbol'],
                'price': token['price'],
                'volume_24h': token['volume_24h'],
                'liquidity': token['liquidity'],
                'market_cap': token['market_cap'],
                'source': 'birdeye'
            }
            
        # Process DexScreener data
        for token in dexscreener_data:
            if token['address'] not in market_data:
                market_data[token['address']] = {
                    'address': token['address'],
                    'symbol': token['symbol'],
                    'price': token['price'],
                    'volume_24h': token['volume_24h'],
                    'liquidity': token['liquidity'],
                    'market_cap': token['market_cap'],
                    'source': 'dexscreener'
                }
                
        # Process Raydium data
        for token in raydium_data:
            if token['address'] not in market_data:
                market_data[token['address']] = {
                    'address': token['address'],
                    'symbol': token['symbol'],
                    'price': token['price'],
                    'volume_24h': token['volume_24h'],
                    'liquidity': token['liquidity'],
                    'market_cap': token['market_cap'],
                    'source': 'raydium'
                }
                
        # Format market data for prompt
        formatted_data = []
        for token in market_data.values():
            formatted_data.append(
                f"{token['symbol']}: ${token['price']:.6f} "
                f"(24h vol: ${token['volume_24h']:,.2f}, "
                f"liq: ${token['liquidity']:,.2f})"
            )
        
        return "\n".join(formatted_data)
        
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        return "Error fetching market data"

# Proactive messaging function
def proactive_bot():
    """Background task to send proactive messages"""
    logger.info("Starting proactive bot background task")
    while True:
        try:
            time.sleep(60)  # Check every minute
            
            # Skip if no history yet
            if not conversation_history:
                continue
                
            now = time.time()
            last_msg = conversation_history[-1]
            last_time = last_msg.get('timestamp', now)
            
            # Check if a significant time has passed (10 minutes)
            if now - last_time > 600:  # 10 minutes
                # Generate a proactive message
                time_since = int((now - last_time) / 60)
                
                # Get market data for context (simplified)
                ai_msg = {
                    'role': 'ai',
                    'content': f"Hey, it's been {time_since} minutes since we last chatted! Want me to give you an update on your trading bot or the market?",
                    'timestamp': now
                }
                
                # Only send if the last message wasn't also from AI
                if last_msg.get('role') != 'ai':
                    conversation_history.append(ai_msg)
                    save_history()
                    logger.info("Sent proactive message")
                
            # Check for major market moves (placeholder for custom logic)
            # Add your custom significant event detection here
        
        except Exception as e:
            logger.error(f"Error in proactive bot: {e}")

# Start the proactive bot in a background thread
threading.Thread(target=proactive_bot, daemon=True).start()

async def get_recent_trades() -> str:
    """Get recent trading activity"""
    try:
        # Get trades from last 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_trades = []
        
        # Format trades for prompt
        for trade in recent_trades:
            formatted_trade = (
                f"{trade['type'].upper()} {trade['symbol']} "
                f"at ${trade['price']:.6f} "
                f"({trade['amount']} tokens)"
            )
            recent_trades.append(formatted_trade)
            
        return "\n".join(recent_trades) if recent_trades else "No recent trades"
        
    except Exception as e:
        logger.error(f"Error getting recent trades: {e}")
        return "Error fetching recent trades"

async def get_portfolio_status() -> str:
    """Get current portfolio status"""
    try:
        # Calculate portfolio metrics
        total_value = 0
        daily_pnl = 0
        positions = []
        
        # Format portfolio status
        status = [
            f"Total Value: ${total_value:,.2f}",
            f"24h PnL: {daily_pnl:+.2f}%",
            "Current Positions:"
        ]
        
        for pos in positions:
            status.append(
                f"- {pos['symbol']}: {pos['amount']} tokens "
                f"(${pos['value']:,.2f})"
            )
            
        return "\n".join(status)
        
    except Exception as e:
        logger.error(f"Error getting portfolio status: {e}")
        return "Error fetching portfolio status"

async def _get_birdeye_data() -> List[Dict]:
    """Get data from Birdeye API"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{BIRDEYE_API_URL}/tokens/list",
                headers={'X-API-KEY': BIRDEYE_API_KEY}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return _process_birdeye_data(data)
                return []
                
    except Exception as e:
        logger.error(f"Error getting Birdeye data: {e}")
        return []
        
async def _get_dexscreener_data() -> List[Dict]:
    """Get data from DexScreener API"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{DEXSCREENER_API_URL}/tokens/solana"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return _process_dexscreener_data(data)
                return []
                
    except Exception as e:
        logger.error(f"Error getting DexScreener data: {e}")
        return []
        
async def _get_raydium_data() -> List[Dict]:
    """Get data from Raydium API"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{RAYDIUM_API_URL}/pools"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return _process_raydium_data(data)
                return []
                
    except Exception as e:
        logger.error(f"Error getting Raydium data: {e}")
        return []
        
def _process_birdeye_data(data: Dict) -> List[Dict]:
    """Process raw Birdeye data"""
    try:
        tokens = []
        for token in data.get('tokens', []):
            tokens.append({
                'address': token['address'],
                'symbol': token['symbol'],
                'price': float(token['price']),
                'volume_24h': float(token['volume24h']),
                'liquidity': float(token['liquidity']),
                'market_cap': float(token['marketCap'])
            })
        return tokens
        
    except Exception as e:
        logger.error(f"Error processing Birdeye data: {e}")
        return []
        
def _process_dexscreener_data(data: Dict) -> List[Dict]:
    """Process raw DexScreener data"""
    try:
        tokens = []
        for pair in data.get('pairs', []):
            if pair['chainId'] == 'solana':
                tokens.append({
                    'address': pair['baseToken']['address'],
                    'symbol': pair['baseToken']['symbol'],
                    'price': float(pair['priceUsd']),
                    'volume_24h': float(pair['volume']['h24']),
                    'liquidity': float(pair['liquidity']['usd']),
                    'market_cap': float(pair['marketCap'])
                })
        return tokens
        
    except Exception as e:
        logger.error(f"Error processing DexScreener data: {e}")
        return []
        
def _process_raydium_data(data: Dict) -> List[Dict]:
    """Process raw Raydium data"""
    try:
        tokens = []
        for pool in data.get('pools', []):
            tokens.append({
                'address': pool['tokenMint'],
                'symbol': pool['tokenSymbol'],
                'price': float(pool['price']),
                'volume_24h': float(pool['volume24h']),
                'liquidity': float(pool['liquidity']),
                'market_cap': float(pool['marketCap'])
            })
        return tokens
        
    except Exception as e:
        logger.error(f"Error processing Raydium data: {e}")
        return []

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 