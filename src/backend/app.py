from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from typing import Dict, List
import aiohttp

app = Flask(__name__)
CORS(app)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "mistral")  # or 'llama2'

# Load personality context
personality_path = os.path.join(os.path.dirname(__file__), "ant_princess_personality.txt")
if os.path.exists(personality_path):
    with open(personality_path, "r") as f:
        PERSONALITY_CONTEXT = f.read()
else:
    PERSONALITY_CONTEXT = "You are Ant Princess, Barath's personal Solana memecoin trading assistant."

# In-memory conversation history (replace with DB for persistence)
conversation_history = []

def build_prompt(user_message, grok_sentiment="", recent_trades="", portfolio_status="", last_5_messages=""):
    context = f"""
Market sentiment: {grok_sentiment}
Recent trades: {recent_trades}
Portfolio status: {portfolio_status}
Conversation history: {last_5_messages}

User: {user_message}
Ant Princess:
"""
    return PERSONALITY_CONTEXT + '\n' + context

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    # TODO: Replace with real data sources
    grok_sentiment = "Bullish, lots of hype around new Solana memecoins."
    recent_trades = "Bought $SOLMOON, sold $DOGESOL."
    portfolio_status = "Up 12% this week."
    last_5 = "\n".join([f"{m['role']}: {m['content']}" for m in conversation_history[-5:]])

    prompt = build_prompt(user_message, grok_sentiment, recent_trades, portfolio_status, last_5)
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    })
    ai_response = response.json()["response"]
    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "ai", "content": ai_response})

    return jsonify({"response": ai_response})

async def get_market_data(self) -> Dict:
    """Get real-time market data from multiple sources"""
    try:
        # Initialize data sources
        birdeye_data = await self._get_birdeye_data()
        dexscreener_data = await self._get_dexscreener_data()
        raydium_data = await self._get_raydium_data()
        
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
                
        return market_data
        
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        return {}
        
async def _get_birdeye_data(self) -> List[Dict]:
    """Get data from Birdeye API"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{settings.BIRDEYE_API_URL}/tokens/list",
                headers={'X-API-KEY': settings.BIRDEYE_API_KEY}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_birdeye_data(data)
                return []
                
    except Exception as e:
        logger.error(f"Error getting Birdeye data: {e}")
        return []
        
async def _get_dexscreener_data(self) -> List[Dict]:
    """Get data from DexScreener API"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{settings.DEXSCREENER_API_URL}/tokens/solana"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_dexscreener_data(data)
                return []
                
    except Exception as e:
        logger.error(f"Error getting DexScreener data: {e}")
        return []
        
async def _get_raydium_data(self) -> List[Dict]:
    """Get data from Raydium API"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{settings.RAYDIUM_API_URL}/pools"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_raydium_data(data)
                return []
                
    except Exception as e:
        logger.error(f"Error getting Raydium data: {e}")
        return []
        
def _process_birdeye_data(self, data: Dict) -> List[Dict]:
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
        
def _process_dexscreener_data(self, data: Dict) -> List[Dict]:
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
        
def _process_raydium_data(self, data: Dict) -> List[Dict]:
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True) 