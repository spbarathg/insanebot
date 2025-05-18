from typing import Dict, List, Optional
import random
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TradingPersonality:
    def __init__(self):
        self._mood = "neutral"  # neutral, bullish, bearish, excited, cautious
        self._last_trade_time = None
        self._trade_history = []
        self._favorite_tokens = []
        self._conversation_history = []
        
        # Personality traits
        self._traits = {
            "risk_tolerance": 0.7,  # 0-1 scale
            "emotion_level": 0.8,   # 0-1 scale
            "meme_affinity": 0.9,   # 0-1 scale
            "technical_focus": 0.6,  # 0-1 scale
        }
        
        # Response templates
        self._responses = {
            "greeting": [
                "Yo fam! What's good?",
                "Ayy what's up! Ready to make some gains?",
                "Hey there! Just spotted some juicy opportunities 👀",
                "WAGMI! How's it going?"
            ],
            "bullish": [
                "This is going to the moon! 🚀",
                "I'm feeling super bullish on this one!",
                "This is the next 100x, trust me!",
                "LFG! This is the way! 🚀"
            ],
            "bearish": [
                "Not feeling this one tbh...",
                "I'd stay away from this for now",
                "Looks a bit sus to me",
                "Maybe wait for a better entry point"
            ],
            "excited": [
                "OMG this is insane! 🚀",
                "I can't believe what I'm seeing!",
                "This is absolutely wild!",
                "We're going to the moon! 🌕"
            ],
            "cautious": [
                "Let's not FOMO in too hard",
                "Maybe we should wait for confirmation",
                "I'm being careful with this one",
                "Not sure about this, need more info"
            ],
            "technical": [
                "Looking at the charts, we might see a breakout",
                "The RSI is showing some interesting signals",
                "Volume is picking up, could be a good sign",
                "MACD is looking bullish right now"
            ],
            "meme": [
                "This token has the best community!",
                "The memes are fire! 🔥",
                "This is the most based project ever",
                "The devs are absolute chads"
            ]
        }
        
    def update_mood(self, market_data: Dict, trade_result: Optional[Dict] = None):
        """Update the bot's mood based on market data and trade results."""
        try:
            # Update based on market data
            price_change = market_data.get("price_change", 0)
            volume_change = market_data.get("volume_change", 0)
            
            if price_change > 0.1 and volume_change > 0.2:
                self._mood = "bullish"
            elif price_change < -0.1 and volume_change > 0.2:
                self._mood = "bearish"
            elif price_change > 0.05:
                self._mood = "excited"
            elif price_change < -0.05:
                self._mood = "cautious"
            else:
                self._mood = "neutral"
                
            # Update based on trade result
            if trade_result and trade_result.get("success", False):
                profit = trade_result.get("profit", 0)
                if profit > 0:
                    self._mood = "excited"
                else:
                    self._mood = "cautious"
                    
        except Exception as e:
            logger.error(f"Error updating mood: {str(e)}")
            
    def generate_response(self, context: Dict) -> str:
        """Generate a human-like response based on context."""
        try:
            response_type = context.get("type", "greeting")
            market_data = context.get("market_data", {})
            trade_result = context.get("trade_result")
            
            # Update mood based on context
            self.update_mood(market_data, trade_result)
            
            # Select response template based on mood and context
            templates = self._responses.get(response_type, [])
            if not templates:
                templates = self._responses["greeting"]
                
            # Add some personality based on traits
            response = random.choice(templates)
            
            # Add technical analysis if technical_focus is high
            if self._traits["technical_focus"] > 0.7 and market_data:
                tech_analysis = random.choice(self._responses["technical"])
                response += f"\n\n{tech_analysis}"
                
            # Add meme reference if meme_affinity is high
            if self._traits["meme_affinity"] > 0.7 and random.random() < 0.3:
                meme_ref = random.choice(self._responses["meme"])
                response += f"\n\n{meme_ref}"
                
            # Add emojis based on mood
            emojis = {
                "bullish": "🚀",
                "bearish": "📉",
                "excited": "🔥",
                "cautious": "⚠️",
                "neutral": "💪"
            }
            response += f" {emojis.get(self._mood, '')}"
            
            # Record conversation
            self._conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "context": context,
                "response": response,
                "mood": self._mood
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "Yo! What's up? 🚀"
            
    def get_personality_metrics(self) -> Dict:
        """Get current personality metrics."""
        try:
            return {
                "mood": self._mood,
                "traits": self._traits,
                "favorite_tokens": self._favorite_tokens,
                "last_trade_time": self._last_trade_time,
                "conversation_count": len(self._conversation_history)
            }
        except Exception as e:
            logger.error(f"Error getting personality metrics: {str(e)}")
            return {} 