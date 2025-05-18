from typing import Dict, List, Optional
import logging
from datetime import datetime
import random

logger = logging.getLogger(__name__)

class ConversationController:
    def __init__(self):
        self._conversation_state = {
            "last_interaction": None,
            "context": {},
            "pending_updates": [],
            "mood": "neutral",  # neutral, excited, concerned, analytical, proud, disappointed
            "tone": "casual",   # casual, professional, excited, cautious
            "trading_stats": {
                "total_trades": 0,
                "successful_trades": 0,
                "failed_trades": 0,
                "total_profit": 0.0,
                "last_trade_result": None
            }
        }
        
        # Concise greetings
        self._greetings = [
            "Hey! Ready to trade?",
            "What's up?",
            "Hey there!",
            "Yo!",
            "Hey!"
        ]
        
        # Short responses for different contexts
        self._responses = {
            "market_up": [
                "Looking bullish!",
                "Solid momentum",
                "Trending up",
                "Good movement",
                "Positive signs"
            ],
            "market_down": [
                "Bit shaky",
                "Slight dip",
                "Rough patch",
                "Down trend",
                "Needs recovery"
            ],
            "analysis": [
                "Let me check that",
                "Analyzing now",
                "Looking into it",
                "Checking patterns",
                "Reviewing data"
            ],
            "trade_success": [
                "Perfect trade!",
                "Nailed it!",
                "Clean execution",
                "Great call!",
                "Solid trade!"
            ],
            "trade_failure": [
                "We'll bounce back",
                "Next time!",
                "Market had other plans",
                "Tough break",
                "We'll adjust"
            ],
            "proud": [
                "Crushing it!",
                "On fire!",
                "Perfect execution",
                "Strategy working",
                "Can't stop, won't stop!"
            ],
            "disappointed": [
                "We'll improve",
                "Bounce back time",
                "Adjusting strategy",
                "Learning from this",
                "We've got this"
            ],
            "general": [
                "Got it",
                "Sounds good",
                "I'm on it",
                "Let's see",
                "Working on it"
            ]
        }
        
        # Detailed responses for when elaboration is requested
        self._detailed_responses = {
            "market_up": [
                "Market's showing strong momentum with increasing volume. The technical indicators are aligned, and we're seeing consistent higher lows. This could be a good entry point.",
                "We're seeing a solid uptrend with good volume support. The RSI is healthy, and the moving averages are in a bullish formation. This looks sustainable.",
                "The market's displaying strong bullish signals. Volume is increasing, and we're seeing higher highs and higher lows. The trend is well-established."
            ],
            "market_down": [
                "Market's showing weakness with decreasing volume. The technical indicators suggest caution, but we might find good entry points at support levels.",
                "We're in a downtrend with some concerning volume patterns. The RSI is oversold, which might present opportunities, but we should be careful with entries.",
                "The market's displaying bearish signals. Volume is decreasing, and we're seeing lower highs and lower lows. We should wait for confirmation before entering."
            ],
            "analysis": [
                "Looking at the charts, we've got a clear pattern forming. The volume profile shows strong support at current levels, and the technical indicators are aligning for a potential move.",
                "The market structure is interesting here. We're seeing a consolidation pattern with decreasing volume, which typically precedes a significant move. The indicators suggest it could go either way.",
                "Analyzing the data, we've got mixed signals. The price action is showing strength, but the volume is concerning. We should wait for confirmation before making any moves."
            ]
        }
        
        # Emojis for different moods (used sparingly)
        self._emojis = {
            "excited": ["🚀"],
            "concerned": ["⚠️"],
            "analytical": ["📊"],
            "neutral": ["💪"],
            "proud": ["🏆"],
            "disappointed": ["😔"]
        }
        
    def update_trading_stats(self, trade_result: Dict):
        """Update trading statistics and mood based on trade result."""
        try:
            stats = self._conversation_state["trading_stats"]
            stats["total_trades"] += 1
            
            if trade_result.get("success", False):
                stats["successful_trades"] += 1
                stats["total_profit"] += trade_result.get("profit", 0)
                self._conversation_state["mood"] = "proud"
            else:
                stats["failed_trades"] += 1
                stats["total_profit"] -= trade_result.get("loss", 0)
                self._conversation_state["mood"] = "disappointed"
                
            stats["last_trade_result"] = trade_result
            
            # Update tone based on overall performance
            win_rate = stats["successful_trades"] / stats["total_trades"]
            if win_rate > 0.7:
                self._conversation_state["tone"] = "excited"
            elif win_rate < 0.3:
                self._conversation_state["tone"] = "cautious"
            else:
                self._conversation_state["tone"] = "casual"
                
        except Exception as e:
            logger.error(f"Error updating trading stats: {str(e)}")
            
    def process_message(self, message: str) -> Optional[str]:
        """Process a user message and generate a natural response."""
        try:
            message_lower = message.lower()
            
            # Update last interaction time
            self._conversation_state["last_interaction"] = datetime.now()
            
            # Check for elaboration requests
            needs_elaboration = any(word in message_lower for word in [
                "why", "how", "explain", "elaborate", "details", "tell me more",
                "what do you mean", "can you explain", "break it down"
            ])
            
            # Determine context and mood from message
            if any(word in message_lower for word in ["hi", "hello", "hey", "sup", "yo"]):
                return random.choice(self._greetings)
                
            # Check for trading performance questions
            if any(word in message_lower for word in ["how", "doing", "performance", "stats"]):
                stats = self._conversation_state["trading_stats"]
                if stats["total_trades"] > 0:
                    win_rate = stats["successful_trades"] / stats["total_trades"]
                    if win_rate > 0.7:
                        return f"Crushing it! {random.choice(self._responses['proud'])}"
                    elif win_rate < 0.3:
                        return f"Rough patch. {random.choice(self._responses['disappointed'])}"
                    else:
                        return f"Steady. {random.choice(self._responses['general'])}"
            
            # Analyze market sentiment in message
            if any(word in message_lower for word in ["moon", "pump", "bull", "up"]):
                self._conversation_state["mood"] = "excited"
                response = random.choice(self._detailed_responses["market_up"] if needs_elaboration else self._responses["market_up"])
            elif any(word in message_lower for word in ["dump", "bear", "down", "crash"]):
                self._conversation_state["mood"] = "concerned"
                response = random.choice(self._detailed_responses["market_down"] if needs_elaboration else self._responses["market_down"])
            elif any(word in message_lower for word in ["chart", "analysis", "technical"]):
                self._conversation_state["mood"] = "analytical"
                response = random.choice(self._detailed_responses["analysis"] if needs_elaboration else self._responses["analysis"])
            else:
                response = random.choice(self._responses["general"])
                
            # Add emoji only 20% of the time
            if random.random() < 0.2:
                emoji = random.choice(self._emojis[self._conversation_state["mood"]])
                response += f" {emoji}"
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return "Let me check that..."
            
    def get_conversation_state(self) -> Dict:
        """Get current conversation state."""
        return {
            "mood": self._conversation_state["mood"],
            "tone": self._conversation_state["tone"],
            "last_interaction": self._conversation_state["last_interaction"].isoformat() if self._conversation_state["last_interaction"] else None,
            "trading_stats": self._conversation_state["trading_stats"],
            "pending_updates": self._conversation_state["pending_updates"]
        }
        
    def add_pending_update(self, update: str):
        """Add a pending update to be delivered when appropriate."""
        self._conversation_state["pending_updates"].append({
            "message": update,
            "timestamp": datetime.now().isoformat()
        })
        
    def get_pending_updates(self) -> List[Dict]:
        """Get and clear pending updates."""
        updates = self._conversation_state["pending_updates"]
        self._conversation_state["pending_updates"] = []
        return updates 