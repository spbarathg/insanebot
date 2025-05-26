from typing import Dict, List, Optional
import asyncio
import logging
from datetime import datetime
from ..ai.personality import TradingPersonality
from ..ai.conversation_controller import ConversationController
from ..market.market_data import MarketData
from ..trading.trade_execution import TradeExecution

logger = logging.getLogger(__name__)

class ChatInterface:
    def __init__(self):
        self.personality = TradingPersonality()
        self.controller = ConversationController()
        self.market_data = MarketData()
        self.trade_execution = TradeExecution()
        self._active_chats = {}
        
    async def initialize(self):
        """Initialize the chat interface"""
        try:
            # Initialize market data if it has an initialize method
            if hasattr(self.market_data, 'initialize'):
                await self.market_data.initialize()
            logger.info("Chat interface initialized")
        except Exception as e:
            logger.error(f"Failed to initialize chat interface: {str(e)}")

    async def close(self):
        """Close the chat interface and clean up resources"""
        try:
            # Close market data if it has a close method
            if hasattr(self.market_data, 'close'):
                await self.market_data.close()
            logger.info("Chat interface closed")
        except Exception as e:
            logger.error(f"Error closing chat interface: {str(e)}")
        
    async def process_message(self, user_id: str, message: str) -> str:
        """Process a user message and generate a response."""
        try:
            # Get or create chat session
            if user_id not in self._active_chats:
                self._active_chats[user_id] = {
                    "start_time": datetime.now(),
                    "message_count": 0,
                    "last_market_check": None
                }
                
            chat_session = self._active_chats[user_id]
            chat_session["message_count"] += 1
            
            # Get controller's response
            response = self.controller.process_message(message)
            
            # If we have market data to share, add it naturally
            if self._should_update_market_data(chat_session):
                market_data = await self._get_market_update()
                chat_session["last_market_check"] = datetime.now()
                
                # Add market updates naturally to the conversation
                for token, data in market_data.items():
                    update = self._format_market_update(token, data)
                    self.controller.add_pending_update(update)
            else:
                market_data = {}
                
            # Add any pending updates to the response
            pending_updates = self.controller.get_pending_updates()
            if pending_updates and response:
                response += "\n\n" + "\n".join(update["message"] for update in pending_updates)
            
            return response or "I'm here to help! What's on your mind? 💭"
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return "Oops, something went wrong. Let me check what's happening... 🔍"
            
    def _format_market_update(self, token: str, data: Dict) -> str:
        """Format market data in a natural way."""
        try:
            price = data.get("price", "N/A")
            volume = data.get("volume", "N/A")
            change = data.get("price_change", 0)
            
            if change > 0.05:
                return f"By the way, {token} is looking pretty good! Up {change:.1%} at ${price} with {volume} volume 🚀"
            elif change < -0.05:
                return f"Just a heads up, {token} is down {abs(change):.1%} to ${price}. Volume at {volume} 📉"
            else:
                return f"Quick update: {token} is at ${price} with {volume} volume. Pretty stable right now 📊"
                
        except Exception as e:
            logger.error(f"Error formatting market update: {str(e)}")
            return f"Got some data for {token}, but having trouble formatting it..."
            
    def _should_update_market_data(self, chat_session: Dict) -> bool:
        """Check if we should update market data."""
        if not chat_session["last_market_check"]:
            return True
            
        time_since_last_check = datetime.now() - chat_session["last_market_check"]
        return time_since_last_check.total_seconds() > 300  # 5 minutes
        
    async def _get_market_update(self) -> Dict:
        """Get latest market data."""
        try:
            # Get data for favorite tokens
            favorite_tokens = self.personality.get_personality_metrics().get("favorite_tokens", [])
            if not favorite_tokens:
                return {}
                
            market_data = {}
            for token in favorite_tokens:
                data = await self.market_data.get_market_data(token)
                if data:
                    market_data[token] = data
                    
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting market update: {str(e)}")
            return {}
            
    async def get_chat_metrics(self, user_id: str) -> Dict:
        """Get metrics for a chat session."""
        try:
            if user_id not in self._active_chats:
                return {"error": "No active chat session"}
                
            chat_session = self._active_chats[user_id]
            personality_metrics = self.personality.get_personality_metrics()
            conversation_state = self.controller.get_conversation_state()
            
            return {
                "session_duration": (datetime.now() - chat_session["start_time"]).total_seconds(),
                "message_count": chat_session["message_count"],
                "mood": conversation_state["mood"],
                "tone": conversation_state["tone"],
                "personality": personality_metrics,
                "last_market_check": chat_session["last_market_check"].isoformat() if chat_session["last_market_check"] else None
            }
            
        except Exception as e:
            logger.error(f"Error getting chat metrics: {str(e)}")
            return {"error": str(e)} 