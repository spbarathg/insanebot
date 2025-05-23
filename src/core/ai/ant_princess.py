from typing import Dict, List, Optional
import asyncio
import logging
from datetime import datetime
from config.ant_princess_config import ANT_PRINCESS_CONFIG
from .grok_engine import GrokEngine
from ..trading.trade_execution import TradeExecution

logger = logging.getLogger(__name__)

class AntPrincess:
    def __init__(self, grok_engine: GrokEngine):
        self.config = ANT_PRINCESS_CONFIG
        self.grok_engine = grok_engine
        self.trade_execution = TradeExecution()
        self._performance_score = 0.0
        self._trade_history = []
        self._last_analysis = {}
        
    async def initialize(self):
        """Initialize the Ant Princess."""
        await self.trade_execution.initialize()
        
    async def close(self):
        """Close the Ant Princess."""
        await self.trade_execution.close()
        
    async def analyze_opportunity(self, market_data: Dict, wallet_data: Dict) -> Dict:
        """Analyze a trading opportunity."""
        try:
            # Get market analysis
            market_analysis = await self.grok_engine.analyze_market(market_data)
            if "error" in market_analysis:
                return {"error": market_analysis["error"]}
                
            # Get sentiment analysis
            sentiment = await self.grok_engine.get_market_sentiment(market_data["token_address"])
            if "error" in sentiment:
                return {"error": sentiment["error"]}
                
            # Combine analyses
            opportunity = self._combine_analyses(market_analysis, sentiment, wallet_data)
            
            # Update last analysis
            self._last_analysis = opportunity
            
            # Execute trade if opportunity is good
            if opportunity["action"] != "hold":
                trade_result = await self.trade_execution.execute_trade(
                    market_data["token_address"],
                    opportunity["position_size"],
                    opportunity["action"]
                )
                self.update_performance(trade_result)
                opportunity["trade_result"] = trade_result
            
            return opportunity
            
        except Exception as e:
            logger.error(f"Error analyzing opportunity: {str(e)}")
            return {"error": str(e)}
            
    def _combine_analyses(self, market_analysis: Dict, sentiment: Dict, wallet_data: Dict) -> Dict:
        """Combine different analyses into a trading decision."""
        try:
            # Calculate opportunity score
            market_score = market_analysis.get("confidence", 0.0)
            sentiment_score = sentiment.get("score", 0.0)
            wallet_score = self._analyze_wallet_activity(wallet_data)
            
            # Weight the scores
            opportunity_score = (
                market_score * self.config["market_weight"] +
                sentiment_score * self.config["sentiment_weight"] +
                wallet_score * self.config["wallet_weight"]
            )
            
            # Determine action and position size
            action = self._determine_action(opportunity_score)
            position_size = self._calculate_position_size(opportunity_score)
            
            return {
                "opportunity_score": opportunity_score,
                "action": action,
                "position_size": position_size,
                "market_analysis": market_analysis,
                "sentiment": sentiment,
                "wallet_analysis": wallet_score,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error combining analyses: {str(e)}")
            return {"error": "Failed to combine analyses"}
            
    def _analyze_wallet_activity(self, wallet_data: Dict) -> float:
        """Analyze wallet activity to determine its significance."""
        try:
            transactions = wallet_data.get("transactions", [])
            balances = wallet_data.get("balances", [])
            
            # Calculate activity score
            tx_count = len(transactions)
            balance_value = sum(float(b.get("account", {}).get("data", {}).get("parsed", {}).get("info", {}).get("tokenAmount", {}).get("uiAmount", 0)) for b in balances)
            
            # Normalize scores
            tx_score = min(tx_count / 10, 1.0)  # Normalize to 10 transactions
            balance_score = min(balance_value / 1000, 1.0)  # Normalize to 1000 tokens
            
            return (tx_score * 0.6) + (balance_score * 0.4)
            
        except Exception as e:
            logger.error(f"Error analyzing wallet activity: {str(e)}")
            return 0.0
            
    def _determine_action(self, opportunity_score: float) -> str:
        """Determine the trading action based on the opportunity score."""
        try:
            if opportunity_score >= self.config["buy_threshold"]:
                return "buy"
            elif opportunity_score <= self.config["sell_threshold"]:
                return "sell"
            return "hold"
            
        except Exception as e:
            logger.error(f"Error determining action: {str(e)}")
            return "hold"
            
    def _calculate_position_size(self, opportunity_score: float) -> float:
        """Calculate position size based on opportunity score."""
        try:
            # Scale position size based on opportunity score
            base_size = self.config["base_position_size"]
            max_size = self.config["max_position_size"]
            
            # Linear scaling between base and max size
            position_size = base_size + (max_size - base_size) * abs(opportunity_score)
            
            return min(position_size, max_size)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return self.config["base_position_size"]
            
    def update_performance(self, trade_result: Dict):
        """Update performance score based on trade result."""
        try:
            # Calculate trade performance
            if trade_result.get("success", False):
                profit = trade_result.get("profit", 0.0)
                risk = trade_result.get("risk", 1.0)
                
                # Update performance score
                self._performance_score = (
                    self._performance_score * 0.9 +  # Decay old score
                    (profit / risk) * 0.1  # Add new score
                )
                
                # Record trade
                self._trade_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "profit": profit,
                    "risk": risk,
                    "performance": self._performance_score
                })
                
        except Exception as e:
            logger.error(f"Error updating performance: {str(e)}")
            
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics."""
        try:
            return {
                "performance_score": self._performance_score,
                "trade_count": len(self._trade_history),
                "last_analysis": self._last_analysis,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {"error": str(e)}
            
    def should_multiply(self) -> bool:
        """Determine if the Ant Princess should multiply."""
        return (
            self._performance_score >= self.config["multiplication_thresholds"]["performance_score"] and
            len(self._trade_history) >= self.config["multiplication_thresholds"]["experience_threshold"]
        ) 