from typing import Dict, List, Optional
import asyncio
import aiohttp
import json
import logging
from ..config import AI_CONFIG

logger = logging.getLogger(__name__)

class GrokEngine:
    def __init__(self):
        self.config = AI_CONFIG
        self._session = None
        self._api_key = self.config["grok_api_key"]
        self._base_url = self.config["grok_api_url"]
        
    async def initialize(self):
        """Initialize the Grok engine."""
        if not self._api_key:
            raise ValueError("Grok API key not configured")
            
        self._session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self._api_key}"}
        )
        
    async def close(self):
        """Close the Grok engine."""
        if self._session:
            await self._session.close()
            
    async def analyze_market(self, market_data: Dict) -> Dict:
        """Analyze market data using Grok."""
        try:
            if not self._session:
                await self.initialize()
                
            # Prepare analysis request
            request_data = {
                "market_data": market_data,
                "analysis_type": "market_trends",
                "timeframe": self.config["analysis_timeframe"]
            }
            
            # Send request to Grok API
            async with self._session.post(
                f"{self._base_url}/analyze",
                json=request_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return self._process_analysis(result)
                else:
                    error_text = await response.text()
                    logger.error(f"Grok API error: {error_text}")
                    return {"error": "Analysis failed"}
                    
        except Exception as e:
            logger.error(f"Error analyzing market: {str(e)}")
            return {"error": str(e)}
            
    def _process_analysis(self, analysis_result: Dict) -> Dict:
        """Process and validate the analysis result."""
        try:
            # Extract key insights
            insights = {
                "trend": analysis_result.get("trend", "neutral"),
                "confidence": analysis_result.get("confidence", 0.0),
                "key_factors": analysis_result.get("key_factors", []),
                "recommendation": analysis_result.get("recommendation", "hold")
            }
            
            # Validate confidence score
            insights["confidence"] = max(0.0, min(1.0, insights["confidence"]))
            
            return insights
            
        except Exception as e:
            logger.error(f"Error processing analysis: {str(e)}")
            return {"error": "Failed to process analysis"}
            
    async def get_market_sentiment(self, token_address: str) -> Dict:
        """Get market sentiment for a token."""
        try:
            if not self._session:
                await self.initialize()
                
            # Prepare sentiment request
            request_data = {
                "token_address": token_address,
                "timeframe": self.config["sentiment_timeframe"]
            }
            
            # Send request to Grok API
            async with self._session.post(
                f"{self._base_url}/sentiment",
                json=request_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return self._process_sentiment(result)
                else:
                    error_text = await response.text()
                    logger.error(f"Grok API error: {error_text}")
                    return {"error": "Sentiment analysis failed"}
                    
        except Exception as e:
            logger.error(f"Error getting market sentiment: {str(e)}")
            return {"error": str(e)}
            
    def _process_sentiment(self, sentiment_result: Dict) -> Dict:
        """Process and validate the sentiment result."""
        try:
            # Extract sentiment data
            sentiment = {
                "score": sentiment_result.get("score", 0.0),
                "magnitude": sentiment_result.get("magnitude", 0.0),
                "sources": sentiment_result.get("sources", []),
                "timestamp": sentiment_result.get("timestamp", "")
            }
            
            # Validate sentiment score
            sentiment["score"] = max(-1.0, min(1.0, sentiment["score"]))
            sentiment["magnitude"] = max(0.0, min(1.0, sentiment["magnitude"]))
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error processing sentiment: {str(e)}")
            return {"error": "Failed to process sentiment"} 