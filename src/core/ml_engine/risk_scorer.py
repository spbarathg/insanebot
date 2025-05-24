"""
Risk Scoring Engine for Trading Decisions

This module provides comprehensive risk assessment including:
- Volatility risk analysis
- Liquidity risk evaluation 
- Market cap and holder concentration risks
- Smart money and whale activity risks
- Technical and fundamental risk factors
- Overall risk scoring and position sizing recommendations
"""

import time
import logging
import math
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import numpy as np

from .ml_types import RiskScore

logger = logging.getLogger(__name__)

class RiskScorer:
    """Advanced risk scoring for token trading decisions"""
    
    def __init__(self):
        """Initialize the risk scorer"""
        self.risk_cache = {}
        self.risk_history = {}
        self.volatility_cache = {}
        self.cache_duration = 300  # 5 minutes
        
        # Risk scoring weights
        self.risk_weights = {
            'volatility_risk': 0.25,
            'liquidity_risk': 0.20,
            'market_cap_risk': 0.15,
            'holder_concentration_risk': 0.15,
            'smart_money_risk': 0.10,
            'technical_risk': 0.10,
            'fundamental_risk': 0.05
        }
        
        # Risk thresholds for categorization
        self.risk_thresholds = {
            'low': 0.25,
            'medium': 0.5,
            'high': 0.75
        }
        
        logger.info("RiskScorer initialized")
    
    async def initialize(self) -> bool:
        """Initialize the risk scorer"""
        try:
            logger.info("⚠️ Initializing risk scoring engine...")
            
            # Initialize risk scoring parameters
            self.position_size_limits = {
                'low_risk': 0.05,      # 5% max position for low risk
                'medium_risk': 0.03,   # 3% max position for medium risk
                'high_risk': 0.015,    # 1.5% max position for high risk
                'extreme_risk': 0.005  # 0.5% max position for extreme risk
            }
            
            # Stop loss levels based on risk
            self.stop_loss_multipliers = {
                'low_risk': 0.05,      # 5% stop loss
                'medium_risk': 0.08,   # 8% stop loss
                'high_risk': 0.12,     # 12% stop loss
                'extreme_risk': 0.20   # 20% stop loss
            }
            
            logger.info("✅ Risk scoring engine initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize risk scorer: {str(e)}")
            return False
    
    def _calculate_volatility_risk(self, price_history: List[float], token_data: Dict) -> Tuple[float, List[str]]:
        """Calculate volatility-based risk"""
        try:
            if not price_history or len(price_history) < 5:
                return 0.5, ["Insufficient price history"]
            
            risk_factors = []
            volatility_risk = 0.0
            
            # Calculate returns
            returns = [(price_history[i] / price_history[i-1] - 1) for i in range(1, len(price_history))]
            
            if len(returns) < 2:
                return 0.5, ["Insufficient return data"]
            
            # Standard deviation of returns
            volatility = np.std(returns)
            
            # Risk scoring based on volatility
            if volatility > 0.3:  # >30% volatility
                volatility_risk += 0.8
                risk_factors.append("Extremely high volatility (>30%)")
            elif volatility > 0.2:  # >20% volatility
                volatility_risk += 0.6
                risk_factors.append("Very high volatility (>20%)")
            elif volatility > 0.1:  # >10% volatility
                volatility_risk += 0.4
                risk_factors.append("High volatility (>10%)")
            elif volatility > 0.05:  # >5% volatility
                volatility_risk += 0.2
                risk_factors.append("Moderate volatility")
            else:
                volatility_risk += 0.1
                risk_factors.append("Low volatility")
            
            # Calculate maximum drawdown
            peak = price_history[0]
            max_drawdown = 0
            
            for price in price_history[1:]:
                if price > peak:
                    peak = price
                else:
                    drawdown = (peak - price) / peak
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
            
            # Additional risk from drawdown
            if max_drawdown > 0.5:  # >50% drawdown
                volatility_risk += 0.3
                risk_factors.append(f"Severe drawdown ({max_drawdown:.1%})")
            elif max_drawdown > 0.3:  # >30% drawdown
                volatility_risk += 0.2
                risk_factors.append(f"High drawdown ({max_drawdown:.1%})")
            elif max_drawdown > 0.2:  # >20% drawdown
                volatility_risk += 0.1
                risk_factors.append(f"Moderate drawdown ({max_drawdown:.1%})")
            
            # Price stability over recent period
            if len(price_history) >= 10:
                recent_prices = price_history[-10:]
                recent_volatility = np.std([(recent_prices[i] / recent_prices[i-1] - 1) for i in range(1, len(recent_prices))])
                
                if recent_volatility > volatility * 1.5:  # Recent volatility much higher
                    volatility_risk += 0.2
                    risk_factors.append("Increasing volatility trend")
            
            return min(1.0, volatility_risk), risk_factors
            
        except Exception as e:
            logger.error(f"Error calculating volatility risk: {str(e)}")
            return 0.5, ["Error in volatility calculation"]
    
    def _calculate_liquidity_risk(self, token_data: Dict) -> Tuple[float, List[str]]:
        """Calculate liquidity-based risk"""
        try:
            risk_factors = []
            liquidity_risk = 0.0
            
            liquidity_usd = token_data.get('liquidity_usd', 0)
            volume_24h = token_data.get('volumeUsd24h', 0)
            market_cap = token_data.get('market_cap', 0)
            
            # Absolute liquidity risk
            if liquidity_usd < 1000:  # <$1K liquidity
                liquidity_risk += 0.9
                risk_factors.append("Extremely low liquidity (<$1K)")
            elif liquidity_usd < 5000:  # <$5K liquidity
                liquidity_risk += 0.7
                risk_factors.append("Very low liquidity (<$5K)")
            elif liquidity_usd < 25000:  # <$25K liquidity
                liquidity_risk += 0.5
                risk_factors.append("Low liquidity (<$25K)")
            elif liquidity_usd < 100000:  # <$100K liquidity
                liquidity_risk += 0.3
                risk_factors.append("Moderate liquidity (<$100K)")
            elif liquidity_usd < 500000:  # <$500K liquidity
                liquidity_risk += 0.1
                risk_factors.append("Good liquidity")
            else:
                risk_factors.append("Excellent liquidity")
            
            # Volume to liquidity ratio
            if liquidity_usd > 0:
                volume_liquidity_ratio = volume_24h / liquidity_usd
                
                if volume_liquidity_ratio > 5.0:  # Very high turnover
                    liquidity_risk += 0.3
                    risk_factors.append("Extremely high volume/liquidity ratio")
                elif volume_liquidity_ratio > 2.0:  # High turnover
                    liquidity_risk += 0.1
                    risk_factors.append("High volume/liquidity ratio")
                elif volume_liquidity_ratio < 0.1:  # Very low activity
                    liquidity_risk += 0.2
                    risk_factors.append("Very low trading activity")
            
            # Market cap to liquidity ratio
            if market_cap > 0 and liquidity_usd > 0:
                mcap_liquidity_ratio = liquidity_usd / market_cap
                
                if mcap_liquidity_ratio < 0.01:  # <1% of market cap in liquidity
                    liquidity_risk += 0.3
                    risk_factors.append("Very low liquidity relative to market cap")
                elif mcap_liquidity_ratio < 0.05:  # <5% of market cap in liquidity
                    liquidity_risk += 0.1
                    risk_factors.append("Low liquidity relative to market cap")
            
            return min(1.0, liquidity_risk), risk_factors
            
        except Exception as e:
            logger.error(f"Error calculating liquidity risk: {str(e)}")
            return 0.5, ["Error in liquidity calculation"]
    
    def _calculate_market_cap_risk(self, token_data: Dict) -> Tuple[float, List[str]]:
        """Calculate market cap-based risk"""
        try:
            risk_factors = []
            mcap_risk = 0.0
            
            market_cap = token_data.get('market_cap', 0)
            
            if market_cap == 0:
                return 0.8, ["No market cap data"]
            
            # Market cap size risk
            if market_cap < 100000:  # <$100K market cap
                mcap_risk += 0.8
                risk_factors.append("Micro cap (<$100K) - extreme risk")
            elif market_cap < 1000000:  # <$1M market cap
                mcap_risk += 0.6
                risk_factors.append("Very small cap (<$1M) - high risk")
            elif market_cap < 10000000:  # <$10M market cap
                mcap_risk += 0.4
                risk_factors.append("Small cap (<$10M) - moderate risk")
            elif market_cap < 100000000:  # <$100M market cap
                mcap_risk += 0.2
                risk_factors.append("Mid cap (<$100M) - low-moderate risk")
            else:  # >$100M market cap
                mcap_risk += 0.1
                risk_factors.append("Large cap (>$100M) - lower risk")
            
            # Age and stability assessment (simulated)
            # In practice, this would look at token creation date and price history length
            holders = token_data.get('holders', 0)
            
            if holders < 50:  # Very few holders suggests new/risky token
                mcap_risk += 0.2
                risk_factors.append("Very few holders - likely new token")
            elif holders < 200:
                mcap_risk += 0.1
                risk_factors.append("Limited holder base")
            
            return min(1.0, mcap_risk), risk_factors
            
        except Exception as e:
            logger.error(f"Error calculating market cap risk: {str(e)}")
            return 0.5, ["Error in market cap calculation"]
    
    def _calculate_holder_concentration_risk(self, token_data: Dict, holders_data: List[Dict] = None) -> Tuple[float, List[str]]:
        """Calculate holder concentration risk"""
        try:
            risk_factors = []
            concentration_risk = 0.0
            
            if not holders_data:
                return 0.4, ["No holder distribution data"]
            
            # Top holder concentration
            top_holder_percentage = holders_data[0].get('percentage', 0) if holders_data else 0
            
            if top_holder_percentage > 0.7:  # >70% held by top holder
                concentration_risk += 0.9
                risk_factors.append(f"Extreme concentration: top holder has {top_holder_percentage:.1%}")
            elif top_holder_percentage > 0.5:  # >50% held by top holder
                concentration_risk += 0.7
                risk_factors.append(f"Very high concentration: top holder has {top_holder_percentage:.1%}")
            elif top_holder_percentage > 0.3:  # >30% held by top holder
                concentration_risk += 0.5
                risk_factors.append(f"High concentration: top holder has {top_holder_percentage:.1%}")
            elif top_holder_percentage > 0.15:  # >15% held by top holder
                concentration_risk += 0.3
                risk_factors.append(f"Moderate concentration: top holder has {top_holder_percentage:.1%}")
            elif top_holder_percentage > 0.05:  # >5% held by top holder
                concentration_risk += 0.1
                risk_factors.append(f"Low concentration: top holder has {top_holder_percentage:.1%}")
            else:
                risk_factors.append("Very distributed ownership")
            
            # Top 10 holders concentration
            if len(holders_data) >= 10:
                top_10_percentage = sum(holder.get('percentage', 0) for holder in holders_data[:10])
                
                if top_10_percentage > 0.9:  # >90% held by top 10
                    concentration_risk += 0.4
                    risk_factors.append(f"Top 10 holders control {top_10_percentage:.1%}")
                elif top_10_percentage > 0.8:  # >80% held by top 10
                    concentration_risk += 0.2
                    risk_factors.append(f"Top 10 holders control {top_10_percentage:.1%}")
                elif top_10_percentage < 0.5:  # <50% held by top 10
                    concentration_risk -= 0.1  # Reduce risk for good distribution
                    risk_factors.append("Well-distributed ownership among top holders")
            
            # Number of significant holders (>1% each)
            significant_holders = len([h for h in holders_data if h.get('percentage', 0) > 0.01])
            
            if significant_holders < 5:
                concentration_risk += 0.2
                risk_factors.append(f"Only {significant_holders} significant holders")
            elif significant_holders > 20:
                concentration_risk -= 0.1
                risk_factors.append("Many significant holders")
            
            return min(1.0, max(0.0, concentration_risk)), risk_factors
            
        except Exception as e:
            logger.error(f"Error calculating holder concentration risk: {str(e)}")
            return 0.4, ["Error in concentration calculation"]
    
    def _calculate_smart_money_risk(self, token_data: Dict, price_history: List[float] = None) -> Tuple[float, List[str]]:
        """Calculate smart money and whale activity risk"""
        try:
            risk_factors = []
            smart_money_risk = 0.0
            
            liquidity_usd = token_data.get('liquidity_usd', 0)
            volume_24h = token_data.get('volumeUsd24h', 0)
            market_cap = token_data.get('market_cap', 0)
            
            # Volume to market cap analysis
            if market_cap > 0:
                volume_mcap_ratio = volume_24h / market_cap
                
                # Very high volume could indicate manipulation or dumping
                if volume_mcap_ratio > 2.0:  # >200% daily turnover
                    smart_money_risk += 0.6
                    risk_factors.append("Extremely high turnover - possible manipulation")
                elif volume_mcap_ratio > 1.0:  # >100% daily turnover
                    smart_money_risk += 0.3
                    risk_factors.append("Very high turnover")
                elif volume_mcap_ratio < 0.01:  # <1% daily turnover
                    smart_money_risk += 0.2
                    risk_factors.append("Very low activity - illiquid")
            
            # Liquidity manipulation risk
            if liquidity_usd > 0 and volume_24h > 0:
                volume_liquidity_ratio = volume_24h / liquidity_usd
                
                # Extremely high volume relative to liquidity suggests potential manipulation
                if volume_liquidity_ratio > 10.0:
                    smart_money_risk += 0.4
                    risk_factors.append("Potential liquidity manipulation")
                elif volume_liquidity_ratio > 5.0:
                    smart_money_risk += 0.2
                    risk_factors.append("High volume/liquidity ratio")
            
            # Price action analysis for smart money activity
            if price_history and len(price_history) >= 10:
                recent_prices = price_history[-10:]
                
                # Check for unusual price movements
                price_changes = [(recent_prices[i] / recent_prices[i-1] - 1) for i in range(1, len(recent_prices))]
                max_single_move = max(abs(change) for change in price_changes)
                
                if max_single_move > 0.5:  # >50% single move
                    smart_money_risk += 0.4
                    risk_factors.append("Extreme price volatility - possible manipulation")
                elif max_single_move > 0.3:  # >30% single move
                    smart_money_risk += 0.2
                    risk_factors.append("High price volatility")
                
                # Check for consistent direction (possible pump/dump)
                positive_moves = sum(1 for change in price_changes if change > 0.05)
                negative_moves = sum(1 for change in price_changes if change < -0.05)
                
                if positive_moves >= 7:  # Mostly pumping
                    smart_money_risk += 0.3
                    risk_factors.append("Consistent upward pressure - possible pump")
                elif negative_moves >= 7:  # Mostly dumping
                    smart_money_risk += 0.3
                    risk_factors.append("Consistent downward pressure - possible dump")
            
            # Market cap vs volume analysis for institutional interest
            if market_cap > 10000000:  # Large enough for institutional attention
                smart_money_risk -= 0.1  # Reduce risk
                risk_factors.append("Size suggests institutional attention")
            elif market_cap < 100000:  # Too small for serious money
                smart_money_risk += 0.2
                risk_factors.append("Too small for serious institutional money")
            
            return min(1.0, max(0.0, smart_money_risk)), risk_factors
            
        except Exception as e:
            logger.error(f"Error calculating smart money risk: {str(e)}")
            return 0.4, ["Error in smart money calculation"]
    
    def _calculate_technical_risk(self, token_data: Dict, price_history: List[float] = None) -> Tuple[float, List[str]]:
        """Calculate technical analysis-based risk"""
        try:
            risk_factors = []
            technical_risk = 0.0
            
            if not price_history or len(price_history) < 10:
                return 0.4, ["Insufficient price history for technical analysis"]
            
            current_price = price_history[-1]
            
            # Trend analysis
            if len(price_history) >= 20:
                # Short vs long term trend
                short_trend = (price_history[-1] - price_history[-5]) / price_history[-5]
                long_trend = (price_history[-1] - price_history[-20]) / price_history[-20]
                
                # Diverging trends indicate instability
                if short_trend > 0.1 and long_trend < -0.1:
                    technical_risk += 0.3
                    risk_factors.append("Short-term pump on long-term downtrend")
                elif short_trend < -0.1 and long_trend > 0.1:
                    technical_risk += 0.2
                    risk_factors.append("Short-term decline on long-term uptrend")
            
            # Support/resistance analysis
            prices_array = np.array(price_history)
            
            # Find recent highs and lows
            recent_high = np.max(prices_array[-10:]) if len(prices_array) >= 10 else current_price
            recent_low = np.min(prices_array[-10:]) if len(prices_array) >= 10 else current_price
            
            # Position within recent range
            if recent_high != recent_low:
                range_position = (current_price - recent_low) / (recent_high - recent_low)
                
                if range_position > 0.9:  # Near recent high
                    technical_risk += 0.2
                    risk_factors.append("Near recent high - resistance risk")
                elif range_position < 0.1:  # Near recent low
                    technical_risk += 0.2
                    risk_factors.append("Near recent low - support break risk")
            
            # Moving average analysis (simplified)
            if len(price_history) >= 20:
                ma_10 = np.mean(price_history[-10:])
                ma_20 = np.mean(price_history[-20:])
                
                # Price relative to moving averages
                if current_price < ma_10 * 0.9:  # >10% below short MA
                    technical_risk += 0.2
                    risk_factors.append("Significantly below short-term average")
                elif current_price > ma_10 * 1.1:  # >10% above short MA
                    technical_risk += 0.1
                    risk_factors.append("Extended above short-term average")
                
                # Moving average relationship
                if ma_10 < ma_20 * 0.95:  # Short MA significantly below long MA
                    technical_risk += 0.1
                    risk_factors.append("Short-term average below long-term (bearish)")
            
            # Volume-price divergence (simplified)
            volume_24h = token_data.get('volumeUsd24h', 0)
            market_cap = token_data.get('market_cap', 0)
            
            if len(price_history) >= 5 and market_cap > 0:
                recent_price_change = (price_history[-1] - price_history[-5]) / price_history[-5]
                volume_ratio = volume_24h / market_cap
                
                # Low volume on price increases is concerning
                if recent_price_change > 0.1 and volume_ratio < 0.05:
                    technical_risk += 0.2
                    risk_factors.append("Price increase on low volume")
            
            return min(1.0, technical_risk), risk_factors
            
        except Exception as e:
            logger.error(f"Error calculating technical risk: {str(e)}")
            return 0.4, ["Error in technical calculation"]
    
    def _calculate_fundamental_risk(self, token_data: Dict) -> Tuple[float, List[str]]:
        """Calculate fundamental analysis-based risk"""
        try:
            risk_factors = []
            fundamental_risk = 0.0
            
            # Token name and symbol quality
            token_name = token_data.get('name', '').lower()
            token_symbol = token_data.get('symbol', '').lower()
            
            # Basic quality indicators
            if not token_name or len(token_name) < 2:
                fundamental_risk += 0.3
                risk_factors.append("No proper token name")
            elif not token_name.replace(' ', '').isalpha():
                fundamental_risk += 0.2
                risk_factors.append("Non-standard token name")
            
            if not token_symbol or len(token_symbol) < 2:
                fundamental_risk += 0.3
                risk_factors.append("No proper token symbol")
            elif len(token_symbol) > 8:
                fundamental_risk += 0.1
                risk_factors.append("Unusually long symbol")
            
            # Market presence and adoption
            holders = token_data.get('holders', 0)
            market_cap = token_data.get('market_cap', 0)
            
            # Holder adoption
            if holders < 10:
                fundamental_risk += 0.4
                risk_factors.append("Very few holders - minimal adoption")
            elif holders < 50:
                fundamental_risk += 0.2
                risk_factors.append("Limited adoption")
            elif holders > 1000:
                fundamental_risk -= 0.1  # Reduce risk for good adoption
                risk_factors.append("Good adoption")
            
            # Market validation
            if market_cap < 50000:  # <$50K market cap
                fundamental_risk += 0.3
                risk_factors.append("Minimal market validation")
            elif market_cap > 1000000:  # >$1M market cap
                fundamental_risk -= 0.1  # Reduce risk
                risk_factors.append("Significant market validation")
            
            # Utility assessment (basic heuristics)
            # In practice, this would analyze:
            # - Use case and utility
            # - Team and development activity
            # - Partnerships and integrations
            # - Audit status
            
            # For now, use token characteristics as proxy
            if 'test' in token_name or 'meme' in token_name:
                fundamental_risk += 0.2
                risk_factors.append("Appears to be test/meme token")
            
            # Professional naming suggests better fundamentals
            if len(token_symbol) <= 5 and token_symbol.isupper() and token_symbol.isalpha():
                fundamental_risk -= 0.1
                risk_factors.append("Professional token symbol")
            
            return min(1.0, max(0.0, fundamental_risk)), risk_factors
            
        except Exception as e:
            logger.error(f"Error calculating fundamental risk: {str(e)}")
            return 0.4, ["Error in fundamental calculation"]
    
    async def calculate_risk_score(self, token_address: str, token_data: Dict, price_history: List[float] = None, 
                                 holders_data: List[Dict] = None) -> Optional[RiskScore]:
        """Calculate comprehensive risk score for a token"""
        try:
            current_time = time.time()
            
            # Check cache
            cache_key = f"{token_address}_{int(current_time // self.cache_duration)}"
            if cache_key in self.risk_cache:
                return self.risk_cache[cache_key]
            
            logger.debug(f"⚠️ Calculating risk score for {token_data.get('symbol', 'UNKNOWN')}")
            
            # Calculate individual risk components
            volatility_risk, volatility_factors = self._calculate_volatility_risk(price_history, token_data)
            liquidity_risk, liquidity_factors = self._calculate_liquidity_risk(token_data)
            market_cap_risk, mcap_factors = self._calculate_market_cap_risk(token_data)
            concentration_risk, concentration_factors = self._calculate_holder_concentration_risk(token_data, holders_data)
            smart_money_risk, smart_money_factors = self._calculate_smart_money_risk(token_data, price_history)
            technical_risk, technical_factors = self._calculate_technical_risk(token_data, price_history)
            fundamental_risk, fundamental_factors = self._calculate_fundamental_risk(token_data)
            
            # Calculate weighted overall risk score
            overall_risk = (
                volatility_risk * self.risk_weights['volatility_risk'] +
                liquidity_risk * self.risk_weights['liquidity_risk'] +
                market_cap_risk * self.risk_weights['market_cap_risk'] +
                concentration_risk * self.risk_weights['holder_concentration_risk'] +
                smart_money_risk * self.risk_weights['smart_money_risk'] +
                technical_risk * self.risk_weights['technical_risk'] +
                fundamental_risk * self.risk_weights['fundamental_risk']
            )
            
            # Determine risk category
            if overall_risk < self.risk_thresholds['low']:
                risk_category = "low"
            elif overall_risk < self.risk_thresholds['medium']:
                risk_category = "medium"
            elif overall_risk < self.risk_thresholds['high']:
                risk_category = "high"
            else:
                risk_category = "extreme"
            
            # Combine all risk factors
            all_risk_factors = (
                volatility_factors + liquidity_factors + mcap_factors + 
                concentration_factors + smart_money_factors + technical_factors + fundamental_factors
            )
            
            # Calculate position sizing recommendations
            recommended_position_size = self.position_size_limits.get(risk_category, 0.01)
            max_position_size = recommended_position_size * 1.5  # Allow some flexibility
            
            # Calculate stop loss level
            current_price = token_data.get('price_usd', 0)
            stop_loss_multiplier = self.stop_loss_multipliers.get(risk_category, 0.15)
            stop_loss_level = current_price * (1 - stop_loss_multiplier)
            
            # Create result
            result = RiskScore(
                token_address=token_address,
                token_symbol=token_data.get('symbol', 'UNKNOWN'),
                overall_risk_score=overall_risk,
                volatility_risk=volatility_risk,
                liquidity_risk=liquidity_risk,
                market_cap_risk=market_cap_risk,
                holder_concentration_risk=concentration_risk,
                smart_money_risk=smart_money_risk,
                technical_risk=technical_risk,
                fundamental_risk=fundamental_risk,
                sentiment_risk=0.0,  # Will be filled by sentiment analyzer
                risk_category=risk_category,
                risk_factors=all_risk_factors,
                recommended_position_size=recommended_position_size,
                max_position_size=max_position_size,
                stop_loss_level=stop_loss_level,
                analysis_timestamp=current_time
            )
            
            # Cache result
            self.risk_cache[cache_key] = result
            
            # Store in history
            if token_address not in self.risk_history:
                self.risk_history[token_address] = deque(maxlen=100)
            self.risk_history[token_address].append({
                'timestamp': current_time,
                'overall_risk_score': overall_risk,
                'risk_category': risk_category
            })
            
            # Cleanup old cache
            current_bucket = int(current_time // self.cache_duration)
            self.risk_cache = {k: v for k, v in self.risk_cache.items() 
                             if int(k.split('_')[-1]) >= current_bucket - 10}
            
            logger.debug(f"✅ Risk analysis complete for {token_data.get('symbol', 'UNKNOWN')}: {risk_category} ({overall_risk:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating risk score for {token_address}: {str(e)}")
            return None
    
    def get_risk_history(self, token_address: str) -> List[Dict]:
        """Get risk score history for a token"""
        return list(self.risk_history.get(token_address, []))
    
    def get_risk_stats(self) -> Dict[str, Any]:
        """Get statistics about risk scoring"""
        total_analyses = sum(len(history) for history in self.risk_history.values())
        
        risk_distribution = {}
        for history in self.risk_history.values():
            for entry in history:
                risk_category = entry['risk_category']
                risk_distribution[risk_category] = risk_distribution.get(risk_category, 0) + 1
        
        return {
            'total_risk_analyses': total_analyses,
            'tokens_analyzed': len(self.risk_history),
            'risk_distribution': risk_distribution,
            'cache_size': len(self.risk_cache),
            'risk_weights': self.risk_weights,
            'position_size_limits': self.position_size_limits
        } 