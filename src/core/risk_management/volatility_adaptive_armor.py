"""
Volatility-Adaptive Armor - Dynamic Protection Against Market Chaos

This module implements real-time parameter adaptation:
- Dynamic slippage: min(15%, 3× volatility_index)
- Adaptive stop loss: max(15%, current_support_level)  
- Smart position size: min(40%, 2% of liquidity_depth)
- Front-running protection and MEV resistance

SURVIVAL PROTOCOL: Parameters adapt to market conditions in real-time
"""

import asyncio
import time
import math
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import statistics

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market volatility regimes"""
    CALM = "calm"           # <2% hourly volatility
    ACTIVE = "active"       # 2-5% hourly volatility
    VOLATILE = "volatile"   # 5-10% hourly volatility
    CHAOTIC = "chaotic"     # 10-20% hourly volatility
    APOCALYPTIC = "apocalyptic"  # >20% hourly volatility

class LiquidityCondition(Enum):
    """Liquidity depth conditions"""
    DEEP = "deep"           # >1000 SOL depth
    MODERATE = "moderate"   # 100-1000 SOL depth
    SHALLOW = "shallow"     # 10-100 SOL depth
    THIN = "thin"          # 1-10 SOL depth
    DESERT = "desert"      # <1 SOL depth

@dataclass
class VolatilityMetrics:
    """Real-time volatility measurements"""
    price_volatility_1m: float    # 1-minute price volatility
    price_volatility_5m: float    # 5-minute price volatility
    price_volatility_1h: float    # 1-hour price volatility
    volume_volatility: float      # Volume volatility
    spread_volatility: float      # Bid-ask spread volatility
    market_regime: MarketRegime
    volatility_trend: str         # "increasing", "decreasing", "stable"
    confidence: float             # Confidence in measurements
    measurement_time: float = field(default_factory=time.time)

@dataclass
class LiquidityMetrics:
    """Real-time liquidity measurements"""
    total_liquidity_sol: float
    bid_liquidity_sol: float
    ask_liquidity_sol: float
    liquidity_imbalance: float    # bid/ask ratio
    order_book_depth: float       # Depth to 1% slippage
    liquidity_condition: LiquidityCondition
    front_running_risk: float     # 0-1 risk score
    mev_risk_score: float         # 0-1 MEV exploitation risk
    measurement_time: float = field(default_factory=time.time)

@dataclass
class AdaptiveParameters:
    """Dynamic trading parameters"""
    max_slippage_percent: float
    stop_loss_percent: float
    max_position_size_sol: float
    max_position_percent: float
    priority_fee_lamports: int
    transaction_timeout_seconds: int
    
    # Risk adjustments
    volatility_multiplier: float = 1.0
    liquidity_multiplier: float = 1.0
    mev_protection_level: float = 1.0
    
    # Metadata
    calculation_time: float = field(default_factory=time.time)
    market_regime: MarketRegime = MarketRegime.ACTIVE
    reasoning: List[str] = field(default_factory=list)

class VolatilityAnalyzer:
    """Analyzes real-time market volatility"""
    
    def __init__(self):
        self.price_history = deque(maxlen=300)  # 5 minutes at 1s intervals
        self.volume_history = deque(maxlen=60)  # 1 hour at 1m intervals
        self.spread_history = deque(maxlen=100) # Spread history
        
    def add_price_data(self, price: float, volume: float, bid: float, ask: float, timestamp: float = None):
        """Add new market data for volatility analysis"""
        if timestamp is None:
            timestamp = time.time()
            
        self.price_history.append({
            'price': price,
            'volume': volume,
            'timestamp': timestamp
        })
        
        if bid > 0 and ask > 0:
            spread = (ask - bid) / ((ask + bid) / 2)  # Relative spread
            self.spread_history.append({
                'spread': spread,
                'timestamp': timestamp
            })
    
    def calculate_volatility_metrics(self) -> VolatilityMetrics:
        """Calculate comprehensive volatility metrics"""
        try:
            current_time = time.time()
            
            # Calculate price volatilities for different timeframes
            vol_1m = self._calculate_price_volatility(60)    # 1 minute
            vol_5m = self._calculate_price_volatility(300)   # 5 minutes
            vol_1h = self._calculate_price_volatility(3600)  # 1 hour
            
            # Calculate volume volatility
            vol_volume = self._calculate_volume_volatility()
            
            # Calculate spread volatility
            vol_spread = self._calculate_spread_volatility()
            
            # Determine market regime
            max_vol = max(vol_1m, vol_5m, vol_1h)
            if max_vol < 0.02:
                regime = MarketRegime.CALM
            elif max_vol < 0.05:
                regime = MarketRegime.ACTIVE
            elif max_vol < 0.10:
                regime = MarketRegime.VOLATILE
            elif max_vol < 0.20:
                regime = MarketRegime.CHAOTIC
            else:
                regime = MarketRegime.APOCALYPTIC
            
            # Determine volatility trend
            trend = self._calculate_volatility_trend()
            
            # Calculate confidence based on data quality
            confidence = min(1.0, len(self.price_history) / 60.0)  # Need 1 minute of data for confidence
            
            return VolatilityMetrics(
                price_volatility_1m=vol_1m,
                price_volatility_5m=vol_5m,
                price_volatility_1h=vol_1h,
                volume_volatility=vol_volume,
                spread_volatility=vol_spread,
                market_regime=regime,
                volatility_trend=trend,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {str(e)}")
            return VolatilityMetrics(0.0, 0.0, 0.0, 0.0, 0.0, MarketRegime.ACTIVE, "unknown", 0.0)
    
    def _calculate_price_volatility(self, lookback_seconds: int) -> float:
        """Calculate price volatility over specified period"""
        try:
            cutoff_time = time.time() - lookback_seconds
            recent_prices = [
                data['price'] for data in self.price_history
                if data['timestamp'] >= cutoff_time and data['price'] > 0
            ]
            
            if len(recent_prices) < 2:
                return 0.0
            
            # Calculate returns
            returns = []
            for i in range(1, len(recent_prices)):
                if recent_prices[i-1] > 0:
                    ret = (recent_prices[i] / recent_prices[i-1]) - 1
                    returns.append(ret)
            
            if len(returns) < 2:
                return 0.0
            
            # Return standard deviation of returns (volatility)
            return float(np.std(returns))
            
        except Exception as e:
            logger.error(f"Error calculating price volatility: {str(e)}")
            return 0.0
    
    def _calculate_volume_volatility(self) -> float:
        """Calculate volume volatility"""
        try:
            if len(self.price_history) < 10:
                return 0.0
            
            volumes = [data['volume'] for data in self.price_history if data['volume'] > 0]
            if len(volumes) < 2:
                return 0.0
            
            return float(np.std(volumes) / max(np.mean(volumes), 1e-8))  # Coefficient of variation
            
        except Exception as e:
            logger.error(f"Error calculating volume volatility: {str(e)}")
            return 0.0
    
    def _calculate_spread_volatility(self) -> float:
        """Calculate bid-ask spread volatility"""
        try:
            if len(self.spread_history) < 2:
                return 0.0
            
            spreads = [data['spread'] for data in self.spread_history]
            return float(np.std(spreads))
            
        except Exception as e:
            logger.error(f"Error calculating spread volatility: {str(e)}")
            return 0.0
    
    def _calculate_volatility_trend(self) -> str:
        """Determine if volatility is increasing, decreasing, or stable"""
        try:
            if len(self.price_history) < 60:  # Need at least 1 minute of data
                return "unknown"
            
            # Compare recent volatility to earlier volatility
            mid_point = len(self.price_history) // 2
            recent_prices = [data['price'] for data in list(self.price_history)[mid_point:]]
            earlier_prices = [data['price'] for data in list(self.price_history)[:mid_point]]
            
            recent_vol = np.std([recent_prices[i]/recent_prices[i-1] - 1 
                               for i in range(1, len(recent_prices)) if recent_prices[i-1] > 0])
            earlier_vol = np.std([earlier_prices[i]/earlier_prices[i-1] - 1 
                                for i in range(1, len(earlier_prices)) if earlier_prices[i-1] > 0])
            
            if recent_vol > earlier_vol * 1.2:
                return "increasing"
            elif recent_vol < earlier_vol * 0.8:
                return "decreasing"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error calculating volatility trend: {str(e)}")
            return "unknown"

class LiquidityAnalyzer:
    """Analyzes real-time liquidity conditions"""
    
    def __init__(self):
        self.order_book_history = deque(maxlen=60)  # 1 minute of order book data
        
    def add_order_book_data(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], timestamp: float = None):
        """Add order book data for liquidity analysis"""
        if timestamp is None:
            timestamp = time.time()
            
        self.order_book_history.append({
            'bids': bids,  # [(price, size), ...]
            'asks': asks,  # [(price, size), ...]
            'timestamp': timestamp
        })
    
    def calculate_liquidity_metrics(self, current_price: float) -> LiquidityMetrics:
        """Calculate comprehensive liquidity metrics"""
        try:
            if not self.order_book_history:
                return self._create_empty_liquidity_metrics()
            
            latest_book = self.order_book_history[-1]
            bids = latest_book['bids']
            asks = latest_book['asks']
            
            # Calculate total liquidity
            bid_liquidity = sum(price * size for price, size in bids) if bids else 0
            ask_liquidity = sum(price * size for price, size in asks) if asks else 0
            total_liquidity = bid_liquidity + ask_liquidity
            
            # Calculate liquidity imbalance
            imbalance = bid_liquidity / max(ask_liquidity, 1e-8) if ask_liquidity > 0 else 1.0
            
            # Calculate order book depth (distance to 1% slippage)
            depth = self._calculate_depth_to_slippage(bids, asks, current_price, 0.01)
            
            # Determine liquidity condition
            if total_liquidity > 1000:
                condition = LiquidityCondition.DEEP
            elif total_liquidity > 100:
                condition = LiquidityCondition.MODERATE
            elif total_liquidity > 10:
                condition = LiquidityCondition.SHALLOW
            elif total_liquidity > 1:
                condition = LiquidityCondition.THIN
            else:
                condition = LiquidityCondition.DESERT
            
            # Calculate front-running risk
            front_run_risk = self._calculate_front_running_risk(total_liquidity, depth)
            
            # Calculate MEV risk
            mev_risk = self._calculate_mev_risk(bids, asks, current_price)
            
            return LiquidityMetrics(
                total_liquidity_sol=total_liquidity,
                bid_liquidity_sol=bid_liquidity,
                ask_liquidity_sol=ask_liquidity,
                liquidity_imbalance=imbalance,
                order_book_depth=depth,
                liquidity_condition=condition,
                front_running_risk=front_run_risk,
                mev_risk_score=mev_risk
            )
            
        except Exception as e:
            logger.error(f"Error calculating liquidity metrics: {str(e)}")
            return self._create_empty_liquidity_metrics()
    
    def _calculate_depth_to_slippage(self, bids: List[Tuple[float, float]], 
                                   asks: List[Tuple[float, float]], 
                                   current_price: float, slippage: float) -> float:
        """Calculate order book depth to achieve specific slippage"""
        try:
            target_price_up = current_price * (1 + slippage)
            target_price_down = current_price * (1 - slippage)
            
            # Calculate depth on ask side (buying pressure)
            ask_depth = 0.0
            for price, size in sorted(asks):
                if price <= target_price_up:
                    ask_depth += price * size
                else:
                    break
            
            # Calculate depth on bid side (selling pressure)
            bid_depth = 0.0
            for price, size in sorted(bids, reverse=True):
                if price >= target_price_down:
                    bid_depth += price * size
                else:
                    break
            
            return min(ask_depth, bid_depth)
            
        except Exception as e:
            logger.error(f"Error calculating depth to slippage: {str(e)}")
            return 0.0
    
    def _calculate_front_running_risk(self, total_liquidity: float, depth: float) -> float:
        """Calculate front-running risk based on liquidity conditions"""
        try:
            # Higher risk with lower liquidity and depth
            liquidity_risk = 1.0 / (1.0 + total_liquidity / 100.0)  # Normalize around 100 SOL
            depth_risk = 1.0 / (1.0 + depth / 10.0)  # Normalize around 10 SOL depth
            
            # Combine risks
            front_run_risk = (liquidity_risk + depth_risk) / 2.0
            return min(1.0, front_run_risk)
            
        except Exception as e:
            logger.error(f"Error calculating front-running risk: {str(e)}")
            return 0.5  # Moderate risk as default
    
    def _calculate_mev_risk(self, bids: List[Tuple[float, float]], 
                          asks: List[Tuple[float, float]], current_price: float) -> float:
        """Calculate MEV (Maximum Extractable Value) risk"""
        try:
            if not bids or not asks:
                return 1.0  # Maximum risk if no order book
            
            best_bid = max(bids, key=lambda x: x[0])[0] if bids else 0
            best_ask = min(asks, key=lambda x: x[0])[0] if asks else float('inf')
            
            # Calculate spread
            if best_ask == float('inf') or best_bid == 0:
                return 1.0
            
            spread = (best_ask - best_bid) / current_price
            
            # Higher spread = higher MEV opportunity = higher risk for us
            mev_risk = min(1.0, spread * 10)  # Scale spread to risk
            
            return mev_risk
            
        except Exception as e:
            logger.error(f"Error calculating MEV risk: {str(e)}")
            return 0.5
    
    def _create_empty_liquidity_metrics(self) -> LiquidityMetrics:
        """Create empty metrics for error cases"""
        return LiquidityMetrics(
            total_liquidity_sol=0.0,
            bid_liquidity_sol=0.0,
            ask_liquidity_sol=0.0,
            liquidity_imbalance=1.0,
            order_book_depth=0.0,
            liquidity_condition=LiquidityCondition.DESERT,
            front_running_risk=1.0,
            mev_risk_score=1.0
        )

class VolatilityAdaptiveArmor:
    """Main armor system that adapts parameters to market conditions"""
    
    def __init__(self):
        self.volatility_analyzer = VolatilityAnalyzer()
        self.liquidity_analyzer = LiquidityAnalyzer()
        
        # Base parameters (conservative defaults)
        self.base_params = {
            'max_slippage_percent': 15.0,      # 15% max slippage
            'stop_loss_percent': 15.0,         # 15% base stop loss
            'max_position_size_sol': 2.0,      # 2 SOL base position
            'max_position_percent': 5.0,       # 5% base position (reduced from 40%)
            'priority_fee_lamports': 10000,    # 0.00001 SOL base priority fee
            'transaction_timeout_seconds': 30   # 30 second timeout
        }
        
        # Regime-specific multipliers
        self.regime_multipliers = {
            MarketRegime.CALM: {
                'slippage': 0.5,     # Reduce slippage tolerance in calm markets
                'stop_loss': 0.8,    # Tighter stops in calm markets
                'position_size': 1.5, # Larger positions in calm markets
                'priority_fee': 1.0
            },
            MarketRegime.ACTIVE: {
                'slippage': 1.0,     # Normal parameters
                'stop_loss': 1.0,
                'position_size': 1.0,
                'priority_fee': 1.0
            },
            MarketRegime.VOLATILE: {
                'slippage': 2.0,     # Higher slippage tolerance
                'stop_loss': 1.5,    # Wider stops
                'position_size': 0.7, # Smaller positions
                'priority_fee': 2.0   # Higher priority fees
            },
            MarketRegime.CHAOTIC: {
                'slippage': 3.0,     # Much higher slippage tolerance
                'stop_loss': 2.0,    # Much wider stops
                'position_size': 0.4, # Much smaller positions
                'priority_fee': 5.0   # Much higher priority fees
            },
            MarketRegime.APOCALYPTIC: {
                'slippage': 5.0,     # Extreme slippage tolerance
                'stop_loss': 3.0,    # Extreme stops
                'position_size': 0.2, # Minimal positions
                'priority_fee': 10.0  # Maximum priority fees
            }
        }
        
        logger.info("🛡️ Volatility-Adaptive Armor initialized - Dynamic protection active")
    
    def add_market_data(self, price: float, volume: float, bid: float, ask: float, 
                       order_book: Dict[str, List], timestamp: float = None):
        """Add new market data for analysis"""
        # Add to volatility analyzer
        self.volatility_analyzer.add_price_data(price, volume, bid, ask, timestamp)
        
        # Add to liquidity analyzer
        bids = [(float(p), float(s)) for p, s in order_book.get('bids', [])]
        asks = [(float(p), float(s)) for p, s in order_book.get('asks', [])]
        self.liquidity_analyzer.add_order_book_data(bids, asks, timestamp)
    
    def calculate_adaptive_parameters(self, base_capital_sol: float, 
                                    current_price: float,
                                    support_level: float = None) -> AdaptiveParameters:
        """Calculate adaptive parameters based on current market conditions"""
        try:
            # Get current market metrics
            vol_metrics = self.volatility_analyzer.calculate_volatility_metrics()
            liq_metrics = self.liquidity_analyzer.calculate_liquidity_metrics(current_price)
            
            # Get regime multipliers
            regime = vol_metrics.market_regime
            multipliers = self.regime_multipliers[regime]
            
            reasoning = []
            reasoning.append(f"Market regime: {regime.value}")
            reasoning.append(f"1m volatility: {vol_metrics.price_volatility_1m:.2%}")
            reasoning.append(f"Liquidity: {liq_metrics.total_liquidity_sol:.1f} SOL")
            
            # Calculate adaptive slippage
            volatility_factor = max(vol_metrics.price_volatility_1m, vol_metrics.price_volatility_5m)
            adaptive_slippage = min(
                self.base_params['max_slippage_percent'],
                max(1.0, volatility_factor * 300 * multipliers['slippage'])  # 3x volatility with regime adjustment
            )
            reasoning.append(f"Adaptive slippage: {adaptive_slippage:.1f}%")
            
            # Calculate adaptive stop loss
            if support_level and support_level > 0:
                # Use technical support level if available
                support_distance = (current_price - support_level) / current_price
                technical_stop = max(0.05, support_distance)  # Minimum 5% stop
                
                base_stop = self.base_params['stop_loss_percent'] / 100.0
                adaptive_stop = max(base_stop, technical_stop) * multipliers['stop_loss']
                reasoning.append(f"Using support level: {support_level:.6f}")
            else:
                # Use volatility-based stop loss
                volatility_stop = max(
                    self.base_params['stop_loss_percent'] / 100.0,
                    volatility_factor * 2.0  # 2x volatility for stop loss
                )
                adaptive_stop = volatility_stop * multipliers['stop_loss']
            
            adaptive_stop_percent = adaptive_stop * 100
            reasoning.append(f"Adaptive stop loss: {adaptive_stop_percent:.1f}%")
            
            # Calculate adaptive position size
            # Factor 1: Base capital percentage
            base_position_percent = self.base_params['max_position_percent'] * multipliers['position_size']
            
            # Factor 2: Liquidity constraint (max 2% of available liquidity)
            liquidity_constraint_sol = liq_metrics.total_liquidity_sol * 0.02
            
            # Factor 3: Volatility constraint
            volatility_constraint_percent = max(1.0, base_position_percent * (1.0 - volatility_factor * 5))
            
            # Use most restrictive constraint
            position_from_percent = (base_capital_sol * volatility_constraint_percent) / 100.0
            max_position_sol = min(
                position_from_percent,
                liquidity_constraint_sol,
                self.base_params['max_position_size_sol'] * multipliers['position_size']
            )
            
            reasoning.append(f"Position constraints - Percent: {position_from_percent:.2f}, "
                           f"Liquidity: {liquidity_constraint_sol:.2f}, Final: {max_position_sol:.2f}")
            
            # Calculate adaptive priority fee (based on network congestion proxy)
            mev_risk_multiplier = 1.0 + liq_metrics.mev_risk_score * 2.0  # Up to 3x for high MEV risk
            adaptive_priority_fee = int(
                self.base_params['priority_fee_lamports'] * 
                multipliers['priority_fee'] * 
                mev_risk_multiplier
            )
            reasoning.append(f"Priority fee: {adaptive_priority_fee} lamports (MEV risk: {liq_metrics.mev_risk_score:.2f})")
            
            # Calculate adaptive timeout
            if regime in [MarketRegime.CHAOTIC, MarketRegime.APOCALYPTIC]:
                adaptive_timeout = int(self.base_params['transaction_timeout_seconds'] * 0.5)  # Shorter timeout
                reasoning.append("Reduced timeout for extreme volatility")
            elif liq_metrics.liquidity_condition == LiquidityCondition.DESERT:
                adaptive_timeout = int(self.base_params['transaction_timeout_seconds'] * 2.0)  # Longer timeout
                reasoning.append("Extended timeout for low liquidity")
            else:
                adaptive_timeout = self.base_params['transaction_timeout_seconds']
            
            # Calculate risk multipliers for monitoring
            volatility_multiplier = 1.0 + volatility_factor * 5.0
            liquidity_multiplier = 1.0 + (1.0 - min(1.0, liq_metrics.total_liquidity_sol / 100.0))
            mev_protection_level = 1.0 + liq_metrics.mev_risk_score
            
            return AdaptiveParameters(
                max_slippage_percent=adaptive_slippage,
                stop_loss_percent=adaptive_stop_percent,
                max_position_size_sol=max_position_sol,
                max_position_percent=min(base_position_percent, 
                                       (max_position_sol / base_capital_sol) * 100),
                priority_fee_lamports=adaptive_priority_fee,
                transaction_timeout_seconds=adaptive_timeout,
                volatility_multiplier=volatility_multiplier,
                liquidity_multiplier=liquidity_multiplier,
                mev_protection_level=mev_protection_level,
                market_regime=regime,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error calculating adaptive parameters: {str(e)}")
            return self._get_emergency_parameters(base_capital_sol)
    
    def _get_emergency_parameters(self, base_capital_sol: float) -> AdaptiveParameters:
        """Get ultra-conservative emergency parameters"""
        return AdaptiveParameters(
            max_slippage_percent=5.0,  # Very conservative
            stop_loss_percent=10.0,     # Tight stop
            max_position_size_sol=min(0.1, base_capital_sol * 0.01),  # 1% max position
            max_position_percent=1.0,
            priority_fee_lamports=50000,  # High priority
            transaction_timeout_seconds=15,  # Short timeout
            volatility_multiplier=5.0,
            liquidity_multiplier=5.0,
            mev_protection_level=3.0,
            market_regime=MarketRegime.APOCALYPTIC,
            reasoning=["EMERGENCY MODE: Using ultra-conservative parameters"]
        )
    
    def get_armor_status(self) -> Dict[str, Any]:
        """Get current armor system status"""
        vol_metrics = self.volatility_analyzer.calculate_volatility_metrics()
        liq_metrics = self.liquidity_analyzer.calculate_liquidity_metrics(100.0)  # Dummy price
        
        return {
            "market_regime": vol_metrics.market_regime.value,
            "volatility_1m": f"{vol_metrics.price_volatility_1m:.2%}",
            "volatility_trend": vol_metrics.volatility_trend,
            "liquidity_condition": liq_metrics.liquidity_condition.value,
            "total_liquidity_sol": f"{liq_metrics.total_liquidity_sol:.1f}",
            "front_running_risk": f"{liq_metrics.front_running_risk:.2f}",
            "mev_risk": f"{liq_metrics.mev_risk_score:.2f}",
            "confidence": f"{vol_metrics.confidence:.2f}",
            "data_points": len(self.volatility_analyzer.price_history)
        }
    
    def emergency_lock_down(self) -> AdaptiveParameters:
        """Emergency lockdown with maximum protection"""
        logger.critical("🚨 ARMOR EMERGENCY LOCKDOWN ACTIVATED")
        
        return AdaptiveParameters(
            max_slippage_percent=1.0,      # 1% max slippage
            stop_loss_percent=5.0,         # 5% tight stop
            max_position_size_sol=0.01,    # 0.01 SOL tiny position
            max_position_percent=0.1,      # 0.1% position
            priority_fee_lamports=100000,  # Maximum priority
            transaction_timeout_seconds=10, # Very short timeout
            volatility_multiplier=10.0,
            liquidity_multiplier=10.0,
            mev_protection_level=5.0,
            market_regime=MarketRegime.APOCALYPTIC,
            reasoning=["🚨 EMERGENCY LOCKDOWN: Maximum protection engaged"]
        ) 