"""
Data Layer - Pattern recognition compounding

This layer implements data-driven pattern recognition that compounds over time through:
- Historical data accumulation and analysis
- Pattern mining and extraction improvement  
- Predictive model accuracy enhancement
- Cross-timeframe pattern correlation
- Data quality and preprocessing optimization
"""

import logging
import time
import asyncio
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

@dataclass
class DataPattern:
    """Represents a discovered data pattern"""
    pattern_id: str
    pattern_type: str  # trend, cycle, correlation, anomaly
    timeframe: str     # 1m, 5m, 15m, 1h, 4h, 1d
    features: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    accuracy: float = 0.5
    frequency: int = 0
    last_seen: float = 0.0
    predictive_power: float = 0.0

@dataclass
class DataMetrics:
    """Metrics for tracking data layer performance"""
    total_patterns: int = 0
    total_data_points: int = 0
    avg_pattern_accuracy: float = 0.0
    avg_predictive_power: float = 0.0
    data_quality_score: float = 0.0
    pattern_discovery_rate: float = 0.0
    compound_data_score: float = 1.0

class DataLayer:
    """
    Layer 5: Data compounding
    
    Handles pattern recognition and data analysis that compounds over time,
    creating increasingly accurate predictive capabilities.
    """
    
    def __init__(self):
        self.layer_id = "data_layer"
        self.initialized = False
        
        # Data storage
        self.historical_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.discovered_patterns: Dict[str, DataPattern] = {}
        self.data_metrics = DataMetrics()
        
        # Pattern recognition
        self.pattern_thresholds = {
            "trend": 0.7,
            "cycle": 0.6,
            "correlation": 0.8,
            "anomaly": 0.9
        }
        
        # Compounding effects
        self.compound_data_multiplier = 1.0
        self.pattern_accuracy_multiplier = 1.0
        self.data_quality_multiplier = 1.0
        self.discovery_rate_multiplier = 1.0
        
        # Compound rates
        self.data_compound_rates = {
            "trend": 1.010,      # 1.0% per discovery
            "cycle": 1.015,      # 1.5% per discovery  
            "correlation": 1.020, # 2.0% per discovery
            "anomaly": 1.025     # 2.5% per discovery
        }
        
        logger.info(f"DataLayer {self.layer_id} created")
    
    async def initialize(self) -> bool:
        """Initialize the data layer"""
        try:
            logger.info(f"Initializing DataLayer {self.layer_id}...")
            
            # Initialize pattern recognition
            await self._initialize_pattern_recognition()
            
            # Load historical patterns
            await self._load_historical_patterns()
            
            # Calculate metrics
            await self._calculate_data_metrics()
            
            self.initialized = True
            logger.info(f"DataLayer {self.layer_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing DataLayer {self.layer_id}: {e}")
            return False
    
    async def _initialize_pattern_recognition(self):
        """Initialize basic pattern recognition capabilities"""
        # Initialize with basic market patterns
        basic_patterns = [
            {
                "pattern_id": "uptrend_pattern",
                "pattern_type": "trend",
                "timeframe": "1h",
                "features": ["price", "volume", "momentum"],
                "conditions": {"consecutive_highs": ">= 3", "volume_increasing": True},
                "confidence": 0.8,
                "accuracy": 0.7
            },
            {
                "pattern_id": "reversal_pattern", 
                "pattern_type": "cycle",
                "timeframe": "15m",
                "features": ["price", "rsi", "volume"],
                "conditions": {"rsi_oversold": "< 30", "volume_spike": "> 1.5"},
                "confidence": 0.6,
                "accuracy": 0.65
            }
        ]
        
        for pattern_data in basic_patterns:
            pattern = DataPattern(
                pattern_id=pattern_data["pattern_id"],
                pattern_type=pattern_data["pattern_type"],
                timeframe=pattern_data["timeframe"],
                features=pattern_data["features"],
                conditions=pattern_data["conditions"],
                confidence=pattern_data["confidence"],
                accuracy=pattern_data["accuracy"],
                last_seen=time.time()
            )
            self.discovered_patterns[pattern.pattern_id] = pattern
    
    async def _load_historical_patterns(self):
        """Load historical pattern data"""
        # Initialize compound multipliers
        self.compound_data_multiplier = 1.0
        self.pattern_accuracy_multiplier = 1.0
    
    async def ingest_data(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest new data point and analyze for patterns"""
        try:
            # Store data point
            timeframe = data_point.get("timeframe", "1m")
            self.historical_data[timeframe].append({
                "timestamp": time.time(),
                "data": data_point
            })
            
            # Analyze for patterns with compound efficiency
            patterns_found = await self._analyze_for_patterns(data_point, timeframe)
            
            # Apply compound effects to pattern discovery
            compound_patterns = int(patterns_found * self.discovery_rate_multiplier)
            
            # Update metrics
            self.data_metrics.total_data_points += 1
            await self._update_data_metrics()
            
            result = {
                "success": True,
                "data_points_stored": 1,
                "patterns_discovered": compound_patterns,
                "compound_data_score": self.compound_data_multiplier,
                "timeframe": timeframe
            }
            
            logger.debug(f"Data ingested: {compound_patterns} patterns found")
            return result
            
        except Exception as e:
            logger.error(f"Error ingesting data: {e}")
            return {"success": False, "error": str(e)}
    
    async def _analyze_for_patterns(self, data_point: Dict[str, Any], timeframe: str) -> int:
        """Analyze data point for new patterns"""
        patterns_found = 0
        
        try:
            # Get recent data for analysis
            recent_data = list(self.historical_data[timeframe])[-100:]  # Last 100 points
            
            if len(recent_data) < 10:  # Need minimum data
                return 0
            
            # Extract features
            prices = [d["data"].get("price", 0) for d in recent_data]
            volumes = [d["data"].get("volume", 0) for d in recent_data]
            
            # Trend pattern detection
            if await self._detect_trend_pattern(prices, volumes, timeframe):
                patterns_found += 1
                await self._apply_compound_effect("trend")
            
            # Cycle pattern detection  
            if await self._detect_cycle_pattern(prices, timeframe):
                patterns_found += 1
                await self._apply_compound_effect("cycle")
            
            # Correlation pattern detection
            if await self._detect_correlation_pattern(prices, volumes, timeframe):
                patterns_found += 1
                await self._apply_compound_effect("correlation")
            
            return patterns_found
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            return 0
    
    async def _detect_trend_pattern(self, prices: List[float], volumes: List[float], timeframe: str) -> bool:
        """Detect trend patterns in price data"""
        try:
            if len(prices) < 5:
                return False
            
            # Simple trend detection
            recent_prices = prices[-5:]
            is_uptrend = all(recent_prices[i] >= recent_prices[i-1] for i in range(1, len(recent_prices)))
            is_downtrend = all(recent_prices[i] <= recent_prices[i-1] for i in range(1, len(recent_prices)))
            
            if is_uptrend or is_downtrend:
                pattern_id = f"trend_{timeframe}_{int(time.time())}"
                pattern = DataPattern(
                    pattern_id=pattern_id,
                    pattern_type="trend",
                    timeframe=timeframe,
                    features=["price"],
                    conditions={"direction": "up" if is_uptrend else "down"},
                    confidence=0.7 * self.pattern_accuracy_multiplier,
                    accuracy=0.6,
                    last_seen=time.time(),
                    predictive_power=0.5
                )
                
                self.discovered_patterns[pattern_id] = pattern
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting trend pattern: {e}")
            return False
    
    async def _detect_cycle_pattern(self, prices: List[float], timeframe: str) -> bool:
        """Detect cyclical patterns in price data"""
        try:
            if len(prices) < 20:
                return False
            
            # Simple cycle detection using peaks and troughs
            peaks = []
            troughs = []
            
            for i in range(1, len(prices) - 1):
                if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                    peaks.append(i)
                elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                    troughs.append(i)
            
            # Check for regular intervals
            if len(peaks) >= 3 and len(troughs) >= 3:
                peak_intervals = [peaks[i] - peaks[i-1] for i in range(1, len(peaks))]
                avg_interval = sum(peak_intervals) / len(peak_intervals)
                
                # If intervals are relatively consistent, it's a cycle
                interval_variance = sum(abs(interval - avg_interval) for interval in peak_intervals) / len(peak_intervals)
                
                if interval_variance < avg_interval * 0.3:  # 30% variance threshold
                    pattern_id = f"cycle_{timeframe}_{int(time.time())}"
                    pattern = DataPattern(
                        pattern_id=pattern_id,
                        pattern_type="cycle",
                        timeframe=timeframe,
                        features=["price"],
                        conditions={"cycle_length": avg_interval},
                        confidence=0.6 * self.pattern_accuracy_multiplier,
                        accuracy=0.55,
                        last_seen=time.time(),
                        predictive_power=0.7
                    )
                    
                    self.discovered_patterns[pattern_id] = pattern
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting cycle pattern: {e}")
            return False
    
    async def _detect_correlation_pattern(self, prices: List[float], volumes: List[float], timeframe: str) -> bool:
        """Detect correlation patterns between price and volume"""
        try:
            if len(prices) != len(volumes) or len(prices) < 10:
                return False
            
            # Calculate correlation coefficient
            mean_price = sum(prices) / len(prices)
            mean_volume = sum(volumes) / len(volumes)
            
            numerator = sum((prices[i] - mean_price) * (volumes[i] - mean_volume) for i in range(len(prices)))
            
            price_variance = sum((price - mean_price) ** 2 for price in prices)
            volume_variance = sum((volume - mean_volume) ** 2 for volume in volumes)
            
            if price_variance > 0 and volume_variance > 0:
                correlation = numerator / (price_variance * volume_variance) ** 0.5
                
                if abs(correlation) > 0.6:  # Strong correlation threshold
                    pattern_id = f"correlation_{timeframe}_{int(time.time())}"
                    pattern = DataPattern(
                        pattern_id=pattern_id,
                        pattern_type="correlation",
                        timeframe=timeframe,
                        features=["price", "volume"],
                        conditions={"correlation": correlation},
                        confidence=0.8 * self.pattern_accuracy_multiplier,
                        accuracy=0.7,
                        last_seen=time.time(),
                        predictive_power=abs(correlation)
                    )
                    
                    self.discovered_patterns[pattern_id] = pattern
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting correlation pattern: {e}")
            return False
    
    async def _apply_compound_effect(self, pattern_type: str):
        """Apply compounding effect for pattern discovery"""
        try:
            compound_rate = self.data_compound_rates.get(pattern_type, 1.010)
            
            # Apply compound effect to accuracy
            self.pattern_accuracy_multiplier *= compound_rate
            
            # Apply compound effect to discovery rate
            self.discovery_rate_multiplier *= compound_rate
            
            # Apply compound effect to overall data score
            self.compound_data_multiplier *= compound_rate
            
        except Exception as e:
            logger.error(f"Error applying compound effect: {e}")
    
    async def _calculate_data_metrics(self):
        """Calculate current data metrics"""
        try:
            # Basic counts
            self.data_metrics.total_patterns = len(self.discovered_patterns)
            
            # Average accuracy and predictive power
            if self.discovered_patterns:
                total_accuracy = sum(p.accuracy for p in self.discovered_patterns.values())
                total_predictive = sum(p.predictive_power for p in self.discovered_patterns.values())
                
                self.data_metrics.avg_pattern_accuracy = total_accuracy / len(self.discovered_patterns)
                self.data_metrics.avg_predictive_power = total_predictive / len(self.discovered_patterns)
            
            # Data quality score (based on data volume and pattern consistency)
            total_data_points = sum(len(deque_data) for deque_data in self.historical_data.values())
            self.data_metrics.total_data_points = total_data_points
            
            if total_data_points > 0:
                self.data_metrics.data_quality_score = min(1.0, total_data_points / 1000.0)  # Normalize to 1000 points
            
            # Compound data score
            self.data_metrics.compound_data_score = self.compound_data_multiplier
            
        except Exception as e:
            logger.error(f"Error calculating data metrics: {e}")
    
    async def _update_data_metrics(self):
        """Update data metrics"""
        await self._calculate_data_metrics()
    
    async def get_pattern_predictions(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Get predictions based on discovered patterns"""
        try:
            timeframe = query.get("timeframe", "1m")
            current_data = query.get("current_data", {})
            
            predictions = []
            
            # Find relevant patterns
            relevant_patterns = [
                p for p in self.discovered_patterns.values()
                if p.timeframe == timeframe
            ]
            
            for pattern in relevant_patterns:
                # Check if pattern conditions match current data
                match_score = await self._calculate_pattern_match(pattern, current_data)
                
                if match_score > 0.5:  # 50% match threshold
                    confidence = pattern.confidence * match_score * self.pattern_accuracy_multiplier
                    
                    predictions.append({
                        "pattern_id": pattern.pattern_id,
                        "pattern_type": pattern.pattern_type,
                        "prediction_confidence": min(0.99, confidence),
                        "predictive_power": pattern.predictive_power,
                        "match_score": match_score,
                        "conditions": pattern.conditions
                    })
            
            # Sort by confidence * predictive power
            predictions.sort(
                key=lambda x: x["prediction_confidence"] * x["predictive_power"],
                reverse=True
            )
            
            return {
                "success": True,
                "predictions": predictions[:5],  # Top 5 predictions
                "total_patterns_checked": len(relevant_patterns),
                "compound_data_score": self.compound_data_multiplier
            }
            
        except Exception as e:
            logger.error(f"Error getting pattern predictions: {e}")
            return {"success": False, "error": str(e)}
    
    async def _calculate_pattern_match(self, pattern: DataPattern, current_data: Dict[str, Any]) -> float:
        """Calculate how well current data matches a pattern"""
        try:
            match_score = 0.0
            
            # Simple matching based on features
            for feature in pattern.features:
                if feature in current_data:
                    match_score += 0.2  # 20% per matching feature
            
            # Check specific conditions
            conditions = pattern.conditions
            if "direction" in conditions and "price_change" in current_data:
                price_change = current_data["price_change"]
                if conditions["direction"] == "up" and price_change > 0:
                    match_score += 0.3
                elif conditions["direction"] == "down" and price_change < 0:
                    match_score += 0.3
            
            if "correlation" in conditions and "volume" in current_data:
                # Simple correlation check
                match_score += 0.2
            
            return min(1.0, match_score)
            
        except Exception as e:
            logger.error(f"Error calculating pattern match: {e}")
            return 0.0
    
    def get_layer_metrics(self) -> Dict[str, Any]:
        """Get comprehensive data layer metrics"""
        return {
            "layer_id": self.layer_id,
            "initialized": self.initialized,
            "compound_data_multiplier": self.compound_data_multiplier,
            "pattern_accuracy_multiplier": self.pattern_accuracy_multiplier,
            "discovery_rate_multiplier": self.discovery_rate_multiplier,
            "total_patterns": self.data_metrics.total_patterns,
            "total_data_points": self.data_metrics.total_data_points,
            "avg_pattern_accuracy": self.data_metrics.avg_pattern_accuracy,
            "avg_predictive_power": self.data_metrics.avg_predictive_power,
            "data_quality_score": self.data_metrics.data_quality_score,
            "compound_data_score": self.data_metrics.compound_data_score,
            "timeframes_tracked": len(self.historical_data),
            "pattern_types": len(set(p.pattern_type for p in self.discovered_patterns.values()))
        }
    
    def get_compound_effects(self) -> Dict[str, Any]:
        """Get current compounding effects"""
        return {
            "compound_data_multiplier": self.compound_data_multiplier,
            "pattern_accuracy_multiplier": self.pattern_accuracy_multiplier,
            "discovery_rate_multiplier": self.discovery_rate_multiplier,
            "data_quality_multiplier": self.data_quality_multiplier,
            "compound_rates": self.data_compound_rates,
            "total_patterns_discovered": len(self.discovered_patterns),
            "data_efficiency": self.compound_data_multiplier * self.pattern_accuracy_multiplier
        } 