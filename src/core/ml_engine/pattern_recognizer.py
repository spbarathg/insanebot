"""
Pattern Recognition Engine for Trading Signals

This module implements pattern recognition algorithms to detect:
- Technical chart patterns (triangles, flags, head & shoulders, etc.)
- Support and resistance levels
- Volume patterns
- Momentum divergences
- Trend reversals
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import math

from .ml_types import PatternType, PatternRecognition

logger = logging.getLogger(__name__)

class PatternRecognizer:
    """Advanced pattern recognition for trading signals"""
    
    def __init__(self):
        """Initialize the pattern recognizer"""
        self.pattern_cache = {}
        self.support_resistance_levels = {}
        self.pattern_history = {}
        self.min_pattern_length = 10
        self.cache_duration = 300  # 5 minutes
        
        logger.info("PatternRecognizer initialized")
    
    async def initialize(self) -> bool:
        """Initialize the pattern recognizer"""
        try:
            logger.info("🔍 Initializing pattern recognition engine...")
            
            # Initialize pattern recognition parameters
            self.pattern_weights = {
                PatternType.BULLISH_FLAG: 0.8,
                PatternType.BEARISH_FLAG: 0.8,
                PatternType.HEAD_AND_SHOULDERS: 0.9,
                PatternType.INVERSE_HEAD_AND_SHOULDERS: 0.9,
                PatternType.DOUBLE_TOP: 0.85,
                PatternType.DOUBLE_BOTTOM: 0.85,
                PatternType.TRIANGLE_ASCENDING: 0.75,
                PatternType.TRIANGLE_DESCENDING: 0.75,
                PatternType.SUPPORT_BOUNCE: 0.7,
                PatternType.RESISTANCE_BREAK: 0.7,
                PatternType.VOLUME_SPIKE: 0.6,
                PatternType.MOMENTUM_DIVERGENCE: 0.8
            }
            
            logger.info("✅ Pattern recognition engine initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize pattern recognizer: {str(e)}")
            return False
    
    def _calculate_support_resistance(self, prices: List[float], volumes: List[float] = None) -> Tuple[List[float], List[float]]:
        """Calculate support and resistance levels"""
        try:
            if len(prices) < 20:
                return [], []
            
            # Find local minima and maxima
            window = 5
            support_levels = []
            resistance_levels = []
            
            for i in range(window, len(prices) - window):
                # Check for local minimum (support)
                is_support = all(prices[i] <= prices[j] for j in range(i - window, i + window + 1) if j != i)
                if is_support:
                    support_levels.append(prices[i])
                
                # Check for local maximum (resistance)
                is_resistance = all(prices[i] >= prices[j] for j in range(i - window, i + window + 1) if j != i)
                if is_resistance:
                    resistance_levels.append(prices[i])
            
            # Remove levels that are too close to each other
            def consolidate_levels(levels, threshold=0.02):
                if not levels:
                    return []
                
                sorted_levels = sorted(levels)
                consolidated = [sorted_levels[0]]
                
                for level in sorted_levels[1:]:
                    if abs(level - consolidated[-1]) / consolidated[-1] > threshold:
                        consolidated.append(level)
                
                return consolidated
            
            support_levels = consolidate_levels(support_levels)
            resistance_levels = consolidate_levels(resistance_levels)
            
            return support_levels, resistance_levels
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {str(e)}")
            return [], []
    
    def _detect_flag_pattern(self, prices: List[float], volumes: List[float] = None) -> List[PatternRecognition]:
        """Detect bullish and bearish flag patterns"""
        patterns = []
        
        try:
            if len(prices) < 30:
                return patterns
            
            for i in range(20, len(prices) - 10):
                # Look for a sharp move followed by consolidation
                lookback = 15
                consolidation_period = 10
                
                if i < lookback + consolidation_period:
                    continue
                
                # Check for initial sharp move
                initial_prices = prices[i - lookback - consolidation_period:i - consolidation_period]
                consolidation_prices = prices[i - consolidation_period:i]
                current_price = prices[i]
                
                if len(initial_prices) < 5 or len(consolidation_prices) < 5:
                    continue
                
                # Calculate move strength
                initial_start = initial_prices[0]
                initial_end = initial_prices[-1]
                move_strength = abs(initial_end - initial_start) / initial_start
                
                if move_strength < 0.05:  # Minimum 5% move
                    continue
                
                # Check consolidation (sideways movement)
                consolidation_range = (max(consolidation_prices) - min(consolidation_prices)) / min(consolidation_prices)
                
                if consolidation_range > 0.03:  # Too much volatility in consolidation
                    continue
                
                # Determine pattern type
                if initial_end > initial_start:
                    pattern_type = PatternType.BULLISH_FLAG
                    expected_direction = "up"
                    expected_move = move_strength * 0.618  # Fibonacci retracement
                else:
                    pattern_type = PatternType.BEARISH_FLAG
                    expected_direction = "down"
                    expected_move = -move_strength * 0.618
                
                # Calculate confidence based on volume confirmation
                volume_confirmation = False
                if volumes and len(volumes) >= i:
                    initial_volumes = volumes[i - lookback - consolidation_period:i - consolidation_period]
                    consolidation_volumes = volumes[i - consolidation_period:i]
                    
                    if initial_volumes and consolidation_volumes:
                        avg_initial_volume = sum(initial_volumes) / len(initial_volumes)
                        avg_consolidation_volume = sum(consolidation_volumes) / len(consolidation_volumes)
                        
                        # Volume should decrease during consolidation
                        if avg_consolidation_volume < avg_initial_volume * 0.8:
                            volume_confirmation = True
                
                confidence = 0.6 + (0.2 if volume_confirmation else 0) + min(move_strength * 2, 0.2)
                
                pattern = PatternRecognition(
                    pattern_type=pattern_type,
                    confidence=confidence,
                    strength=move_strength,
                    timeframe="1h",
                    start_time=time.time() - (len(prices) - i + lookback + consolidation_period) * 3600,
                    end_time=time.time() - (len(prices) - i) * 3600,
                    key_levels=[initial_start, initial_end, min(consolidation_prices), max(consolidation_prices)],
                    expected_move=expected_move,
                    expected_direction=expected_direction,
                    reliability_score=confidence * self.pattern_weights.get(pattern_type, 0.5),
                    volume_confirmation=volume_confirmation
                )
                
                patterns.append(pattern)
                
        except Exception as e:
            logger.error(f"Error detecting flag patterns: {str(e)}")
        
        return patterns
    
    def _detect_head_and_shoulders(self, prices: List[float], volumes: List[float] = None) -> List[PatternRecognition]:
        """Detect head and shoulders patterns"""
        patterns = []
        
        try:
            if len(prices) < 50:
                return patterns
            
            # Look for three peaks pattern
            for i in range(25, len(prices) - 25):
                window = 12
                
                # Find potential peaks
                peaks = []
                peak_indices = []
                
                for j in range(i - 20, i + 21, 5):
                    if j < window or j >= len(prices) - window:
                        continue
                    
                    # Check if it's a local maximum
                    is_peak = all(prices[j] >= prices[k] for k in range(j - window, j + window + 1))
                    if is_peak:
                        peaks.append(prices[j])
                        peak_indices.append(j)
                
                if len(peaks) < 3:
                    continue
                
                # Sort peaks by index
                sorted_peaks = sorted(zip(peak_indices, peaks))
                
                if len(sorted_peaks) >= 3:
                    # Check for head and shoulders pattern
                    left_shoulder_idx, left_shoulder = sorted_peaks[0]
                    head_idx, head = sorted_peaks[1]
                    right_shoulder_idx, right_shoulder = sorted_peaks[2]
                    
                    # Head should be higher than both shoulders
                    if head > left_shoulder and head > right_shoulder:
                        # Shoulders should be roughly equal
                        shoulder_diff = abs(left_shoulder - right_shoulder) / max(left_shoulder, right_shoulder)
                        
                        if shoulder_diff < 0.05:  # Within 5%
                            # Find neckline (valleys between peaks)
                            neckline_points = []
                            
                            # Valley between left shoulder and head
                            valley1_idx = left_shoulder_idx + np.argmin(prices[left_shoulder_idx:head_idx])
                            valley1_price = prices[valley1_idx]
                            
                            # Valley between head and right shoulder
                            valley2_idx = head_idx + np.argmin(prices[head_idx:right_shoulder_idx])
                            valley2_price = prices[valley2_idx]
                            
                            neckline_level = (valley1_price + valley2_price) / 2
                            
                            # Calculate pattern metrics
                            head_height = head - neckline_level
                            expected_move = -head_height  # Bearish pattern
                            
                            # Check volume confirmation
                            volume_confirmation = False
                            if volumes and len(volumes) > right_shoulder_idx:
                                # Volume should be lower on right shoulder than left
                                left_vol = sum(volumes[max(0, left_shoulder_idx-5):left_shoulder_idx+5]) / 10
                                right_vol = sum(volumes[max(0, right_shoulder_idx-5):right_shoulder_idx+5]) / 10
                                
                                if right_vol < left_vol * 0.8:
                                    volume_confirmation = True
                            
                            confidence = 0.75 + (0.15 if volume_confirmation else 0)
                            
                            pattern = PatternRecognition(
                                pattern_type=PatternType.HEAD_AND_SHOULDERS,
                                confidence=confidence,
                                strength=head_height / head,
                                timeframe="1h",
                                start_time=time.time() - (len(prices) - left_shoulder_idx) * 3600,
                                end_time=time.time() - (len(prices) - right_shoulder_idx) * 3600,
                                key_levels=[left_shoulder, head, right_shoulder, neckline_level],
                                expected_move=expected_move / head,
                                expected_direction="down",
                                reliability_score=confidence * self.pattern_weights.get(PatternType.HEAD_AND_SHOULDERS, 0.9),
                                volume_confirmation=volume_confirmation
                            )
                            
                            patterns.append(pattern)
                
                # Check for inverse head and shoulders (same logic, inverted)
                valleys = []
                valley_indices = []
                
                for j in range(i - 20, i + 21, 5):
                    if j < window or j >= len(prices) - window:
                        continue
                    
                    # Check if it's a local minimum
                    is_valley = all(prices[j] <= prices[k] for k in range(j - window, j + window + 1))
                    if is_valley:
                        valleys.append(prices[j])
                        valley_indices.append(j)
                
                if len(valleys) >= 3:
                    sorted_valleys = sorted(zip(valley_indices, valleys))
                    
                    left_shoulder_idx, left_shoulder = sorted_valleys[0]
                    head_idx, head = sorted_valleys[1]
                    right_shoulder_idx, right_shoulder = sorted_valleys[2]
                    
                    # Head should be lower than both shoulders
                    if head < left_shoulder and head < right_shoulder:
                        shoulder_diff = abs(left_shoulder - right_shoulder) / max(left_shoulder, right_shoulder)
                        
                        if shoulder_diff < 0.05:
                            # Find neckline (peaks between valleys)
                            peak1_idx = left_shoulder_idx + np.argmax(prices[left_shoulder_idx:head_idx])
                            peak1_price = prices[peak1_idx]
                            
                            peak2_idx = head_idx + np.argmax(prices[head_idx:right_shoulder_idx])
                            peak2_price = prices[peak2_idx]
                            
                            neckline_level = (peak1_price + peak2_price) / 2
                            
                            head_depth = neckline_level - head
                            expected_move = head_depth  # Bullish pattern
                            
                            volume_confirmation = False
                            if volumes and len(volumes) > right_shoulder_idx:
                                left_vol = sum(volumes[max(0, left_shoulder_idx-5):left_shoulder_idx+5]) / 10
                                right_vol = sum(volumes[max(0, right_shoulder_idx-5):right_shoulder_idx+5]) / 10
                                
                                if right_vol > left_vol * 1.2:  # Increasing volume for bullish pattern
                                    volume_confirmation = True
                            
                            confidence = 0.75 + (0.15 if volume_confirmation else 0)
                            
                            pattern = PatternRecognition(
                                pattern_type=PatternType.INVERSE_HEAD_AND_SHOULDERS,
                                confidence=confidence,
                                strength=head_depth / head,
                                timeframe="1h",
                                start_time=time.time() - (len(prices) - left_shoulder_idx) * 3600,
                                end_time=time.time() - (len(prices) - right_shoulder_idx) * 3600,
                                key_levels=[left_shoulder, head, right_shoulder, neckline_level],
                                expected_move=expected_move / head,
                                expected_direction="up",
                                reliability_score=confidence * self.pattern_weights.get(PatternType.INVERSE_HEAD_AND_SHOULDERS, 0.9),
                                volume_confirmation=volume_confirmation
                            )
                            
                            patterns.append(pattern)
                
        except Exception as e:
            logger.error(f"Error detecting head and shoulders: {str(e)}")
        
        return patterns
    
    def _detect_double_top_bottom(self, prices: List[float], volumes: List[float] = None) -> List[PatternRecognition]:
        """Detect double top and double bottom patterns"""
        patterns = []
        
        try:
            if len(prices) < 30:
                return patterns
            
            window = 8
            
            # Find peaks and valleys
            peaks = []
            valleys = []
            
            for i in range(window, len(prices) - window):
                # Check for peaks
                is_peak = all(prices[i] >= prices[j] for j in range(i - window, i + window + 1))
                if is_peak:
                    peaks.append((i, prices[i]))
                
                # Check for valleys
                is_valley = all(prices[i] <= prices[j] for j in range(i - window, i + window + 1))
                if is_valley:
                    valleys.append((i, prices[i]))
            
            # Look for double tops
            for i in range(len(peaks) - 1):
                for j in range(i + 1, len(peaks)):
                    peak1_idx, peak1_price = peaks[i]
                    peak2_idx, peak2_price = peaks[j]
                    
                    # Peaks should be similar height
                    price_diff = abs(peak1_price - peak2_price) / max(peak1_price, peak2_price)
                    
                    if price_diff < 0.03:  # Within 3%
                        # Find valley between peaks
                        between_valleys = [v for v in valleys if peak1_idx < v[0] < peak2_idx]
                        
                        if between_valleys:
                            valley_idx, valley_price = min(between_valleys, key=lambda x: x[1])
                            
                            # Calculate pattern strength
                            pattern_height = max(peak1_price, peak2_price) - valley_price
                            pattern_strength = pattern_height / max(peak1_price, peak2_price)
                            
                            if pattern_strength > 0.05:  # Minimum 5% pattern
                                expected_move = -pattern_height
                                
                                # Volume confirmation
                                volume_confirmation = False
                                if volumes and len(volumes) > peak2_idx:
                                    peak1_vol = sum(volumes[max(0, peak1_idx-3):peak1_idx+3]) / 6
                                    peak2_vol = sum(volumes[max(0, peak2_idx-3):peak2_idx+3]) / 6
                                    
                                    if peak2_vol < peak1_vol * 0.8:  # Decreasing volume
                                        volume_confirmation = True
                                
                                confidence = 0.7 + (0.15 if volume_confirmation else 0)
                                
                                pattern = PatternRecognition(
                                    pattern_type=PatternType.DOUBLE_TOP,
                                    confidence=confidence,
                                    strength=pattern_strength,
                                    timeframe="1h",
                                    start_time=time.time() - (len(prices) - peak1_idx) * 3600,
                                    end_time=time.time() - (len(prices) - peak2_idx) * 3600,
                                    key_levels=[peak1_price, valley_price, peak2_price],
                                    expected_move=expected_move / max(peak1_price, peak2_price),
                                    expected_direction="down",
                                    reliability_score=confidence * self.pattern_weights.get(PatternType.DOUBLE_TOP, 0.85),
                                    volume_confirmation=volume_confirmation
                                )
                                
                                patterns.append(pattern)
            
            # Look for double bottoms
            for i in range(len(valleys) - 1):
                for j in range(i + 1, len(valleys)):
                    valley1_idx, valley1_price = valleys[i]
                    valley2_idx, valley2_price = valleys[j]
                    
                    # Valleys should be similar depth
                    price_diff = abs(valley1_price - valley2_price) / max(valley1_price, valley2_price)
                    
                    if price_diff < 0.03:
                        # Find peak between valleys
                        between_peaks = [p for p in peaks if valley1_idx < p[0] < valley2_idx]
                        
                        if between_peaks:
                            peak_idx, peak_price = max(between_peaks, key=lambda x: x[1])
                            
                            pattern_height = peak_price - min(valley1_price, valley2_price)
                            pattern_strength = pattern_height / min(valley1_price, valley2_price)
                            
                            if pattern_strength > 0.05:
                                expected_move = pattern_height
                                
                                # Volume confirmation
                                volume_confirmation = False
                                if volumes and len(volumes) > valley2_idx:
                                    valley1_vol = sum(volumes[max(0, valley1_idx-3):valley1_idx+3]) / 6
                                    valley2_vol = sum(volumes[max(0, valley2_idx-3):valley2_idx+3]) / 6
                                    
                                    if valley2_vol > valley1_vol * 1.2:  # Increasing volume
                                        volume_confirmation = True
                                
                                confidence = 0.7 + (0.15 if volume_confirmation else 0)
                                
                                pattern = PatternRecognition(
                                    pattern_type=PatternType.DOUBLE_BOTTOM,
                                    confidence=confidence,
                                    strength=pattern_strength,
                                    timeframe="1h",
                                    start_time=time.time() - (len(prices) - valley1_idx) * 3600,
                                    end_time=time.time() - (len(prices) - valley2_idx) * 3600,
                                    key_levels=[valley1_price, peak_price, valley2_price],
                                    expected_move=expected_move / min(valley1_price, valley2_price),
                                    expected_direction="up",
                                    reliability_score=confidence * self.pattern_weights.get(PatternType.DOUBLE_BOTTOM, 0.85),
                                    volume_confirmation=volume_confirmation
                                )
                                
                                patterns.append(pattern)
                
        except Exception as e:
            logger.error(f"Error detecting double top/bottom: {str(e)}")
        
        return patterns
    
    def _detect_volume_patterns(self, prices: List[float], volumes: List[float]) -> List[PatternRecognition]:
        """Detect volume-based patterns"""
        patterns = []
        
        try:
            if not volumes or len(volumes) < 20:
                return patterns
            
            # Calculate volume moving average
            volume_sma = []
            window = 10
            
            for i in range(len(volumes)):
                if i < window - 1:
                    volume_sma.append(volumes[i])
                else:
                    volume_sma.append(sum(volumes[i - window + 1:i + 1]) / window)
            
            # Detect volume spikes
            for i in range(len(volumes) - 1):
                if volume_sma[i] > 0:
                    volume_ratio = volumes[i] / volume_sma[i]
                    
                    if volume_ratio > 2.0:  # Volume spike (2x average)
                        # Determine price action during spike
                        if i > 0:
                            price_change = (prices[i] - prices[i - 1]) / prices[i - 1]
                            
                            if abs(price_change) > 0.02:  # Significant price movement
                                expected_direction = "up" if price_change > 0 else "down"
                                
                                pattern = PatternRecognition(
                                    pattern_type=PatternType.VOLUME_SPIKE,
                                    confidence=min(0.9, 0.5 + volume_ratio * 0.1),
                                    strength=volume_ratio,
                                    timeframe="1h",
                                    start_time=time.time() - (len(prices) - i) * 3600,
                                    end_time=time.time() - (len(prices) - i) * 3600 + 3600,
                                    key_levels=[prices[i], volume_sma[i], volumes[i]],
                                    expected_move=price_change * 0.5,  # Expect continuation
                                    expected_direction=expected_direction,
                                    reliability_score=min(0.9, 0.5 + volume_ratio * 0.1) * self.pattern_weights.get(PatternType.VOLUME_SPIKE, 0.6),
                                    volume_confirmation=True
                                )
                                
                                patterns.append(pattern)
                
        except Exception as e:
            logger.error(f"Error detecting volume patterns: {str(e)}")
        
        return patterns
    
    async def recognize_patterns(self, token_address: str, token_data: Dict, price_history: List[float], volume_history: List[float] = None) -> List[PatternRecognition]:
        """Recognize all patterns in the given price data"""
        try:
            current_time = time.time()
            
            # Check cache
            cache_key = f"{token_address}_{int(current_time // self.cache_duration)}"
            if cache_key in self.pattern_cache:
                return self.pattern_cache[cache_key]
            
            if len(price_history) < self.min_pattern_length:
                return []
            
            all_patterns = []
            
            # Detect different pattern types
            logger.debug(f"🔍 Recognizing patterns for {token_data.get('symbol', 'UNKNOWN')}")
            
            # Flag patterns
            flag_patterns = self._detect_flag_pattern(price_history, volume_history)
            all_patterns.extend(flag_patterns)
            
            # Head and shoulders patterns
            hs_patterns = self._detect_head_and_shoulders(price_history, volume_history)
            all_patterns.extend(hs_patterns)
            
            # Double top/bottom patterns
            double_patterns = self._detect_double_top_bottom(price_history, volume_history)
            all_patterns.extend(double_patterns)
            
            # Volume patterns
            if volume_history:
                volume_patterns = self._detect_volume_patterns(price_history, volume_history)
                all_patterns.extend(volume_patterns)
            
            # Support/resistance patterns
            support_levels, resistance_levels = self._calculate_support_resistance(price_history, volume_history)
            current_price = price_history[-1]
            
            # Check for support bounces
            for support in support_levels:
                if abs(current_price - support) / support < 0.02:  # Within 2% of support
                    pattern = PatternRecognition(
                        pattern_type=PatternType.SUPPORT_BOUNCE,
                        confidence=0.6,
                        strength=0.5,
                        timeframe="1h",
                        start_time=current_time - 3600,
                        end_time=current_time,
                        key_levels=[support, current_price],
                        expected_move=0.03,  # 3% bounce expected
                        expected_direction="up",
                        reliability_score=0.6 * self.pattern_weights.get(PatternType.SUPPORT_BOUNCE, 0.7),
                        volume_confirmation=False
                    )
                    all_patterns.append(pattern)
            
            # Check for resistance breaks
            for resistance in resistance_levels:
                if current_price > resistance and (current_price - resistance) / resistance < 0.02:
                    pattern = PatternRecognition(
                        pattern_type=PatternType.RESISTANCE_BREAK,
                        confidence=0.7,
                        strength=0.6,
                        timeframe="1h",
                        start_time=current_time - 3600,
                        end_time=current_time,
                        key_levels=[resistance, current_price],
                        expected_move=0.05,  # 5% continuation expected
                        expected_direction="up",
                        reliability_score=0.7 * self.pattern_weights.get(PatternType.RESISTANCE_BREAK, 0.7),
                        volume_confirmation=False
                    )
                    all_patterns.append(pattern)
            
            # Store support/resistance levels
            self.support_resistance_levels[token_address] = {
                'support': support_levels,
                'resistance': resistance_levels,
                'timestamp': current_time
            }
            
            # Sort patterns by reliability score
            all_patterns.sort(key=lambda x: x.reliability_score, reverse=True)
            
            # Cache results
            self.pattern_cache[cache_key] = all_patterns
            
            # Cleanup old cache entries
            current_bucket = int(current_time // self.cache_duration)
            self.pattern_cache = {k: v for k, v in self.pattern_cache.items() 
                                if int(k.split('_')[-1]) >= current_bucket - 10}
            
            logger.debug(f"✅ Found {len(all_patterns)} patterns for {token_data.get('symbol', 'UNKNOWN')}")
            
            return all_patterns
            
        except Exception as e:
            logger.error(f"Error recognizing patterns for {token_address}: {str(e)}")
            return []
    
    def get_support_resistance_levels(self, token_address: str) -> Dict[str, List[float]]:
        """Get current support and resistance levels for a token"""
        return self.support_resistance_levels.get(token_address, {'support': [], 'resistance': []})
    
    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about pattern recognition"""
        total_patterns = sum(len(patterns) for patterns in self.pattern_cache.values())
        
        pattern_counts = {}
        for patterns in self.pattern_cache.values():
            for pattern in patterns:
                pattern_type = pattern.pattern_type.value
                pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
        
        return {
            'total_patterns_detected': total_patterns,
            'pattern_counts': pattern_counts,
            'cache_size': len(self.pattern_cache),
            'tokens_analyzed': len(self.support_resistance_levels),
            'pattern_types_available': [pt.value for pt in PatternType]
        } 