"""
Data types and models for the ML Engine
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
import time

class PatternType(Enum):
    """Types of trading patterns that can be recognized"""
    BULLISH_FLAG = "bullish_flag"
    BEARISH_FLAG = "bearish_flag"
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIANGLE_ASCENDING = "triangle_ascending"
    TRIANGLE_DESCENDING = "triangle_descending"
    WEDGE_RISING = "wedge_rising"
    WEDGE_FALLING = "wedge_falling"
    SUPPORT_BOUNCE = "support_bounce"
    RESISTANCE_BREAK = "resistance_break"
    VOLUME_SPIKE = "volume_spike"
    MOMENTUM_DIVERGENCE = "momentum_divergence"

class MarketCondition(Enum):
    """Overall market condition classifications"""
    STRONG_BULL = "strong_bull"
    MODERATE_BULL = "moderate_bull"
    SIDEWAYS = "sideways"
    MODERATE_BEAR = "moderate_bear"
    STRONG_BEAR = "strong_bear"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

class SentimentType(Enum):
    """Market sentiment classifications"""
    EXTREMELY_BULLISH = "extremely_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    EXTREMELY_BEARISH = "extremely_bearish"

@dataclass
class PredictionResult:
    """Result from price prediction model"""
    token_address: str
    token_symbol: str
    current_price: float
    predicted_price_1h: float
    predicted_price_4h: float
    predicted_price_24h: float
    confidence_1h: float
    confidence_4h: float
    confidence_24h: float
    prediction_timestamp: float
    model_version: str
    feature_importance: Dict[str, float]
    
    @property
    def expected_return_1h(self) -> float:
        """Calculate expected return for 1 hour"""
        return (self.predicted_price_1h / self.current_price) - 1
    
    @property
    def expected_return_4h(self) -> float:
        """Calculate expected return for 4 hours"""
        return (self.predicted_price_4h / self.current_price) - 1
    
    @property
    def expected_return_24h(self) -> float:
        """Calculate expected return for 24 hours"""
        return (self.predicted_price_24h / self.current_price) - 1
    
    @property
    def weighted_confidence(self) -> float:
        """Calculate weighted average confidence"""
        return (self.confidence_1h * 0.5 + self.confidence_4h * 0.3 + self.confidence_24h * 0.2)

@dataclass
class PatternRecognition:
    """Result from pattern recognition"""
    pattern_type: PatternType
    confidence: float
    strength: float
    timeframe: str
    start_time: float
    end_time: float
    key_levels: List[float]
    expected_move: float
    expected_direction: str  # "up", "down", "sideways"
    reliability_score: float
    volume_confirmation: bool

@dataclass
class SentimentResult:
    """Result from sentiment analysis"""
    token_address: str
    token_symbol: str
    overall_sentiment: SentimentType
    sentiment_score: float  # -1 to 1
    social_mentions: int
    social_sentiment: float
    whale_activity: float
    developer_activity: float
    holder_sentiment: float
    fear_greed_index: float
    analysis_timestamp: float
    confidence: float
    
    @property
    def is_bullish(self) -> bool:
        """Check if sentiment is bullish"""
        return self.sentiment_score > 0.2
    
    @property
    def is_bearish(self) -> bool:
        """Check if sentiment is bearish"""
        return self.sentiment_score < -0.2

@dataclass
class RiskScore:
    """Comprehensive risk scoring result"""
    token_address: str
    token_symbol: str
    overall_risk_score: float  # 0-1, higher = riskier
    volatility_risk: float
    liquidity_risk: float
    market_cap_risk: float
    holder_concentration_risk: float
    smart_money_risk: float
    technical_risk: float
    fundamental_risk: float
    sentiment_risk: float
    risk_category: str  # "low", "medium", "high", "extreme"
    risk_factors: List[str]
    recommended_position_size: float
    max_position_size: float
    stop_loss_level: float
    analysis_timestamp: float
    
    @property
    def is_safe_to_trade(self) -> bool:
        """Check if token is safe to trade"""
        return self.overall_risk_score < 0.7
    
    @property
    def risk_level(self) -> str:
        """Get risk level description"""
        if self.overall_risk_score < 0.25:
            return "low"
        elif self.overall_risk_score < 0.5:
            return "medium"
        elif self.overall_risk_score < 0.75:
            return "high"
        else:
            return "extreme"

@dataclass
class MLSignal:
    """Combined ML signal for trading decisions"""
    token_address: str
    token_symbol: str
    signal_type: str  # "buy", "sell", "hold"
    signal_strength: float  # 0-1
    confidence: float  # 0-1
    predicted_return: float
    risk_score: float
    sentiment_score: float
    pattern_signals: List[PatternRecognition]
    price_prediction: Optional[PredictionResult]
    timeframe: str
    entry_price: float
    target_price: float
    stop_loss_price: float
    position_size_recommendation: float
    reasoning: List[str]
    signal_timestamp: float
    expires_at: float
    
    @property
    def is_expired(self) -> bool:
        """Check if signal has expired"""
        return time.time() > self.expires_at
    
    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio"""
        if self.signal_type == "buy":
            potential_gain = self.target_price - self.entry_price
            potential_loss = self.entry_price - self.stop_loss_price
        else:
            potential_gain = self.entry_price - self.target_price
            potential_loss = self.target_price - self.entry_price
        
        if potential_loss <= 0:
            return float('inf')
        return potential_gain / potential_loss
    
    @property
    def quality_score(self) -> float:
        """Calculate overall signal quality"""
        return (
            self.confidence * 0.4 +
            self.signal_strength * 0.3 +
            (1 - self.risk_score) * 0.2 +
            min(self.risk_reward_ratio / 2, 1) * 0.1
        )

@dataclass
class MarketRegime:
    """Market regime classification"""
    condition: MarketCondition
    volatility_level: float
    trend_strength: float
    momentum: float
    volume_profile: str
    dominant_patterns: List[PatternType]
    regime_start_time: float
    confidence: float
    expected_duration: float
    recommended_strategies: List[str] 