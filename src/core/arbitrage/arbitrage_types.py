"""
Data types and models for arbitrage operations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import time

class ArbitrageStatus(Enum):
    """Status of an arbitrage opportunity"""
    DETECTED = "detected"
    CALCULATING = "calculating"
    PROFITABLE = "profitable" 
    UNPROFITABLE = "unprofitable"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"

class DEXName(Enum):
    """Supported DEX names"""
    JUPITER = "jupiter"
    RAYDIUM = "raydium"
    ORCA = "orca"
    SERUM = "serum"

@dataclass
class DEXInfo:
    """Information about a DEX"""
    name: DEXName
    api_url: str
    fee_percentage: float
    min_liquidity: float
    supported_tokens: List[str]
    
    def __post_init__(self):
        self.last_updated = time.time()

@dataclass
class PriceQuote:
    """Price quote from a DEX"""
    dex: DEXName
    token_address: str
    input_token: str
    output_token: str
    input_amount: float
    output_amount: float
    price: float
    price_impact: float
    liquidity: float
    fee: float
    slippage: float
    timestamp: float
    raw_data: Optional[Dict[str, Any]] = None
    
    @property
    def effective_price(self) -> float:
        """Price after fees and slippage"""
        return self.output_amount / self.input_amount if self.input_amount > 0 else 0
    
    @property
    def age_seconds(self) -> float:
        """Age of quote in seconds"""
        return time.time() - self.timestamp
    
    def is_expired(self, max_age_seconds: int = 30) -> bool:
        """Check if quote is expired"""
        return self.age_seconds > max_age_seconds

@dataclass
class ArbitrageOpportunity:
    """Represents an arbitrage opportunity between two DEXs"""
    id: str
    token_address: str
    token_symbol: str
    buy_dex: DEXName
    sell_dex: DEXName
    buy_quote: PriceQuote
    sell_quote: PriceQuote
    amount: float
    potential_profit_sol: float
    potential_profit_usd: float
    profit_percentage: float
    total_fees: float
    net_profit: float
    confidence_score: float
    risk_level: str
    status: ArbitrageStatus
    detected_at: float
    expires_at: float
    execution_window_seconds: int = 30
    
    def __post_init__(self):
        if not hasattr(self, 'detected_at'):
            self.detected_at = time.time()
        if not hasattr(self, 'expires_at'):
            self.expires_at = self.detected_at + self.execution_window_seconds
    
    @property
    def is_expired(self) -> bool:
        """Check if opportunity is expired"""
        return time.time() > self.expires_at
    
    @property
    def time_remaining(self) -> float:
        """Time remaining to execute in seconds"""
        return max(0, self.expires_at - time.time())
    
    @property
    def roi_percentage(self) -> float:
        """Return on investment percentage"""
        return (self.net_profit / self.amount) * 100 if self.amount > 0 else 0

@dataclass
class ArbitrageResult:
    """Result of an arbitrage execution"""
    opportunity_id: str
    status: ArbitrageStatus
    executed: bool
    buy_transaction_id: Optional[str] = None
    sell_transaction_id: Optional[str] = None
    actual_profit_sol: float = 0.0
    actual_profit_usd: float = 0.0
    execution_time_seconds: float = 0.0
    gas_fees_paid: float = 0.0
    slippage_experienced: float = 0.0
    error_message: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    def __post_init__(self):
        if self.started_at is None:
            self.started_at = time.time()
    
    @property
    def success_rate(self) -> float:
        """Success rate as percentage"""
        return 100.0 if self.executed and self.status == ArbitrageStatus.COMPLETED else 0.0
    
    @property
    def actual_roi(self) -> float:
        """Actual ROI achieved"""
        if hasattr(self, '_initial_investment') and self._initial_investment > 0:
            return (self.actual_profit_sol / self._initial_investment) * 100
        return 0.0

@dataclass 
class MarketConditions:
    """Current market conditions affecting arbitrage"""
    network_congestion: float  # 0-1 scale
    gas_price_gwei: float
    sol_price_usd: float
    overall_volatility: float
    dex_spreads: Dict[str, float]
    liquidity_conditions: Dict[str, str]  # "high", "medium", "low"
    timestamp: float
    
    def __post_init__(self):
        if not hasattr(self, 'timestamp'):
            self.timestamp = time.time()
    
    @property
    def is_favorable_for_arbitrage(self) -> bool:
        """Check if conditions are favorable for arbitrage"""
        return (
            self.network_congestion < 0.7 and
            self.overall_volatility > 0.02 and  # Some volatility needed for price differences
            any(spread > 0.5 for spread in self.dex_spreads.values())  # At least 0.5% spread somewhere
        ) 