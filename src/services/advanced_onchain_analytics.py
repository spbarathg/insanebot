"""
Advanced On-Chain Analytics Engine

Deep blockchain analysis system for whale tracking, insider detection,
and advanced pattern recognition for memecoin trading.

Features:
- Large holder transaction analysis
- Insider trading detection
- Liquidity pool monitoring
- Token distribution analysis
- Whale movement tracking
- Dev wallet monitoring
- Smart contract analysis
"""

import asyncio
import aiohttp
import time
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class WhaleTransaction:
    """Large transaction detected on-chain"""
    transaction_signature: str
    wallet_address: str
    token_address: str
    token_symbol: str
    transaction_type: str  # 'buy', 'sell', 'transfer'
    amount: float
    value_usd: float
    price_impact: float
    timestamp: float
    block_height: int
    wallet_balance_before: float
    wallet_balance_after: float
    is_first_interaction: bool
    confidence_score: float

@dataclass
class WalletProfile:
    """Whale wallet profile and behavior analysis"""
    wallet_address: str
    total_balance: float
    total_value_usd: float
    token_count: int
    first_seen: float
    last_activity: float
    transaction_count: int
    avg_transaction_size: float
    success_rate: float
    win_rate: float
    total_pnl: float
    behavior_tags: List[str]
    risk_score: float  # 0-1, higher is riskier
    influence_score: float  # 0-1, higher is more influential

@dataclass
class InsiderAlert:
    """Insider trading alert"""
    token_address: str
    token_symbol: str
    alert_type: str  # 'dev_dump', 'pre_listing_activity', 'coordinated_buying'
    description: str
    involved_wallets: List[str]
    total_volume: float
    time_window: float
    confidence: float
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    timestamp: float

@dataclass
class TokenAnalytics:
    """Comprehensive token analytics"""
    token_address: str
    token_symbol: str
    total_supply: float
    circulating_supply: float
    holder_count: int
    top_holders: List[Dict[str, Any]]
    concentration_score: float  # 0-1, higher is more concentrated
    liquidity_score: float
    developer_allocation: float
    insider_activity_score: float
    whale_activity_score: float
    distribution_health: str  # 'poor', 'fair', 'good', 'excellent'
    risk_factors: List[str]
    bullish_indicators: List[str]

class OnChainAnalyzer:
    """
    Advanced on-chain analytics for memecoin trading
    
    Provides deep blockchain analysis including whale tracking,
    insider detection, and comprehensive token analytics.
    """
    
    def __init__(self, helius_service=None):
        self.helius_service = helius_service
        
        # Whale tracking configuration
        self.whale_config = {
            'min_transaction_usd': 10000,    # $10K minimum for whale transactions
            'min_wallet_balance_usd': 50000, # $50K minimum for whale classification
            'price_impact_threshold': 0.05,  # 5% price impact threshold
            'volume_spike_multiplier': 3,    # 3x normal volume = spike
            'insider_time_window': 3600,     # 1 hour window for insider analysis
        }
        
        # Known wallet categories
        self.known_wallets = {
            'exchanges': set(),
            'market_makers': set(),
            'known_whales': set(),
            'dev_wallets': set(),
            'suspicious_wallets': set()
        }
        
        # Data storage
        self.whale_transactions: deque = deque(maxlen=10000)
        self.wallet_profiles: Dict[str, WalletProfile] = {}
        self.token_analytics: Dict[str, TokenAnalytics] = {}
        self.insider_alerts: Dict[str, List[InsiderAlert]] = defaultdict(list)
        
        # Performance tracking
        self.analytics_stats = {
            'transactions_analyzed': 0,
            'whales_tracked': 0,
            'insider_alerts_generated': 0,
            'tokens_analyzed': 0,
            'prediction_accuracy': 0.0
        }
        
        logger.info("🔍 Advanced On-Chain Analytics initialized - Ready for whale hunting!")
    
    async def analyze_token(self, token_address: str, token_symbol: str = "") -> TokenAnalytics:
        """Perform comprehensive on-chain analysis of a token"""
        try:
            logger.info(f"🔍 Analyzing token: {token_symbol or token_address[:8]}...")
            
            # Get basic token information
            token_info = await self._get_token_info(token_address)
            
            # Analyze holder distribution
            holders = await self._analyze_holder_distribution(token_address)
            
            # Calculate concentration metrics
            concentration_score = self._calculate_concentration_score(holders)
            
            # Analyze whale activity
            whale_activity = await self._analyze_whale_activity(token_address)
            
            # Check for insider activity
            insider_activity = await self._detect_insider_activity(token_address)
            
            # Calculate liquidity metrics
            liquidity_score = await self._calculate_liquidity_score(token_address)
            
            # Generate risk assessment
            risk_factors = await self._assess_risk_factors(token_address, holders, whale_activity)
            bullish_indicators = await self._identify_bullish_indicators(token_address, holders)
            
            # Determine distribution health
            distribution_health = self._assess_distribution_health(
                concentration_score, len(holders), whale_activity
            )
            
            analytics = TokenAnalytics(
                token_address=token_address,
                token_symbol=token_symbol,
                total_supply=token_info.get('total_supply', 0),
                circulating_supply=token_info.get('circulating_supply', 0),
                holder_count=len(holders),
                top_holders=holders[:20],  # Top 20 holders
                concentration_score=concentration_score,
                liquidity_score=liquidity_score,
                developer_allocation=token_info.get('dev_allocation', 0),
                insider_activity_score=insider_activity,
                whale_activity_score=whale_activity,
                distribution_health=distribution_health,
                risk_factors=risk_factors,
                bullish_indicators=bullish_indicators
            )
            
            self.token_analytics[token_address] = analytics
            self.analytics_stats['tokens_analyzed'] += 1
            
            logger.info(f"✅ Token analysis complete: {token_symbol} | "
                       f"Health: {distribution_health} | "
                       f"Concentration: {concentration_score:.2f} | "
                       f"Risks: {len(risk_factors)}")
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error analyzing token {token_address}: {e}")
            return TokenAnalytics(
                token_address=token_address,
                token_symbol=token_symbol,
                total_supply=0, circulating_supply=0, holder_count=0,
                top_holders=[], concentration_score=1.0, liquidity_score=0.0,
                developer_allocation=0, insider_activity_score=0,
                whale_activity_score=0, distribution_health="unknown",
                risk_factors=["analysis_failed"], bullish_indicators=[]
            )
    
    async def track_whale_movements(self, token_addresses: List[str]) -> List[WhaleTransaction]:
        """Track whale movements for specified tokens"""
        try:
            whale_transactions = []
            
            for token_address in token_addresses:
                try:
                    # Get recent large transactions
                    transactions = await self._get_large_transactions(token_address)
                    
                    for tx in transactions:
                        whale_tx = await self._analyze_whale_transaction(tx)
                        if whale_tx:
                            whale_transactions.append(whale_tx)
                            self.whale_transactions.append(whale_tx)
                            
                            # Update wallet profile
                            await self._update_wallet_profile(whale_tx.wallet_address, whale_tx)
                            
                except Exception as e:
                    logger.error(f"Error tracking whales for {token_address}: {e}")
            
            self.analytics_stats['transactions_analyzed'] += len(whale_transactions)
            
            return whale_transactions
            
        except Exception as e:
            logger.error(f"Error in whale tracking: {e}")
            return []
    
    async def detect_insider_activity(self, token_address: str) -> List[InsiderAlert]:
        """Detect potential insider trading activity"""
        try:
            alerts = []
            
            # Get recent transactions
            recent_txs = await self._get_recent_transactions(token_address, hours=24)
            
            # Check for suspicious patterns
            
            # 1. Pre-listing coordinated buying
            coordinated_alert = await self._detect_coordinated_buying(token_address, recent_txs)
            if coordinated_alert:
                alerts.append(coordinated_alert)
            
            # 2. Developer wallet activity
            dev_alert = await self._detect_dev_activity(token_address, recent_txs)
            if dev_alert:
                alerts.append(dev_alert)
            
            # 3. Unusual whale accumulation
            whale_alert = await self._detect_whale_accumulation(token_address, recent_txs)
            if whale_alert:
                alerts.append(whale_alert)
            
            # Store alerts
            if alerts:
                self.insider_alerts[token_address].extend(alerts)
                self.analytics_stats['insider_alerts_generated'] += len(alerts)
                
                for alert in alerts:
                    logger.warning(f"🚨 INSIDER ALERT: {alert.alert_type} | "
                                 f"{alert.token_symbol} | "
                                 f"Risk: {alert.risk_level} | "
                                 f"Confidence: {alert.confidence:.2f}")
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error detecting insider activity: {e}")
            return []
    
    async def get_wallet_analysis(self, wallet_address: str) -> WalletProfile:
        """Get comprehensive wallet analysis"""
        try:
            if wallet_address in self.wallet_profiles:
                return self.wallet_profiles[wallet_address]
            
            # Analyze wallet from scratch
            profile = await self._analyze_wallet_profile(wallet_address)
            self.wallet_profiles[wallet_address] = profile
            
            return profile
            
        except Exception as e:
            logger.error(f"Error analyzing wallet {wallet_address}: {e}")
            return WalletProfile(
                wallet_address=wallet_address,
                total_balance=0, total_value_usd=0, token_count=0,
                first_seen=time.time(), last_activity=time.time(),
                transaction_count=0, avg_transaction_size=0,
                success_rate=0, win_rate=0, total_pnl=0,
                behavior_tags=[], risk_score=1.0, influence_score=0.0
            )
    
    async def _get_token_info(self, token_address: str) -> Dict[str, Any]:
        """Get basic token information"""
        try:
            # This would integrate with Helius or other RPC services
            # For now, return placeholder data
            return {
                'total_supply': 1_000_000_000,
                'circulating_supply': 800_000_000,
                'dev_allocation': 0.1,  # 10% dev allocation
                'decimals': 6
            }
            
        except Exception as e:
            logger.error(f"Error getting token info: {e}")
            return {}
    
    async def _analyze_holder_distribution(self, token_address: str) -> List[Dict[str, Any]]:
        """Analyze token holder distribution"""
        try:
            # This would use Helius or other services to get holder data
            # For now, simulate holder distribution
            
            holders = []
            total_supply = 1_000_000_000
            
            # Simulate top holders (this would be real data in production)
            for i in range(100):
                # Power law distribution for realistic holder sizes
                holder_pct = (1.0 / (i + 1) ** 0.7) * 20  # Top holder has ~20%
                balance = total_supply * (holder_pct / 100)
                
                holders.append({
                    'address': f"whale_{i+1}_address",
                    'balance': balance,
                    'percentage': holder_pct,
                    'rank': i + 1
                })
            
            return holders
            
        except Exception as e:
            logger.error(f"Error analyzing holder distribution: {e}")
            return []
    
    def _calculate_concentration_score(self, holders: List[Dict[str, Any]]) -> float:
        """Calculate token concentration score (0-1, higher is more concentrated)"""
        try:
            if not holders:
                return 1.0  # Assume high concentration if no data
            
            # Calculate Gini coefficient for concentration
            total_supply = sum(h['balance'] for h in holders)
            if total_supply == 0:
                return 1.0
            
            # Top 10 holders percentage
            top_10_pct = sum(h['balance'] for h in holders[:10]) / total_supply
            
            # Top 1% of holders percentage
            top_1_pct_count = max(1, len(holders) // 100)
            top_1_pct = sum(h['balance'] for h in holders[:top_1_pct_count]) / total_supply
            
            # Concentration score based on top holder distribution
            concentration = (top_10_pct * 0.7) + (top_1_pct * 0.3)
            
            return min(1.0, concentration)
            
        except Exception as e:
            logger.error(f"Error calculating concentration score: {e}")
            return 1.0
    
    async def _analyze_whale_activity(self, token_address: str) -> float:
        """Analyze whale activity level (0-1 score)"""
        try:
            # Get recent whale transactions for this token
            recent_whale_txs = [
                tx for tx in self.whale_transactions
                if tx.token_address == token_address
                and time.time() - tx.timestamp < 86400  # Last 24 hours
            ]
            
            if not recent_whale_txs:
                return 0.0
            
            # Calculate activity metrics
            total_volume = sum(tx.value_usd for tx in recent_whale_txs)
            avg_price_impact = np.mean([tx.price_impact for tx in recent_whale_txs])
            unique_whales = len(set(tx.wallet_address for tx in recent_whale_txs))
            
            # Normalize to 0-1 score
            volume_score = min(1.0, total_volume / 1_000_000)  # $1M max
            impact_score = min(1.0, avg_price_impact / 0.1)    # 10% max impact
            whale_score = min(1.0, unique_whales / 20)         # 20 whales max
            
            activity_score = (volume_score + impact_score + whale_score) / 3
            
            return activity_score
            
        except Exception as e:
            logger.error(f"Error analyzing whale activity: {e}")
            return 0.0
    
    async def _detect_insider_activity(self, token_address: str) -> float:
        """Detect insider activity level (0-1 score)"""
        try:
            # This would analyze transaction patterns for insider trading
            # For now, return a simulated score based on recent alerts
            
            recent_alerts = [
                alert for alert in self.insider_alerts.get(token_address, [])
                if time.time() - alert.timestamp < 86400  # Last 24 hours
            ]
            
            if not recent_alerts:
                return 0.0
            
            # Calculate insider score based on alert severity
            insider_score = 0.0
            for alert in recent_alerts:
                if alert.risk_level == 'critical':
                    insider_score += 0.4
                elif alert.risk_level == 'high':
                    insider_score += 0.3
                elif alert.risk_level == 'medium':
                    insider_score += 0.2
                else:
                    insider_score += 0.1
            
            return min(1.0, insider_score)
            
        except Exception as e:
            logger.error(f"Error detecting insider activity: {e}")
            return 0.0
    
    async def _calculate_liquidity_score(self, token_address: str) -> float:
        """Calculate liquidity health score (0-1)"""
        try:
            # This would analyze DEX liquidity pools
            # For now, return a simulated score
            import random
            return random.uniform(0.3, 0.9)
            
        except Exception as e:
            logger.error(f"Error calculating liquidity score: {e}")
            return 0.0
    
    async def _assess_risk_factors(
        self, 
        token_address: str, 
        holders: List[Dict], 
        whale_activity: float
    ) -> List[str]:
        """Assess risk factors for the token"""
        try:
            risk_factors = []
            
            # High concentration risk
            if len(holders) > 0:
                top_holder_pct = holders[0]['percentage']
                if top_holder_pct > 50:
                    risk_factors.append("single_holder_majority")
                elif top_holder_pct > 30:
                    risk_factors.append("high_concentration")
            
            # Low holder count
            if len(holders) < 100:
                risk_factors.append("low_holder_count")
            
            # High whale activity
            if whale_activity > 0.7:
                risk_factors.append("high_whale_activity")
            
            # Check for known risk patterns
            recent_alerts = self.insider_alerts.get(token_address, [])
            if any(alert.risk_level in ['high', 'critical'] for alert in recent_alerts):
                risk_factors.append("insider_activity_detected")
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Error assessing risk factors: {e}")
            return ["analysis_error"]
    
    async def _identify_bullish_indicators(
        self, 
        token_address: str, 
        holders: List[Dict]
    ) -> List[str]:
        """Identify bullish indicators for the token"""
        try:
            bullish_indicators = []
            
            # Growing holder base
            if len(holders) > 1000:
                bullish_indicators.append("large_holder_base")
            
            # Healthy distribution
            if len(holders) > 0:
                top_5_pct = sum(h['percentage'] for h in holders[:5])
                if top_5_pct < 40:  # Top 5 holders < 40%
                    bullish_indicators.append("healthy_distribution")
            
            # Positive whale activity
            recent_whale_txs = [
                tx for tx in self.whale_transactions
                if tx.token_address == token_address
                and time.time() - tx.timestamp < 3600  # Last hour
                and tx.transaction_type == 'buy'
            ]
            
            if len(recent_whale_txs) > 3:
                bullish_indicators.append("whale_accumulation")
            
            return bullish_indicators
            
        except Exception as e:
            logger.error(f"Error identifying bullish indicators: {e}")
            return []
    
    def _assess_distribution_health(
        self, 
        concentration_score: float, 
        holder_count: int, 
        whale_activity: float
    ) -> str:
        """Assess overall distribution health"""
        try:
            # Calculate health score
            health_score = 0
            
            # Concentration (lower is better)
            if concentration_score < 0.3:
                health_score += 3
            elif concentration_score < 0.5:
                health_score += 2
            elif concentration_score < 0.7:
                health_score += 1
            
            # Holder count
            if holder_count > 5000:
                health_score += 3
            elif holder_count > 1000:
                health_score += 2
            elif holder_count > 500:
                health_score += 1
            
            # Whale activity (moderate is best)
            if 0.3 <= whale_activity <= 0.6:
                health_score += 2
            elif whale_activity < 0.8:
                health_score += 1
            
            # Map score to health rating
            if health_score >= 7:
                return "excellent"
            elif health_score >= 5:
                return "good"
            elif health_score >= 3:
                return "fair"
            else:
                return "poor"
                
        except Exception as e:
            logger.error(f"Error assessing distribution health: {e}")
            return "unknown"
    
    async def _get_large_transactions(self, token_address: str) -> List[Dict]:
        """Get large transactions for a token"""
        try:
            # This would integrate with Helius or other services
            # For now, simulate some transactions
            transactions = []
            
            # Simulate 5 large transactions
            for i in range(5):
                transactions.append({
                    'signature': f"tx_signature_{i}",
                    'wallet': f"whale_wallet_{i}",
                    'type': 'buy' if i % 2 == 0 else 'sell',
                    'amount': 100000 + (i * 50000),
                    'value_usd': (100000 + (i * 50000)) * 0.001,
                    'timestamp': time.time() - (i * 3600),
                    'block': 200000000 + i
                })
            
            return transactions
            
        except Exception as e:
            logger.error(f"Error getting large transactions: {e}")
            return []
    
    async def _analyze_whale_transaction(self, tx_data: Dict) -> Optional[WhaleTransaction]:
        """Analyze a transaction for whale characteristics"""
        try:
            value_usd = tx_data.get('value_usd', 0)
            
            # Only process large transactions
            if value_usd < self.whale_config['min_transaction_usd']:
                return None
            
            # Create whale transaction object
            whale_tx = WhaleTransaction(
                transaction_signature=tx_data['signature'],
                wallet_address=tx_data['wallet'],
                token_address="",  # Would be filled from transaction data
                token_symbol="",
                transaction_type=tx_data['type'],
                amount=tx_data['amount'],
                value_usd=value_usd,
                price_impact=0.05,  # Would calculate from DEX data
                timestamp=tx_data['timestamp'],
                block_height=tx_data['block'],
                wallet_balance_before=0,  # Would get from transaction
                wallet_balance_after=0,
                is_first_interaction=False,  # Would check transaction history
                confidence_score=0.8
            )
            
            return whale_tx
            
        except Exception as e:
            logger.error(f"Error analyzing whale transaction: {e}")
            return None
    
    async def _update_wallet_profile(self, wallet_address: str, whale_tx: WhaleTransaction):
        """Update wallet profile with new transaction data"""
        try:
            if wallet_address not in self.wallet_profiles:
                # Create new profile
                self.wallet_profiles[wallet_address] = WalletProfile(
                    wallet_address=wallet_address,
                    total_balance=0, total_value_usd=0, token_count=0,
                    first_seen=whale_tx.timestamp, last_activity=whale_tx.timestamp,
                    transaction_count=1, avg_transaction_size=whale_tx.value_usd,
                    success_rate=0, win_rate=0, total_pnl=0,
                    behavior_tags=[], risk_score=0.5, influence_score=0.1
                )
            else:
                # Update existing profile
                profile = self.wallet_profiles[wallet_address]
                profile.last_activity = whale_tx.timestamp
                profile.transaction_count += 1
                
                # Update average transaction size
                total_value = (profile.avg_transaction_size * (profile.transaction_count - 1) + 
                              whale_tx.value_usd)
                profile.avg_transaction_size = total_value / profile.transaction_count
            
            self.analytics_stats['whales_tracked'] += 1
            
        except Exception as e:
            logger.error(f"Error updating wallet profile: {e}")
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics performance summary"""
        try:
            active_whale_count = len([
                addr for addr, profile in self.wallet_profiles.items()
                if time.time() - profile.last_activity < 86400  # Active in last 24h
            ])
            
            recent_alerts = sum(
                len([alert for alert in alerts if time.time() - alert.timestamp < 86400])
                for alerts in self.insider_alerts.values()
            )
            
            return {
                **self.analytics_stats,
                'active_whales_24h': active_whale_count,
                'recent_alerts_24h': recent_alerts,
                'tracked_wallets': len(self.wallet_profiles),
                'recent_whale_transactions': len([
                    tx for tx in self.whale_transactions
                    if time.time() - tx.timestamp < 86400
                ])
            }
            
        except Exception as e:
            logger.error(f"Error getting analytics summary: {e}")
            return {'error': str(e)}

# Helper functions
def create_onchain_analyzer(helius_service=None) -> OnChainAnalyzer:
    """Create an on-chain analyzer instance"""
    return OnChainAnalyzer(helius_service) 