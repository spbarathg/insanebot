"""
AI Deception Shield - Advanced Manipulation Detection System

This module implements real-time detection of:
- LP removal pattern monitoring  
- Whale wallet clustering analysis
- Social sentiment credibility scoring
- Telegram/Discord pump signal scanning
- Coordinated attack detection

AI KILL CODE: Force close positions when rug_pull_confidence > 80% OR whale_exit_detected = True
"""

import asyncio
import time
import logging
import hashlib
import re
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
import statistics
import numpy as np

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Threat severity levels"""
    LOW = "low"           # 0-25% confidence
    MODERATE = "moderate" # 26-50% confidence
    HIGH = "high"        # 51-75% confidence
    CRITICAL = "critical" # 76-90% confidence
    LETHAL = "lethal"    # 91-100% confidence

class ManipulationType(Enum):
    """Types of detected manipulation"""
    RUG_PULL = "rug_pull"
    WHALE_MANIPULATION = "whale_manipulation" 
    PUMP_AND_DUMP = "pump_and_dump"
    SOCIAL_ENGINEERING = "social_engineering"
    LIQUIDITY_DRAIN = "liquidity_drain"
    COORDINATED_ATTACK = "coordinated_attack"
    BOT_ACTIVITY = "bot_activity"
    HONEYPOT_TRAP = "honeypot_trap"

@dataclass
class ThreatAlert:
    """Real-time threat detection alert"""
    threat_id: str
    token_address: str
    token_symbol: str
    threat_type: ManipulationType
    threat_level: ThreatLevel
    confidence_score: float  # 0-100
    
    # Detection details
    detected_patterns: List[str]
    evidence: Dict[str, Any]
    risk_indicators: List[str]
    
    # Timeline
    first_detected: float
    last_updated: float
    detection_count: int = 1
    
    # Actions
    recommended_action: str = "MONITOR"  # MONITOR, REDUCE, EXIT, BLACKLIST
    auto_action_taken: bool = False
    
    @property
    def is_kill_signal(self) -> bool:
        """Check if this threat should trigger immediate exit"""
        return (self.confidence_score >= 80.0 or 
                self.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.LETHAL])
    
    @property
    def severity_emoji(self) -> str:
        """Get severity emoji for logging"""
        if self.threat_level == ThreatLevel.LETHAL:
            return "☠️"
        elif self.threat_level == ThreatLevel.CRITICAL:
            return "🚨"
        elif self.threat_level == ThreatLevel.HIGH:
            return "⚠️"
        elif self.threat_level == ThreatLevel.MODERATE:
            return "🟡"
        else:
            return "🟢"

class RugPullDetector:
    """Detects rug pull patterns in real-time"""
    
    def __init__(self):
        self.lp_monitoring = {}  # Track LP changes
        self.ownership_monitoring = {}  # Track ownership changes
        self.contract_monitoring = {}  # Track contract modifications
        
        # Rug pull signatures
        self.rug_pull_patterns = {
            'massive_lp_removal': {'threshold': 0.5, 'weight': 80},  # >50% LP removed
            'creator_dump': {'threshold': 0.3, 'weight': 70},        # Creator sells >30%
            'liquidity_lock_removal': {'threshold': 1.0, 'weight': 90}, # Lock removed
            'contract_modification': {'threshold': 1.0, 'weight': 85}, # Contract changed
            'fee_manipulation': {'threshold': 0.15, 'weight': 75},    # Fees >15%
            'transfer_disabled': {'threshold': 1.0, 'weight': 95},    # Transfers disabled
        }
    
    async def analyze_rug_pull_risk(self, token_address: str, 
                                  current_lp_data: Dict,
                                  holder_data: List[Dict],
                                  contract_data: Dict) -> Tuple[float, List[str]]:
        """Analyze rug pull risk for a token"""
        try:
            risk_score = 0.0
            detected_patterns = []
            
            # 1. LP Removal Pattern Analysis
            lp_risk, lp_patterns = await self._analyze_lp_patterns(token_address, current_lp_data)
            risk_score += lp_risk
            detected_patterns.extend(lp_patterns)
            
            # 2. Creator/Whale Dump Analysis  
            dump_risk, dump_patterns = await self._analyze_creator_activity(token_address, holder_data)
            risk_score += dump_risk
            detected_patterns.extend(dump_patterns)
            
            # 3. Contract Modification Analysis
            contract_risk, contract_patterns = await self._analyze_contract_changes(token_address, contract_data)
            risk_score += contract_risk
            detected_patterns.extend(contract_patterns)
            
            # 4. Fee Manipulation Analysis
            fee_risk, fee_patterns = await self._analyze_fee_changes(token_address, contract_data)
            risk_score += fee_risk
            detected_patterns.extend(fee_patterns)
            
            # Cap at 100
            risk_score = min(100.0, risk_score)
            
            return risk_score, detected_patterns
            
        except Exception as e:
            logger.error(f"Error analyzing rug pull risk: {str(e)}")
            return 50.0, [f"Analysis error: {str(e)}"]
    
    async def _analyze_lp_patterns(self, token_address: str, current_lp_data: Dict) -> Tuple[float, List[str]]:
        """Analyze liquidity pool manipulation patterns"""
        try:
            patterns = []
            risk_score = 0.0
            
            # Track LP changes over time
            if token_address not in self.lp_monitoring:
                self.lp_monitoring[token_address] = deque(maxlen=100)
            
            current_lp = current_lp_data.get('total_liquidity', 0)
            current_time = time.time()
            
            self.lp_monitoring[token_address].append({
                'liquidity': current_lp,
                'timestamp': current_time
            })
            
            lp_history = self.lp_monitoring[token_address]
            
            if len(lp_history) >= 2:
                # Check for rapid LP drainage
                recent_lp = lp_history[-1]['liquidity']
                earlier_lp = lp_history[0]['liquidity']
                
                if earlier_lp > 0:
                    lp_change_percent = (earlier_lp - recent_lp) / earlier_lp
                    
                    if lp_change_percent > 0.5:  # >50% LP removed
                        risk_score += 80
                        patterns.append(f"🚨 MASSIVE LP REMOVAL: {lp_change_percent:.1%}")
                    elif lp_change_percent > 0.3:  # >30% LP removed
                        risk_score += 50
                        patterns.append(f"⚠️ Significant LP removal: {lp_change_percent:.1%}")
                    elif lp_change_percent > 0.15:  # >15% LP removed
                        risk_score += 25
                        patterns.append(f"🟡 Moderate LP removal: {lp_change_percent:.1%}")
                
                # Check for LP removal velocity
                if len(lp_history) >= 10:
                    recent_5min = [h for h in lp_history if current_time - h['timestamp'] <= 300]
                    if len(recent_5min) >= 2:
                        start_lp = recent_5min[0]['liquidity']
                        end_lp = recent_5min[-1]['liquidity']
                        
                        if start_lp > 0:
                            velocity = (start_lp - end_lp) / start_lp
                            if velocity > 0.2:  # >20% in 5 minutes
                                risk_score += 60
                                patterns.append(f"🚨 RAPID LP DRAIN: {velocity:.1%} in 5min")
            
            # Check LP lock status
            if current_lp_data.get('lock_removed', False):
                risk_score += 90
                patterns.append("🚨 LIQUIDITY LOCK REMOVED")
            
            return risk_score, patterns
            
        except Exception as e:
            logger.error(f"Error analyzing LP patterns: {str(e)}")
            return 0.0, []
    
    async def _analyze_creator_activity(self, token_address: str, holder_data: List[Dict]) -> Tuple[float, List[str]]:
        """Analyze creator/whale dumping patterns"""
        try:
            patterns = []
            risk_score = 0.0
            
            if not holder_data:
                return 0.0, []
            
            # Track creator/top holder activity
            if token_address not in self.ownership_monitoring:
                self.ownership_monitoring[token_address] = deque(maxlen=50)
            
            # Identify potential creator (largest holder)
            top_holders = sorted(holder_data, key=lambda x: x.get('balance', 0), reverse=True)[:5]
            
            current_time = time.time()
            self.ownership_monitoring[token_address].append({
                'top_holders': top_holders,
                'timestamp': current_time
            })
            
            ownership_history = self.ownership_monitoring[token_address]
            
            if len(ownership_history) >= 2:
                # Compare recent ownership with earlier
                current_top = ownership_history[-1]['top_holders']
                earlier_top = ownership_history[0]['top_holders']
                
                # Check for major holder dumps
                for i, current_holder in enumerate(current_top[:3]):  # Top 3 holders
                    current_balance = current_holder.get('balance', 0)
                    
                    # Find same holder in earlier data
                    earlier_holder = None
                    current_address = current_holder.get('address', '')
                    
                    for eh in earlier_top:
                        if eh.get('address', '') == current_address:
                            earlier_holder = eh
                            break
                    
                    if earlier_holder:
                        earlier_balance = earlier_holder.get('balance', 0)
                        
                        if earlier_balance > 0:
                            dump_percent = (earlier_balance - current_balance) / earlier_balance
                            
                            if dump_percent > 0.5:  # >50% dump
                                risk_score += 70
                                patterns.append(f"🚨 TOP HOLDER DUMP: {dump_percent:.1%} (Rank {i+1})")
                            elif dump_percent > 0.3:  # >30% dump
                                risk_score += 40
                                patterns.append(f"⚠️ Significant holder dump: {dump_percent:.1%} (Rank {i+1})")
            
            # Check for unusual holder concentration changes
            if len(holder_data) >= 2:
                top_holder_percent = holder_data[0].get('percentage', 0) if holder_data else 0
                
                if top_holder_percent > 80:  # >80% held by top holder
                    risk_score += 60
                    patterns.append(f"🚨 EXTREME CONCENTRATION: {top_holder_percent:.1%}")
                elif top_holder_percent > 60:  # >60% held by top holder
                    risk_score += 35
                    patterns.append(f"⚠️ High concentration: {top_holder_percent:.1%}")
            
            return risk_score, patterns
            
        except Exception as e:
            logger.error(f"Error analyzing creator activity: {str(e)}")
            return 0.0, []
    
    async def _analyze_contract_changes(self, token_address: str, contract_data: Dict) -> Tuple[float, List[str]]:
        """Analyze contract modification patterns"""
        try:
            patterns = []
            risk_score = 0.0
            
            # Track contract state changes
            if token_address not in self.contract_monitoring:
                self.contract_monitoring[token_address] = deque(maxlen=20)
            
            current_state = {
                'can_mint': contract_data.get('can_mint', False),
                'can_pause': contract_data.get('can_pause', False),
                'owner_can_change_fees': contract_data.get('owner_can_change_fees', False),
                'transfers_enabled': contract_data.get('transfers_enabled', True),
                'current_fees': contract_data.get('trading_fees', {}),
                'timestamp': time.time()
            }
            
            self.contract_monitoring[token_address].append(current_state)
            
            contract_history = self.contract_monitoring[token_address]
            
            # Check for dangerous contract features
            if current_state['can_mint']:
                risk_score += 30
                patterns.append("⚠️ Contract can mint new tokens")
            
            if current_state['can_pause']:
                risk_score += 40
                patterns.append("🚨 Contract can be paused")
            
            if not current_state['transfers_enabled']:
                risk_score += 95
                patterns.append("🚨 TRANSFERS DISABLED")
            
            if current_state['owner_can_change_fees']:
                risk_score += 25
                patterns.append("⚠️ Owner can modify fees")
            
            # Check for recent contract modifications
            if len(contract_history) >= 2:
                previous_state = contract_history[-2]
                
                # Check if transfers were disabled recently
                if previous_state['transfers_enabled'] and not current_state['transfers_enabled']:
                    risk_score += 85
                    patterns.append("🚨 TRANSFERS JUST DISABLED")
                
                # Check for fee increases
                current_fees = current_state.get('current_fees', {})
                previous_fees = previous_state.get('current_fees', {})
                
                sell_fee_current = current_fees.get('sell_fee', 0)
                sell_fee_previous = previous_fees.get('sell_fee', 0)
                
                if sell_fee_current > sell_fee_previous + 5:  # Fee increased by >5%
                    risk_score += 50
                    patterns.append(f"🚨 SELL FEE INCREASED: {sell_fee_previous}% → {sell_fee_current}%")
            
            return risk_score, patterns
            
        except Exception as e:
            logger.error(f"Error analyzing contract changes: {str(e)}")
            return 0.0, []
    
    async def _analyze_fee_changes(self, token_address: str, contract_data: Dict) -> Tuple[float, List[str]]:
        """Analyze trading fee manipulation"""
        try:
            patterns = []
            risk_score = 0.0
            
            current_fees = contract_data.get('trading_fees', {})
            buy_fee = current_fees.get('buy_fee', 0)
            sell_fee = current_fees.get('sell_fee', 0)
            
            # Check for excessive fees
            if sell_fee > 25:  # >25% sell fee
                risk_score += 80
                patterns.append(f"🚨 EXCESSIVE SELL FEE: {sell_fee}%")
            elif sell_fee > 15:  # >15% sell fee
                risk_score += 50
                patterns.append(f"⚠️ High sell fee: {sell_fee}%")
            elif sell_fee > 10:  # >10% sell fee
                risk_score += 25
                patterns.append(f"🟡 Moderate sell fee: {sell_fee}%")
            
            if buy_fee > 15:  # >15% buy fee
                risk_score += 40
                patterns.append(f"⚠️ High buy fee: {buy_fee}%")
            
            # Check for fee asymmetry (sell much higher than buy)
            if sell_fee > buy_fee * 3 and sell_fee > 10:
                risk_score += 60
                patterns.append(f"🚨 FEE ASYMMETRY: Buy {buy_fee}%, Sell {sell_fee}%")
            
            return risk_score, patterns
            
        except Exception as e:
            logger.error(f"Error analyzing fee changes: {str(e)}")
            return 0.0, []

class WhaleActivityMonitor:
    """Monitors whale wallet movements and clustering"""
    
    def __init__(self):
        self.whale_wallets = set()  # Known whale addresses
        self.whale_activity = defaultdict(deque)  # Track whale movements
        self.clustering_analysis = {}  # Detect coordinated activity
        
        # Whale thresholds
        self.whale_threshold_sol = 100  # 100+ SOL positions
        self.suspicious_correlation_threshold = 0.8  # 80% correlation
        
    async def analyze_whale_activity(self, token_address: str, 
                                   transaction_data: List[Dict],
                                   holder_data: List[Dict]) -> Tuple[float, List[str]]:
        """Analyze whale manipulation patterns"""
        try:
            risk_score = 0.0
            patterns = []
            
            # 1. Identify whale wallets
            whales = await self._identify_whales(holder_data)
            
            # 2. Analyze whale transaction patterns
            whale_tx_risk, whale_patterns = await self._analyze_whale_transactions(
                token_address, transaction_data, whales
            )
            risk_score += whale_tx_risk
            patterns.extend(whale_patterns)
            
            # 3. Detect coordinated whale activity
            coordination_risk, coordination_patterns = await self._detect_coordination(
                token_address, whales, transaction_data
            )
            risk_score += coordination_risk
            patterns.extend(coordination_patterns)
            
            # 4. Analyze whale exit patterns
            exit_risk, exit_patterns = await self._analyze_exit_patterns(
                token_address, whales, transaction_data
            )
            risk_score += exit_risk
            patterns.extend(exit_patterns)
            
            return min(100.0, risk_score), patterns
            
        except Exception as e:
            logger.error(f"Error analyzing whale activity: {str(e)}")
            return 0.0, []
    
    async def _identify_whales(self, holder_data: List[Dict]) -> List[Dict]:
        """Identify whale wallets based on holdings"""
        whales = []
        
        for holder in holder_data:
            balance_sol = holder.get('balance_sol', 0)
            if balance_sol >= self.whale_threshold_sol:
                whales.append(holder)
                self.whale_wallets.add(holder.get('address', ''))
        
        return whales
    
    async def _analyze_whale_transactions(self, token_address: str, 
                                        transaction_data: List[Dict],
                                        whales: List[Dict]) -> Tuple[float, List[str]]:
        """Analyze whale transaction patterns"""
        try:
            risk_score = 0.0
            patterns = []
            
            whale_addresses = {w.get('address', '') for w in whales}
            
            # Analyze recent transactions
            whale_sells = []
            whale_buys = []
            
            for tx in transaction_data:
                if tx.get('from_address') in whale_addresses:
                    if tx.get('type') == 'sell':
                        whale_sells.append(tx)
                    elif tx.get('type') == 'buy':
                        whale_buys.append(tx)
            
            # Check for whale dumping
            if whale_sells:
                total_whale_sell_volume = sum(tx.get('amount_sol', 0) for tx in whale_sells)
                
                if total_whale_sell_volume > 50:  # >50 SOL in whale sells
                    risk_score += 60
                    patterns.append(f"🚨 WHALE DUMPING: {total_whale_sell_volume:.1f} SOL")
                elif total_whale_sell_volume > 20:  # >20 SOL in whale sells
                    risk_score += 30
                    patterns.append(f"⚠️ Whale selling: {total_whale_sell_volume:.1f} SOL")
            
            # Check for unusual whale activity timing
            if len(whale_sells) > 0 and len(whale_buys) == 0:
                # Only selling, no buying from whales
                risk_score += 40
                patterns.append("⚠️ Whales only selling (no buying)")
            
            # Check for large single whale transactions
            for tx in whale_sells:
                amount = tx.get('amount_sol', 0)
                if amount > 25:  # >25 SOL single transaction
                    risk_score += 35
                    patterns.append(f"⚠️ Large whale sell: {amount:.1f} SOL")
            
            return risk_score, patterns
            
        except Exception as e:
            logger.error(f"Error analyzing whale transactions: {str(e)}")
            return 0.0, []
    
    async def _detect_coordination(self, token_address: str, 
                                 whales: List[Dict],
                                 transaction_data: List[Dict]) -> Tuple[float, List[str]]:
        """Detect coordinated whale activity"""
        try:
            risk_score = 0.0
            patterns = []
            
            if len(whales) < 2:
                return 0.0, []
            
            # Group transactions by time windows
            time_windows = defaultdict(list)
            
            for tx in transaction_data:
                window = int(tx.get('timestamp', 0) // 300)  # 5-minute windows
                time_windows[window].append(tx)
            
            # Check for coordinated activity
            for window, txs in time_windows.items():
                whale_txs = [tx for tx in txs if tx.get('from_address') in {w.get('address') for w in whales}]
                
                if len(whale_txs) >= 3:  # 3+ whale transactions in same window
                    # Check if they're all selling
                    sell_count = sum(1 for tx in whale_txs if tx.get('type') == 'sell')
                    
                    if sell_count >= 3:
                        risk_score += 70
                        patterns.append(f"🚨 COORDINATED WHALE DUMP: {sell_count} whales selling simultaneously")
                    elif sell_count >= 2:
                        risk_score += 40
                        patterns.append(f"⚠️ Multiple whales selling together: {sell_count}")
            
            return risk_score, patterns
            
        except Exception as e:
            logger.error(f"Error detecting coordination: {str(e)}")
            return 0.0, []
    
    async def _analyze_exit_patterns(self, token_address: str,
                                   whales: List[Dict],
                                   transaction_data: List[Dict]) -> Tuple[float, List[str]]:
        """Analyze whale exit patterns"""
        try:
            risk_score = 0.0
            patterns = []
            
            # Check for whale position liquidation patterns
            for whale in whales:
                whale_address = whale.get('address', '')
                
                # Find all transactions from this whale
                whale_txs = [tx for tx in transaction_data if tx.get('from_address') == whale_address]
                whale_sells = [tx for tx in whale_txs if tx.get('type') == 'sell']
                
                if whale_sells:
                    total_sold = sum(tx.get('amount_sol', 0) for tx in whale_sells)
                    current_balance = whale.get('balance_sol', 0)
                    
                    # Check if whale is liquidating position
                    if current_balance > 0:
                        liquidation_ratio = total_sold / (total_sold + current_balance)
                        
                        if liquidation_ratio > 0.8:  # Sold >80% of position
                            risk_score += 50
                            patterns.append(f"🚨 WHALE LIQUIDATION: {liquidation_ratio:.1%} position sold")
                        elif liquidation_ratio > 0.5:  # Sold >50% of position
                            risk_score += 25
                            patterns.append(f"⚠️ Partial whale liquidation: {liquidation_ratio:.1%}")
            
            return risk_score, patterns
            
        except Exception as e:
            logger.error(f"Error analyzing exit patterns: {str(e)}")
            return 0.0, []

class SocialSentimentScanner:
    """Scans social media for pump signals and credibility issues"""
    
    def __init__(self):
        self.pump_signal_patterns = [
            r'🚀+.*moon',
            r'pump.*incoming',
            r'\d+x.*incoming',
            r'next.*\d+x',
            r'gem.*alert',
            r'buy.*before.*pump',
            r'insider.*info',
            r'pre.*pump',
        ]
        
        self.bot_indicators = [
            r'^[A-Z]{8,}$',  # All caps random usernames
            r'^\w+\d{4,}$',  # Username with many numbers
            r'.*crypto.*bot.*',
        ]
        
        # Track suspicious patterns
        self.suspicious_accounts = set()
        self.pump_groups = set()
        
    async def analyze_social_sentiment(self, token_address: str, 
                                     social_data: Dict) -> Tuple[float, List[str]]:
        """Analyze social sentiment for manipulation signals"""
        try:
            risk_score = 0.0
            patterns = []
            
            # 1. Analyze pump signal density
            pump_risk, pump_patterns = await self._analyze_pump_signals(social_data)
            risk_score += pump_risk
            patterns.extend(pump_patterns)
            
            # 2. Detect bot activity
            bot_risk, bot_patterns = await self._detect_bot_activity(social_data)
            risk_score += bot_risk
            patterns.extend(bot_patterns)
            
            # 3. Analyze sentiment credibility
            credibility_risk, credibility_patterns = await self._analyze_credibility(social_data)
            risk_score += credibility_risk
            patterns.extend(credibility_patterns)
            
            # 4. Detect coordinated campaigns
            campaign_risk, campaign_patterns = await self._detect_campaigns(social_data)
            risk_score += campaign_risk
            patterns.extend(campaign_patterns)
            
            return min(100.0, risk_score), patterns
            
        except Exception as e:
            logger.error(f"Error analyzing social sentiment: {str(e)}")
            return 0.0, []
    
    async def _analyze_pump_signals(self, social_data: Dict) -> Tuple[float, List[str]]:
        """Analyze pump signal density and patterns"""
        try:
            risk_score = 0.0
            patterns = []
            
            messages = social_data.get('messages', [])
            if not messages:
                return 0.0, []
            
            pump_signal_count = 0
            
            for message in messages:
                text = message.get('text', '').lower()
                
                # Check for pump signal patterns
                for pattern in self.pump_signal_patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        pump_signal_count += 1
                        break
            
            # Calculate pump signal density
            if len(messages) > 0:
                pump_density = pump_signal_count / len(messages)
                
                if pump_density > 0.3:  # >30% pump signals
                    risk_score += 70
                    patterns.append(f"🚨 HIGH PUMP SIGNAL DENSITY: {pump_density:.1%}")
                elif pump_density > 0.2:  # >20% pump signals
                    risk_score += 40
                    patterns.append(f"⚠️ Moderate pump signals: {pump_density:.1%}")
                elif pump_density > 0.1:  # >10% pump signals
                    risk_score += 20
                    patterns.append(f"🟡 Some pump signals: {pump_density:.1%}")
            
            return risk_score, patterns
            
        except Exception as e:
            logger.error(f"Error analyzing pump signals: {str(e)}")
            return 0.0, []
    
    async def _detect_bot_activity(self, social_data: Dict) -> Tuple[float, List[str]]:
        """Detect bot activity in social mentions"""
        try:
            risk_score = 0.0
            patterns = []
            
            messages = social_data.get('messages', [])
            if not messages:
                return 0.0, []
            
            bot_indicators_found = 0
            total_users = set()
            suspicious_users = set()
            
            for message in messages:
                username = message.get('username', '')
                text = message.get('text', '')
                
                total_users.add(username)
                
                # Check for bot username patterns
                for pattern in self.bot_indicators:
                    if re.match(pattern, username):
                        suspicious_users.add(username)
                        bot_indicators_found += 1
                        break
                
                # Check for copy-paste content
                if len(text) > 100 and text.count('🚀') > 5:
                    suspicious_users.add(username)
                    bot_indicators_found += 1
            
            # Calculate bot activity ratio
            if len(total_users) > 0:
                bot_ratio = len(suspicious_users) / len(total_users)
                
                if bot_ratio > 0.4:  # >40% bots
                    risk_score += 60
                    patterns.append(f"🚨 HIGH BOT ACTIVITY: {bot_ratio:.1%}")
                elif bot_ratio > 0.2:  # >20% bots
                    risk_score += 30
                    patterns.append(f"⚠️ Moderate bot activity: {bot_ratio:.1%}")
            
            # Track suspicious accounts
            self.suspicious_accounts.update(suspicious_users)
            
            return risk_score, patterns
            
        except Exception as e:
            logger.error(f"Error detecting bot activity: {str(e)}")
            return 0.0, []
    
    async def _analyze_credibility(self, social_data: Dict) -> Tuple[float, List[str]]:
        """Analyze sentiment credibility"""
        try:
            risk_score = 0.0
            patterns = []
            
            messages = social_data.get('messages', [])
            if not messages:
                return 0.0, []
            
            # Check for account age and credibility
            new_accounts = 0
            total_accounts = 0
            
            for message in messages:
                account_age_days = message.get('account_age_days', 0)
                follower_count = message.get('follower_count', 0)
                
                total_accounts += 1
                
                # Flag new accounts with low followers
                if account_age_days < 30 and follower_count < 100:
                    new_accounts += 1
            
            if total_accounts > 0:
                new_account_ratio = new_accounts / total_accounts
                
                if new_account_ratio > 0.6:  # >60% new accounts
                    risk_score += 50
                    patterns.append(f"🚨 SUSPICIOUS ACCOUNTS: {new_account_ratio:.1%} new/low-follower")
                elif new_account_ratio > 0.4:  # >40% new accounts
                    risk_score += 25
                    patterns.append(f"⚠️ Many new accounts: {new_account_ratio:.1%}")
            
            return risk_score, patterns
            
        except Exception as e:
            logger.error(f"Error analyzing credibility: {str(e)}")
            return 0.0, []
    
    async def _detect_campaigns(self, social_data: Dict) -> Tuple[float, List[str]]:
        """Detect coordinated social media campaigns"""
        try:
            risk_score = 0.0
            patterns = []
            
            messages = social_data.get('messages', [])
            if not messages:
                return 0.0, []
            
            # Group messages by timestamp to detect coordination
            time_groups = defaultdict(list)
            
            for message in messages:
                timestamp = message.get('timestamp', 0)
                time_window = int(timestamp // 300)  # 5-minute windows
                time_groups[time_window].append(message)
            
            # Check for coordinated posting
            for window, window_messages in time_groups.items():
                if len(window_messages) >= 10:  # 10+ messages in 5 minutes
                    # Check if messages are similar (copy-paste)
                    texts = [msg.get('text', '') for msg in window_messages]
                    unique_texts = set(texts)
                    
                    similarity_ratio = 1 - (len(unique_texts) / len(texts))
                    
                    if similarity_ratio > 0.7:  # >70% similar messages
                        risk_score += 60
                        patterns.append(f"🚨 COORDINATED CAMPAIGN: {len(window_messages)} similar messages")
                    elif similarity_ratio > 0.5:  # >50% similar messages
                        risk_score += 30
                        patterns.append(f"⚠️ Possible coordination: {len(window_messages)} messages")
            
            return risk_score, patterns
            
        except Exception as e:
            logger.error(f"Error detecting campaigns: {str(e)}")
            return 0.0, []

class AIDeceptionShield:
    """Main shield coordinator integrating all deception detection systems"""
    
    def __init__(self):
        self.rug_pull_detector = RugPullDetector()
        self.whale_monitor = WhaleActivityMonitor()
        self.social_scanner = SocialSentimentScanner()
        
        # Active threats tracking
        self.active_threats: Dict[str, ThreatAlert] = {}
        self.threat_history = deque(maxlen=1000)
        
        # Kill switch settings
        self.auto_exit_threshold = 80.0  # Auto-exit at 80% confidence
        self.auto_blacklist_threshold = 90.0  # Auto-blacklist at 90% confidence
        
        # Statistics
        self.total_scans = 0
        self.threats_detected = 0
        self.auto_actions_taken = 0
        
        logger.info("🛡️ AI Deception Shield initialized - Manipulation detection active")
    
    async def initialize(self):
        """Initialize the defense system (compatibility method)"""
        # Defense system is ready to use after __init__
        return True

    async def scan_for_threats(self, token_address: str, 
                             market_data: Dict,
                             social_data: Dict = None,
                             transaction_data: List[Dict] = None,
                             holder_data: List[Dict] = None) -> ThreatAlert:
        """Comprehensive threat scanning"""
        
        self.total_scans += 1
        scan_start = time.time()
        
        logger.info(f"🔍 SHIELD SCAN: Analyzing {token_address[:8]}... for manipulation threats")
        
        try:
            # Initialize threat tracking
            threat_id = f"threat_{int(scan_start)}_{token_address[:8]}"
            detected_patterns = []
            evidence = {}
            risk_indicators = []
            total_confidence = 0.0
            threat_types = []
            
            # 1. Rug Pull Detection
            logger.debug("🔍 Scanning for rug pull patterns...")
            rug_risk, rug_patterns = await self.rug_pull_detector.analyze_rug_pull_risk(
                token_address,
                market_data.get('liquidity_data', {}),
                holder_data or [],
                market_data.get('contract_data', {})
            )
            
            if rug_risk > 0:
                detected_patterns.extend(rug_patterns)
                evidence['rug_pull_risk'] = rug_risk
                total_confidence += rug_risk * 0.4  # 40% weight
                if rug_risk > 50:
                    threat_types.append(ManipulationType.RUG_PULL)
                    risk_indicators.append(f"Rug pull risk: {rug_risk:.1f}%")
            
            # 2. Whale Activity Monitoring
            if transaction_data and holder_data:
                logger.debug("🐋 Scanning for whale manipulation...")
                whale_risk, whale_patterns = await self.whale_monitor.analyze_whale_activity(
                    token_address, transaction_data, holder_data
                )
                
                if whale_risk > 0:
                    detected_patterns.extend(whale_patterns)
                    evidence['whale_risk'] = whale_risk
                    total_confidence += whale_risk * 0.3  # 30% weight
                    if whale_risk > 40:
                        threat_types.append(ManipulationType.WHALE_MANIPULATION)
                        risk_indicators.append(f"Whale manipulation risk: {whale_risk:.1f}%")
            
            # 3. Social Sentiment Analysis
            if social_data:
                logger.debug("📱 Scanning for social manipulation...")
                social_risk, social_patterns = await self.social_scanner.analyze_social_sentiment(
                    token_address, social_data
                )
                
                if social_risk > 0:
                    detected_patterns.extend(social_patterns)
                    evidence['social_risk'] = social_risk
                    total_confidence += social_risk * 0.3  # 30% weight
                    if social_risk > 40:
                        threat_types.append(ManipulationType.SOCIAL_ENGINEERING)
                        risk_indicators.append(f"Social manipulation risk: {social_risk:.1f}%")
            
            # 4. Determine overall threat level and type
            primary_threat_type = ManipulationType.RUG_PULL  # Default
            if threat_types:
                # Use highest risk threat as primary
                risk_scores = {
                    ManipulationType.RUG_PULL: evidence.get('rug_pull_risk', 0),
                    ManipulationType.WHALE_MANIPULATION: evidence.get('whale_risk', 0),
                    ManipulationType.SOCIAL_ENGINEERING: evidence.get('social_risk', 0)
                }
                primary_threat_type = max(threat_types, key=lambda t: risk_scores.get(t, 0))
            
            # Determine threat level
            if total_confidence >= 90:
                threat_level = ThreatLevel.LETHAL
            elif total_confidence >= 75:
                threat_level = ThreatLevel.CRITICAL
            elif total_confidence >= 50:
                threat_level = ThreatLevel.HIGH
            elif total_confidence >= 25:
                threat_level = ThreatLevel.MODERATE
            else:
                threat_level = ThreatLevel.LOW
            
            # Determine recommended action
            if total_confidence >= self.auto_blacklist_threshold:
                recommended_action = "BLACKLIST"
            elif total_confidence >= self.auto_exit_threshold:
                recommended_action = "EXIT"
            elif total_confidence >= 50:
                recommended_action = "REDUCE"
            else:
                recommended_action = "MONITOR"
            
            # Create threat alert
            threat_alert = ThreatAlert(
                threat_id=threat_id,
                token_address=token_address,
                token_symbol=market_data.get('symbol', 'UNKNOWN'),
                threat_type=primary_threat_type,
                threat_level=threat_level,
                confidence_score=total_confidence,
                detected_patterns=detected_patterns,
                evidence=evidence,
                risk_indicators=risk_indicators,
                first_detected=scan_start,
                last_updated=scan_start,
                recommended_action=recommended_action
            )
            
            # Update active threats
            self.active_threats[token_address] = threat_alert
            self.threat_history.append(threat_alert)
            
            if total_confidence > 25:
                self.threats_detected += 1
            
            scan_time = time.time() - scan_start
            logger.info(f"🛡️ SHIELD VERDICT: {threat_alert.token_symbol} - "
                       f"{threat_alert.severity_emoji} {threat_alert.threat_level.value.upper()} "
                       f"({threat_alert.confidence_score:.1f}%) - {recommended_action} "
                       f"(Scan: {scan_time:.2f}s)")
            
            # Log detected patterns
            for pattern in detected_patterns:
                logger.warning(f"  🔍 {pattern}")
            
            return threat_alert
            
        except Exception as e:
            logger.error(f"🚨 SHIELD ERROR during threat scan: {str(e)}")
            return self._create_error_alert(token_address, market_data, str(e))
    
    def _create_error_alert(self, token_address: str, market_data: Dict, error: str) -> ThreatAlert:
        """Create error alert for scan failures"""
        return ThreatAlert(
            threat_id=f"error_{int(time.time())}_{token_address[:8]}",
            token_address=token_address,
            token_symbol=market_data.get('symbol', 'ERROR'),
            threat_type=ManipulationType.RUG_PULL,
            threat_level=ThreatLevel.MODERATE,
            confidence_score=50.0,
            detected_patterns=[f"🚨 SCAN ERROR: {error}"],
            evidence={'error': error},
            risk_indicators=[f"Threat scan failed: {error}"],
            first_detected=time.time(),
            last_updated=time.time(),
            recommended_action="MONITOR"
        )
    
    def check_kill_signals(self, token_address: str) -> Tuple[bool, str]:
        """Check if any active threats require immediate action"""
        if token_address in self.active_threats:
            threat = self.active_threats[token_address]
            
            if threat.is_kill_signal:
                reason = f"{threat.threat_type.value} confidence {threat.confidence_score:.1f}%"
                return True, reason
        
        return False, ""
    
    def get_shield_status(self) -> Dict[str, Any]:
        """Get current shield operational status"""
        active_critical = sum(1 for t in self.active_threats.values() 
                            if t.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.LETHAL])
        
        return {
            "total_scans": self.total_scans,
            "threats_detected": self.threats_detected,
            "active_threats": len(self.active_threats),
            "critical_threats": active_critical,
            "auto_actions_taken": self.auto_actions_taken,
            "detection_rate": (self.threats_detected / max(1, self.total_scans)) * 100,
            "shield_effectiveness": min(100, (self.threats_detected / max(1, self.total_scans)) * 300)
        }
    
    async def emergency_threat_response(self, token_address: str, threat_type: str):
        """Emergency response to detected threat"""
        logger.critical(f"🚨 EMERGENCY THREAT RESPONSE: {token_address[:8]}... - {threat_type}")
        
        # Create emergency threat alert
        emergency_alert = ThreatAlert(
            threat_id=f"emergency_{int(time.time())}",
            token_address=token_address,
            token_symbol="EMERGENCY",
            threat_type=ManipulationType.RUG_PULL,
            threat_level=ThreatLevel.LETHAL,
            confidence_score=100.0,
            detected_patterns=[f"🚨 EMERGENCY: {threat_type}"],
            evidence={'emergency_type': threat_type},
            risk_indicators=[f"Emergency threat: {threat_type}"],
            first_detected=time.time(),
            last_updated=time.time(),
            recommended_action="BLACKLIST",
            auto_action_taken=True
        )
        
        self.active_threats[token_address] = emergency_alert
        self.auto_actions_taken += 1 