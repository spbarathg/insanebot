"""
Smart Money Wallet Tracker

This service tracks profitable wallets and analyzes their trading patterns
to generate high-alpha trading signals for memecoin trading.

Features:
- Real-time wallet monitoring via Helius/QuickNode
- Transaction pattern analysis
- Copy trading signal generation
- Performance tracking and wallet scoring
- Integration with AI decision making
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import aiohttp
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class WalletTransaction:
    """Individual transaction from a tracked wallet"""
    signature: str
    wallet_address: str
    token_address: str
    action: str  # 'buy' or 'sell'
    amount_sol: float
    token_amount: float
    price: float
    timestamp: float
    block_time: int
    success: bool
    
@dataclass
class TradingSignal:
    """Trading signal generated from smart money activity"""
    signal_id: str
    wallet_address: str
    token_address: str
    action: str  # 'buy' or 'sell'
    confidence: float  # 0.0 to 1.0
    amount_sol: float
    reasoning: str
    wallet_score: float
    timestamp: float
    urgency: str  # 'low', 'medium', 'high', 'critical'

@dataclass
class WalletPerformance:
    """Performance metrics for a tracked wallet"""
    wallet_address: str
    total_trades: int
    winning_trades: int
    total_profit_sol: float
    win_rate: float
    avg_profit_per_trade: float
    avg_hold_time_hours: float
    largest_win: float
    largest_loss: float
    sharpe_ratio: float
    last_updated: float
    score: float  # Overall performance score 0-100

class SmartMoneyTracker:
    """
    Smart Money Wallet Tracking System
    
    Monitors profitable wallets and generates trading signals based on their activity.
    Integrates with Helius/QuickNode for real-time transaction monitoring.
    """
    
    def __init__(self, helius_service=None, quicknode_service=None):
        self.helius_service = helius_service
        self.quicknode_service = quicknode_service
        
        # Wallet tracking data
        self.tracked_wallets: Dict[str, Dict] = {}
        self.wallet_performances: Dict[str, WalletPerformance] = {}
        self.recent_transactions: deque = deque(maxlen=1000)
        self.active_signals: Dict[str, TradingSignal] = {}
        
        # Transaction history for analysis
        self.transaction_history: Dict[str, List[WalletTransaction]] = defaultdict(list)
        
        # Configuration
        self.config = {
            'min_transaction_sol': 0.1,  # Minimum SOL amount to track
            'max_copy_delay_seconds': 30,  # Max delay for copy trading
            'signal_expiry_seconds': 300,  # Signal expiry time
            'min_wallet_score': 60,  # Minimum score to generate signals
            'performance_lookback_days': 30,  # Days to analyze performance
            'websocket_enabled': True
        }
        
        # WebSocket connections for real-time monitoring
        self.websocket_connections: Dict[str, Any] = {}
        self.monitoring_active = False
        
        logger.info("Smart Money Tracker initialized")
    
    async def add_wallet(self, wallet_address: str, name: str = "", notes: str = "") -> bool:
        """Add a wallet to track"""
        try:
            # Validate wallet address
            if not self._is_valid_solana_address(wallet_address):
                logger.error(f"Invalid Solana wallet address: {wallet_address}")
                return False
            
            # Add to tracked wallets
            self.tracked_wallets[wallet_address] = {
                'address': wallet_address,
                'name': name,
                'notes': notes,
                'added_timestamp': time.time(),
                'last_seen': 0,
                'total_tracked_trades': 0,
                'monitoring_active': True
            }
            
            # Initialize performance tracking
            await self._initialize_wallet_performance(wallet_address)
            
            # Start monitoring if system is active
            if self.monitoring_active:
                await self._start_wallet_monitoring(wallet_address)
            
            logger.info(f"✅ Added wallet to tracking: {wallet_address[:8]}... ({name})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding wallet {wallet_address}: {e}")
            return False
    
    async def add_multiple_wallets(self, wallet_list: List[Dict[str, str]]) -> int:
        """Add multiple wallets from a list"""
        successful_adds = 0
        
        for wallet_data in wallet_list:
            address = wallet_data.get('address', '')
            name = wallet_data.get('name', '')
            notes = wallet_data.get('notes', '')
            
            if await self.add_wallet(address, name, notes):
                successful_adds += 1
            
            # Small delay to avoid rate limits
            await asyncio.sleep(0.1)
        
        logger.info(f"✅ Successfully added {successful_adds}/{len(wallet_list)} wallets")
        return successful_adds
    
    async def start_monitoring(self) -> bool:
        """Start monitoring all tracked wallets"""
        try:
            logger.info(f"🔍 Starting smart money monitoring for {len(self.tracked_wallets)} wallets...")
            self.monitoring_active = True
            
            # Start monitoring tasks for each wallet
            tasks = []
            for wallet_address in self.tracked_wallets.keys():
                task = asyncio.create_task(self._start_wallet_monitoring(wallet_address))
                tasks.append(task)
            
            # Start signal processing task
            signal_task = asyncio.create_task(self._process_signals_loop())
            tasks.append(signal_task)
            
            # Start performance update task
            performance_task = asyncio.create_task(self._update_performances_loop())
            tasks.append(performance_task)
            
            logger.info("✅ Smart money monitoring started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")
            return False
    
    async def _start_wallet_monitoring(self, wallet_address: str):
        """Start monitoring a specific wallet"""
        try:
            if not self.helius_service and not self.quicknode_service:
                logger.warning("No blockchain service available for monitoring")
                return
            
            # Use Helius for WebSocket monitoring (preferred)
            if self.helius_service and self.config['websocket_enabled']:
                await self._start_helius_websocket_monitoring(wallet_address)
            else:
                # Fallback to polling
                await self._start_polling_monitoring(wallet_address)
                
        except Exception as e:
            logger.error(f"Error starting monitoring for {wallet_address}: {e}")
    
    async def _start_helius_websocket_monitoring(self, wallet_address: str):
        """Start WebSocket monitoring via Helius"""
        try:
            # Note: This would require Helius WebSocket API integration
            # For now, implement polling as fallback
            await self._start_polling_monitoring(wallet_address)
            
        except Exception as e:
            logger.error(f"WebSocket monitoring failed for {wallet_address}: {e}")
            await self._start_polling_monitoring(wallet_address)
    
    async def _start_polling_monitoring(self, wallet_address: str):
        """Start polling-based monitoring for a wallet"""
        try:
            while self.monitoring_active and wallet_address in self.tracked_wallets:
                # Get recent transactions
                transactions = await self._get_wallet_transactions(wallet_address, limit=10)
                
                # Process new transactions
                for tx in transactions:
                    await self._process_transaction(tx)
                
                # Update last seen
                self.tracked_wallets[wallet_address]['last_seen'] = time.time()
                
                # Wait before next poll (adjust based on your needs)
                await asyncio.sleep(5)  # 5 second polling interval
                
        except Exception as e:
            logger.error(f"Polling monitoring error for {wallet_address}: {e}")
    
    async def _get_wallet_transactions(self, wallet_address: str, limit: int = 10) -> List[WalletTransaction]:
        """Get recent transactions for a wallet"""
        try:
            transactions = []
            
            # Try Helius first
            if self.helius_service:
                helius_transactions = await self._get_helius_transactions(wallet_address, limit)
                transactions.extend(helius_transactions)
            
            # Try QuickNode as backup
            elif self.quicknode_service:
                quicknode_transactions = await self._get_quicknode_transactions(wallet_address, limit)
                transactions.extend(quicknode_transactions)
            
            return transactions
            
        except Exception as e:
            logger.error(f"Error getting transactions for {wallet_address}: {e}")
            return []
    
    async def _get_helius_transactions(self, wallet_address: str, limit: int) -> List[WalletTransaction]:
        """Get transactions via Helius API"""
        try:
            # This would integrate with your existing Helius service
            # For now, return empty list as placeholder
            # You'll need to implement the actual API calls
            
            transactions = []
            
            # Example structure - replace with actual Helius API calls
            # helius_data = await self.helius_service.get_wallet_transactions(wallet_address, limit)
            # 
            # for tx_data in helius_data:
            #     transaction = self._parse_helius_transaction(wallet_address, tx_data)
            #     if transaction:
            #         transactions.append(transaction)
            
            return transactions
            
        except Exception as e:
            logger.error(f"Helius transaction fetch error: {e}")
            return []
    
    async def _get_quicknode_transactions(self, wallet_address: str, limit: int) -> List[WalletTransaction]:
        """Get transactions via QuickNode API"""
        try:
            # This would integrate with your existing QuickNode service
            transactions = []
            
            # Example structure - replace with actual QuickNode API calls
            # quicknode_data = await self.quicknode_service.get_wallet_transactions(wallet_address, limit)
            # 
            # for tx_data in quicknode_data:
            #     transaction = self._parse_quicknode_transaction(wallet_address, tx_data)
            #     if transaction:
            #         transactions.append(transaction)
            
            return transactions
            
        except Exception as e:
            logger.error(f"QuickNode transaction fetch error: {e}")
            return []
    
    async def _process_transaction(self, transaction: WalletTransaction):
        """Process a new transaction from a tracked wallet"""
        try:
            # Check if we've already processed this transaction
            if self._is_duplicate_transaction(transaction):
                return
            
            # Add to transaction history
            self.transaction_history[transaction.wallet_address].append(transaction)
            self.recent_transactions.append(transaction)
            
            # Update wallet tracking data
            wallet_data = self.tracked_wallets.get(transaction.wallet_address)
            if wallet_data:
                wallet_data['total_tracked_trades'] += 1
                wallet_data['last_seen'] = transaction.timestamp
            
            # Generate trading signal if criteria met
            signal = await self._generate_trading_signal(transaction)
            if signal:
                self.active_signals[signal.signal_id] = signal
                logger.info(f"🎯 Generated signal: {signal.action.upper()} {signal.token_address[:8]}... "
                           f"(confidence: {signal.confidence:.3f}, urgency: {signal.urgency})")
            
            # Update performance metrics
            await self._update_wallet_performance(transaction.wallet_address)
            
        except Exception as e:
            logger.error(f"Error processing transaction: {e}")
    
    async def _generate_trading_signal(self, transaction: WalletTransaction) -> Optional[TradingSignal]:
        """Generate a trading signal based on wallet transaction"""
        try:
            wallet_address = transaction.wallet_address
            
            # Get wallet performance score
            performance = self.wallet_performances.get(wallet_address)
            if not performance or performance.score < self.config['min_wallet_score']:
                return None
            
            # Check transaction significance
            if transaction.amount_sol < self.config['min_transaction_sol']:
                return None
            
            # Calculate signal confidence
            confidence = self._calculate_signal_confidence(transaction, performance)
            
            # Determine urgency
            urgency = self._determine_signal_urgency(transaction, performance)
            
            # Generate reasoning
            reasoning = self._generate_signal_reasoning(transaction, performance)
            
            # Create signal
            signal = TradingSignal(
                signal_id=f"{wallet_address[:8]}_{transaction.signature[:8]}_{int(time.time())}",
                wallet_address=wallet_address,
                token_address=transaction.token_address,
                action=transaction.action,
                confidence=confidence,
                amount_sol=transaction.amount_sol,
                reasoning=reasoning,
                wallet_score=performance.score,
                timestamp=time.time(),
                urgency=urgency
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return None
    
    def _calculate_signal_confidence(self, transaction: WalletTransaction, performance: WalletPerformance) -> float:
        """Calculate confidence score for a trading signal"""
        try:
            base_confidence = 0.5
            
            # Performance-based adjustments
            performance_multiplier = min(performance.score / 100, 1.0)
            base_confidence *= performance_multiplier
            
            # Win rate adjustment
            if performance.win_rate > 0.7:
                base_confidence += 0.2
            elif performance.win_rate > 0.6:
                base_confidence += 0.1
            elif performance.win_rate < 0.4:
                base_confidence -= 0.2
            
            # Transaction size adjustment
            if transaction.amount_sol > 10:  # Large transaction
                base_confidence += 0.15
            elif transaction.amount_sol > 5:
                base_confidence += 0.1
            elif transaction.amount_sol < 1:
                base_confidence -= 0.1
            
            # Recent performance adjustment
            recent_performance = self._get_recent_performance(transaction.wallet_address)
            if recent_performance and recent_performance > 0.1:  # Recent profits
                base_confidence += 0.1
            elif recent_performance and recent_performance < -0.05:  # Recent losses
                base_confidence -= 0.1
            
            # Cap confidence between 0.1 and 0.95
            return max(0.1, min(0.95, base_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating signal confidence: {e}")
            return 0.5
    
    def _determine_signal_urgency(self, transaction: WalletTransaction, performance: WalletPerformance) -> str:
        """Determine the urgency level of a trading signal"""
        try:
            # High urgency conditions
            if (transaction.amount_sol > 10 and 
                performance.score > 85 and 
                performance.win_rate > 0.75):
                return 'critical'
            
            # Medium-high urgency
            elif (transaction.amount_sol > 5 and 
                  performance.score > 75 and 
                  performance.win_rate > 0.65):
                return 'high'
            
            # Medium urgency
            elif (transaction.amount_sol > 2 and 
                  performance.score > 65 and 
                  performance.win_rate > 0.55):
                return 'medium'
            
            # Low urgency
            else:
                return 'low'
                
        except Exception as e:
            logger.error(f"Error determining signal urgency: {e}")
            return 'low'
    
    def _generate_signal_reasoning(self, transaction: WalletTransaction, performance: WalletPerformance) -> str:
        """Generate human-readable reasoning for the signal"""
        try:
            reasons = []
            
            # Wallet performance
            reasons.append(f"Wallet score: {performance.score:.1f}/100")
            reasons.append(f"Win rate: {performance.win_rate:.1%}")
            
            # Transaction details
            reasons.append(f"Transaction size: {transaction.amount_sol:.2f} SOL")
            
            # Recent performance
            recent_perf = self._get_recent_performance(transaction.wallet_address)
            if recent_perf:
                if recent_perf > 0:
                    reasons.append(f"Recent profit: +{recent_perf:.2f} SOL")
                else:
                    reasons.append(f"Recent loss: {recent_perf:.2f} SOL")
            
            # Timing
            time_since_last = time.time() - performance.last_updated
            if time_since_last < 3600:  # Less than 1 hour
                reasons.append("Recent activity")
            
            return " | ".join(reasons)
            
        except Exception as e:
            logger.error(f"Error generating signal reasoning: {e}")
            return "Smart money activity detected"
    
    async def get_active_signals(self, min_confidence: float = 0.6) -> List[TradingSignal]:
        """Get current active trading signals"""
        try:
            current_time = time.time()
            active_signals = []
            
            # Filter signals by confidence and expiry
            for signal_id, signal in list(self.active_signals.items()):
                # Remove expired signals
                if current_time - signal.timestamp > self.config['signal_expiry_seconds']:
                    del self.active_signals[signal_id]
                    continue
                
                # Filter by confidence
                if signal.confidence >= min_confidence:
                    active_signals.append(signal)
            
            # Sort by confidence and urgency
            active_signals.sort(
                key=lambda s: (
                    {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}[s.urgency],
                    s.confidence
                ),
                reverse=True
            )
            
            return active_signals
            
        except Exception as e:
            logger.error(f"Error getting active signals: {e}")
            return []
    
    def get_wallet_performance(self, wallet_address: str) -> Optional[WalletPerformance]:
        """Get performance metrics for a specific wallet"""
        return self.wallet_performances.get(wallet_address)
    
    def get_all_wallet_performances(self) -> Dict[str, WalletPerformance]:
        """Get performance metrics for all tracked wallets"""
        return self.wallet_performances.copy()
    
    def get_top_performing_wallets(self, limit: int = 10) -> List[WalletPerformance]:
        """Get top performing wallets by score"""
        try:
            all_performances = list(self.wallet_performances.values())
            all_performances.sort(key=lambda p: p.score, reverse=True)
            return all_performances[:limit]
            
        except Exception as e:
            logger.error(f"Error getting top performing wallets: {e}")
            return []
    
    async def _initialize_wallet_performance(self, wallet_address: str):
        """Initialize performance tracking for a new wallet"""
        try:
            # Get historical transactions to calculate initial performance
            historical_transactions = await self._get_wallet_transactions(wallet_address, limit=100)
            
            # Calculate initial performance metrics
            performance = self._calculate_wallet_performance(wallet_address, historical_transactions)
            self.wallet_performances[wallet_address] = performance
            
            logger.debug(f"Initialized performance for {wallet_address[:8]}...: score {performance.score:.1f}")
            
        except Exception as e:
            logger.error(f"Error initializing wallet performance: {e}")
            # Create default performance if calculation fails
            self.wallet_performances[wallet_address] = WalletPerformance(
                wallet_address=wallet_address,
                total_trades=0,
                winning_trades=0,
                total_profit_sol=0.0,
                win_rate=0.0,
                avg_profit_per_trade=0.0,
                avg_hold_time_hours=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                sharpe_ratio=0.0,
                last_updated=time.time(),
                score=50.0  # Default neutral score
            )
    
    def _calculate_wallet_performance(self, wallet_address: str, transactions: List[WalletTransaction]) -> WalletPerformance:
        """Calculate comprehensive performance metrics for a wallet"""
        try:
            if not transactions:
                return WalletPerformance(
                    wallet_address=wallet_address,
                    total_trades=0,
                    winning_trades=0,
                    total_profit_sol=0.0,
                    win_rate=0.0,
                    avg_profit_per_trade=0.0,
                    avg_hold_time_hours=0.0,
                    largest_win=0.0,
                    largest_loss=0.0,
                    sharpe_ratio=0.0,
                    last_updated=time.time(),
                    score=50.0
                )
            
            # Analyze trading pairs (buy -> sell sequences)
            trading_pairs = self._analyze_trading_pairs(transactions)
            
            total_trades = len(trading_pairs)
            winning_trades = sum(1 for pair in trading_pairs if pair['profit'] > 0)
            total_profit = sum(pair['profit'] for pair in trading_pairs)
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            avg_profit = total_profit / total_trades if total_trades > 0 else 0
            
            profits = [pair['profit'] for pair in trading_pairs]
            largest_win = max(profits) if profits else 0
            largest_loss = min(profits) if profits else 0
            
            # Calculate average hold time
            hold_times = [pair['hold_time_hours'] for pair in trading_pairs if pair['hold_time_hours'] > 0]
            avg_hold_time = sum(hold_times) / len(hold_times) if hold_times else 0
            
            # Calculate Sharpe ratio (simplified)
            if profits and len(profits) > 1:
                profit_std = (sum((p - avg_profit) ** 2 for p in profits) / len(profits)) ** 0.5
                sharpe_ratio = avg_profit / profit_std if profit_std > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Calculate overall score (0-100)
            score = self._calculate_performance_score(
                win_rate, avg_profit, total_trades, sharpe_ratio, total_profit
            )
            
            return WalletPerformance(
                wallet_address=wallet_address,
                total_trades=total_trades,
                winning_trades=winning_trades,
                total_profit_sol=total_profit,
                win_rate=win_rate,
                avg_profit_per_trade=avg_profit,
                avg_hold_time_hours=avg_hold_time,
                largest_win=largest_win,
                largest_loss=largest_loss,
                sharpe_ratio=sharpe_ratio,
                last_updated=time.time(),
                score=score
            )
            
        except Exception as e:
            logger.error(f"Error calculating wallet performance: {e}")
            return WalletPerformance(
                wallet_address=wallet_address,
                total_trades=0,
                winning_trades=0,
                total_profit_sol=0.0,
                win_rate=0.0,
                avg_profit_per_trade=0.0,
                avg_hold_time_hours=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                sharpe_ratio=0.0,
                last_updated=time.time(),
                score=50.0
            )
    
    def _analyze_trading_pairs(self, transactions: List[WalletTransaction]) -> List[Dict]:
        """Analyze buy->sell trading pairs to calculate profits"""
        try:
            trading_pairs = []
            
            # Group transactions by token
            token_transactions = defaultdict(list)
            for tx in transactions:
                token_transactions[tx.token_address].append(tx)
            
            # Analyze each token's trading history
            for token_address, token_txs in token_transactions.items():
                # Sort by timestamp
                token_txs.sort(key=lambda x: x.timestamp)
                
                # Track positions
                position = 0
                avg_buy_price = 0
                buy_timestamp = 0
                
                for tx in token_txs:
                    if tx.action == 'buy':
                        if position == 0:
                            # New position
                            position = tx.token_amount
                            avg_buy_price = tx.price
                            buy_timestamp = tx.timestamp
                        else:
                            # Add to position (weighted average)
                            total_value = (position * avg_buy_price) + (tx.token_amount * tx.price)
                            position += tx.token_amount
                            avg_buy_price = total_value / position if position > 0 else 0
                    
                    elif tx.action == 'sell' and position > 0:
                        # Calculate profit for sold amount
                        sold_amount = min(tx.token_amount, position)
                        profit_per_token = tx.price - avg_buy_price
                        total_profit = profit_per_token * sold_amount
                        
                        # Convert to SOL profit (simplified)
                        profit_sol = total_profit * tx.price  # Approximate SOL value
                        
                        # Calculate hold time
                        hold_time_hours = (tx.timestamp - buy_timestamp) / 3600 if buy_timestamp > 0 else 0
                        
                        trading_pairs.append({
                            'token_address': token_address,
                            'buy_price': avg_buy_price,
                            'sell_price': tx.price,
                            'amount': sold_amount,
                            'profit': profit_sol,
                            'profit_percent': (profit_per_token / avg_buy_price * 100) if avg_buy_price > 0 else 0,
                            'hold_time_hours': hold_time_hours,
                            'buy_timestamp': buy_timestamp,
                            'sell_timestamp': tx.timestamp
                        })
                        
                        # Update position
                        position -= sold_amount
                        if position <= 0:
                            position = 0
                            avg_buy_price = 0
                            buy_timestamp = 0
            
            return trading_pairs
            
        except Exception as e:
            logger.error(f"Error analyzing trading pairs: {e}")
            return []
    
    def _calculate_performance_score(self, win_rate: float, avg_profit: float, 
                                   total_trades: int, sharpe_ratio: float, total_profit: float) -> float:
        """Calculate overall performance score (0-100)"""
        try:
            score = 50.0  # Base score
            
            # Win rate component (0-30 points)
            score += (win_rate - 0.5) * 60  # +30 for 100% win rate, -30 for 0%
            
            # Average profit component (0-25 points)
            if avg_profit > 0:
                score += min(25, avg_profit * 5)  # +25 for 5+ SOL avg profit
            else:
                score += max(-25, avg_profit * 10)  # -25 for -2.5 SOL avg loss
            
            # Trade volume component (0-15 points)
            if total_trades >= 50:
                score += 15
            elif total_trades >= 20:
                score += 10
            elif total_trades >= 10:
                score += 5
            elif total_trades < 5:
                score -= 10
            
            # Sharpe ratio component (0-15 points)
            if sharpe_ratio > 2:
                score += 15
            elif sharpe_ratio > 1:
                score += 10
            elif sharpe_ratio > 0.5:
                score += 5
            elif sharpe_ratio < 0:
                score -= 10
            
            # Total profit component (0-15 points)
            if total_profit > 50:
                score += 15
            elif total_profit > 20:
                score += 10
            elif total_profit > 5:
                score += 5
            elif total_profit < -10:
                score -= 15
            
            # Cap score between 0 and 100
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 50.0
    
    # Helper methods and monitoring loops...
    
    def _is_valid_solana_address(self, address: str) -> bool:
        """Validate Solana wallet address format"""
        try:
            # Basic validation - Solana addresses are base58 encoded, 32-44 chars
            if not address or len(address) < 32 or len(address) > 44:
                return False
            
            # Additional validation could be added here
            return True
            
        except Exception:
            return False
    
    def _is_duplicate_transaction(self, transaction: WalletTransaction) -> bool:
        """Check if transaction has already been processed"""
        try:
            # Check recent transactions for duplicates
            for recent_tx in self.recent_transactions:
                if recent_tx.signature == transaction.signature:
                    return True
            return False
            
        except Exception as e:
            logger.error(f"Error checking duplicate transaction: {e}")
            return False
    
    def _get_recent_performance(self, wallet_address: str, days: int = 7) -> Optional[float]:
        """Get recent performance for a wallet"""
        try:
            cutoff_time = time.time() - (days * 24 * 3600)
            recent_transactions = [
                tx for tx in self.transaction_history.get(wallet_address, [])
                if tx.timestamp > cutoff_time
            ]
            
            if not recent_transactions:
                return None
            
            # Calculate profit from recent trading pairs
            trading_pairs = self._analyze_trading_pairs(recent_transactions)
            return sum(pair['profit'] for pair in trading_pairs)
            
        except Exception as e:
            logger.error(f"Error getting recent performance: {e}")
            return None
    
    async def _process_signals_loop(self):
        """Background loop to process and clean up signals"""
        try:
            while self.monitoring_active:
                current_time = time.time()
                
                # Remove expired signals
                expired_signals = [
                    signal_id for signal_id, signal in self.active_signals.items()
                    if current_time - signal.timestamp > self.config['signal_expiry_seconds']
                ]
                
                for signal_id in expired_signals:
                    del self.active_signals[signal_id]
                
                if expired_signals:
                    logger.debug(f"Removed {len(expired_signals)} expired signals")
                
                try:
                    await asyncio.sleep(60)  # Check every minute
                except asyncio.CancelledError:
                    break
                
        except asyncio.CancelledError:
            logger.info("Signal processing loop cancelled")
        except Exception as e:
            if self.monitoring_active:
                logger.error(f"Signal processing loop error: {e}")

    async def _update_performances_loop(self):
        """Background loop to update wallet performances"""
        try:
            while self.monitoring_active:
                # Update performances every 10 minutes
                try:
                    await asyncio.sleep(600)
                except asyncio.CancelledError:
                    break
                
                if not self.monitoring_active:
                    break
                
                for wallet_address in self.tracked_wallets.keys():
                    await self._update_wallet_performance(wallet_address)
                
        except asyncio.CancelledError:
            logger.info("Performance update loop cancelled")
        except Exception as e:
            if self.monitoring_active:
                logger.error(f"Performance update loop error: {e}")
    
    async def _update_wallet_performance(self, wallet_address: str):
        """Update performance metrics for a specific wallet"""
        try:
            # Get recent transactions
            transactions = self.transaction_history.get(wallet_address, [])
            
            # Recalculate performance
            performance = self._calculate_wallet_performance(wallet_address, transactions)
            self.wallet_performances[wallet_address] = performance
            
        except Exception as e:
            logger.error(f"Error updating wallet performance: {e}")
    
    async def stop_monitoring(self):
        """Stop monitoring all wallets"""
        try:
            logger.info("🛑 Stopping smart money monitoring...")
            self.monitoring_active = False
            
            # Close WebSocket connections
            for connection in self.websocket_connections.values():
                if hasattr(connection, 'close'):
                    await connection.close()
            
            logger.info("✅ Smart money monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status"""
        try:
            return {
                'monitoring_active': self.monitoring_active,
                'tracked_wallets_count': len(self.tracked_wallets),
                'active_signals_count': len(self.active_signals),
                'total_transactions_processed': len(self.recent_transactions),
                'average_wallet_score': sum(p.score for p in self.wallet_performances.values()) / len(self.wallet_performances) if self.wallet_performances else 0,
                'top_wallet_score': max(p.score for p in self.wallet_performances.values()) if self.wallet_performances else 0,
                'config': self.config
            }
            
        except Exception as e:
            logger.error(f"Error getting monitoring status: {e}")
            return {'error': str(e)}

# Convenience function to create tracker instance
def create_smart_money_tracker(helius_service=None, quicknode_service=None) -> SmartMoneyTracker:
    """Create and return a Smart Money Tracker instance"""
    return SmartMoneyTracker(helius_service, quicknode_service) 