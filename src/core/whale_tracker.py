import json
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from ..utils.logging_config import (
    whale_logger, handle_errors, log_performance,
    log_async_performance, NetworkError
)
from ..utils.config import settings
from solana.publickey import PublicKey

class WhaleTracker:
    def __init__(self):
        self.whales = self._load_whale_data()
        self.whale_activities = {}  # Cache for whale activities
        self.whale_stats = {}  # Statistics for each whale
        self.last_analysis = 0
        whale_logger.info("WhaleTracker initialized", extra={
            'total_whales': len(self.whales),
            'active_alerts': sum(1 for w in self.whales if w['alertsOn'])
        })

    def _load_whale_data(self) -> List[Dict]:
        """Load whale data from configuration"""
        try:
            with open('config/whales.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            whale_logger.error(f"Error loading whale data: {str(e)}")
            return []

    @handle_errors(whale_logger)
    @log_performance(whale_logger)
    async def track_whale_activity(self, wallet_address: str) -> Dict:
        """Track activity of a specific whale wallet"""
        try:
            # Get recent transactions
            transactions = await self._get_wallet_transactions(wallet_address)
            
            # Analyze transactions
            analysis = self._analyze_transactions(transactions)
            
            # Update whale stats
            self._update_whale_stats(wallet_address, analysis)
            
            whale_logger.info(f"Tracked whale activity", extra={
                'wallet': wallet_address,
                'analysis': analysis
            })
            
            return analysis

        except Exception as e:
            error_msg = f"Error tracking whale activity: {str(e)}"
            whale_logger.error(error_msg)
            raise NetworkError(error_msg)

    @handle_errors(whale_logger)
    @log_performance(whale_logger)
    async def analyze_whale_patterns(self) -> Dict:
        """Analyze patterns across all tracked whales"""
        try:
            current_time = asyncio.get_event_loop().time()
            if current_time - self.last_analysis < settings.WHALE_ANALYSIS_INTERVAL:
                return self.whale_activities

            patterns = {
                'active_whales': [],
                'profitable_trades': [],
                'common_tokens': {},
                'success_rate': {},
                'avg_hold_time': {},
                'risk_levels': {}
            }

            for whale in self.whales:
                if not whale['alertsOn']:
                    continue

                activity = await self.track_whale_activity(whale['trackedWalletAddress'])
                
                if activity['is_active']:
                    patterns['active_whales'].append({
                        'address': whale['trackedWalletAddress'],
                        'name': whale['name'],
                        'emoji': whale['emoji'],
                        'activity': activity
                    })

                # Update patterns
                self._update_patterns(patterns, whale, activity)

            self.whale_activities = patterns
            self.last_analysis = current_time

            whale_logger.info("Whale patterns analyzed", extra={'patterns': patterns})
            return patterns

        except Exception as e:
            error_msg = f"Error analyzing whale patterns: {str(e)}"
            whale_logger.error(error_msg)
            raise NetworkError(error_msg)

    @handle_errors(whale_logger)
    @log_performance(whale_logger)
    async def get_trading_signals(self) -> List[Dict]:
        """Get trading signals based on whale activities"""
        try:
            patterns = await self.analyze_whale_patterns()
            signals = []

            for whale in patterns['active_whales']:
                activity = whale['activity']
                
                if self._is_strong_signal(activity):
                    signal = {
                        'type': 'whale_activity',
                        'whale': whale['name'],
                        'emoji': whale['emoji'],
                        'confidence': self._calculate_confidence(activity),
                        'token': activity['token'],
                        'action': activity['action'],
                        'timestamp': datetime.now().isoformat(),
                        'details': {
                            'success_rate': self.whale_stats[whale['address']]['success_rate'],
                            'avg_profit': self.whale_stats[whale['address']]['avg_profit'],
                            'hold_time': self.whale_stats[whale['address']]['avg_hold_time']
                        }
                    }
                    signals.append(signal)
                    whale_logger.info(f"Trading signal generated", extra={'signal': signal})

            return signals

        except Exception as e:
            error_msg = f"Error getting trading signals: {str(e)}"
            whale_logger.error(error_msg)
            raise NetworkError(error_msg)

    @handle_errors(whale_logger)
    async def _get_wallet_transactions(self, wallet_address: str) -> List[Dict]:
        """Get recent transactions for a wallet"""
        try:
            # Get recent transactions from Solana RPC
            response = await self.client.get_signatures_for_address(
                PublicKey(wallet_address),
                limit=100  # Get last 100 transactions
            )
            
            if not response["result"]:
                return []
                
            transactions = []
            for tx in response["result"]:
                # Get transaction details
                tx_details = await self.client.get_transaction(
                    tx["signature"],
                    commitment="confirmed"
                )
                
                if not tx_details["result"]:
                    continue
                    
                # Extract relevant information
                tx_data = tx_details["result"]
                transaction = {
                    "signature": tx["signature"],
                    "timestamp": tx["blockTime"],
                    "token": None,
                    "action": None,
                    "amount": 0
                }
                
                # Parse transaction instructions
                for instruction in tx_data["transaction"]["message"]["instructions"]:
                    if instruction["programId"] == "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA":
                        # Token program instruction
                        if instruction["data"][0] == 3:  # Transfer instruction
                            transaction["action"] = "transfer"
                            transaction["amount"] = int.from_bytes(
                                bytes.fromhex(instruction["data"][8:24]),
                                "little"
                            ) / 1e9  # Convert to SOL
                            
                    elif instruction["programId"] == "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8":
                        # Raydium program instruction
                        if instruction["data"][0] == 1:  # Swap instruction
                            transaction["action"] = "swap"
                            transaction["amount"] = int.from_bytes(
                                bytes.fromhex(instruction["data"][1:9]),
                                "little"
                            ) / 1e9  # Convert to SOL
                            
                if transaction["action"]:
                    transactions.append(transaction)
                    
            return transactions
            
        except Exception as e:
            error_msg = f"Error getting wallet transactions: {str(e)}"
            whale_logger.error(error_msg)
            raise NetworkError(error_msg)

    @handle_errors(whale_logger)
    def _analyze_transactions(self, transactions: List[Dict]) -> Dict:
        """Analyze transactions to determine whale activity"""
        try:
            if not transactions:
                return {
                    'is_active': False,
                    'last_activity': None,
                    'token': None,
                    'action': None,
                    'amount': 0,
                    'confidence': 0
                }

            # Analyze the most recent transaction
            latest_tx = transactions[0]
            
            analysis = {
                'is_active': True,
                'last_activity': latest_tx['timestamp'],
                'token': latest_tx['token'],
                'action': latest_tx['action'],
                'amount': latest_tx['amount'],
                'confidence': self._calculate_confidence(latest_tx)
            }

            whale_logger.debug("Transactions analyzed", extra={'analysis': analysis})
            return analysis

        except Exception as e:
            error_msg = f"Error analyzing transactions: {str(e)}"
            whale_logger.error(error_msg)
            raise NetworkError(error_msg)

    @handle_errors(whale_logger)
    def _update_whale_stats(self, wallet_address: str, analysis: Dict) -> None:
        """Update statistics for a whale"""
        try:
            if wallet_address not in self.whale_stats:
                self.whale_stats[wallet_address] = {
                    'success_rate': 0,
                    'avg_profit': 0,
                    'avg_hold_time': 0,
                    'total_trades': 0,
                    'profitable_trades': 0
                }

            stats = self.whale_stats[wallet_address]
            
            # Update stats based on analysis
            if analysis['is_active']:
                stats['total_trades'] += 1
                if analysis['confidence'] > 0.7:  # High confidence trade
                    stats['profitable_trades'] += 1
                
                stats['success_rate'] = (stats['profitable_trades'] / stats['total_trades']) * 100

            whale_logger.debug("Whale stats updated", extra={
                'wallet': wallet_address,
                'stats': stats
            })

        except Exception as e:
            error_msg = f"Error updating whale stats: {str(e)}"
            whale_logger.error(error_msg)
            raise NetworkError(error_msg)

    @handle_errors(whale_logger)
    def _update_patterns(self, patterns: Dict, whale: Dict, activity: Dict) -> None:
        """Update trading patterns based on whale activity"""
        try:
            if not activity['is_active']:
                return

            # Update common tokens
            token = activity['token']
            if token:
                patterns['common_tokens'][token] = patterns['common_tokens'].get(token, 0) + 1

            # Update success rate
            wallet = whale['trackedWalletAddress']
            if wallet in self.whale_stats:
                patterns['success_rate'][wallet] = self.whale_stats[wallet]['success_rate']

            # Update risk levels
            patterns['risk_levels'][wallet] = self._calculate_risk_level(activity)

            whale_logger.debug("Patterns updated", extra={
                'wallet': wallet,
                'patterns': patterns
            })

        except Exception as e:
            error_msg = f"Error updating patterns: {str(e)}"
            whale_logger.error(error_msg)
            raise NetworkError(error_msg)

    @handle_errors(whale_logger)
    def _is_strong_signal(self, activity: Dict) -> bool:
        """Determine if an activity represents a strong trading signal"""
        try:
            if not activity['is_active']:
                return False

            # Check confidence threshold
            if activity['confidence'] < settings.WHALE_CONFIDENCE_THRESHOLD:
                return False

            # Check if whale has good track record
            wallet = activity['wallet']
            if wallet in self.whale_stats:
                stats = self.whale_stats[wallet]
                if stats['success_rate'] < settings.WHALE_SUCCESS_RATE_THRESHOLD:
                    return False

            return True

        except Exception as e:
            error_msg = f"Error checking signal strength: {str(e)}"
            whale_logger.error(error_msg)
            raise NetworkError(error_msg)

    @handle_errors(whale_logger)
    def _calculate_confidence(self, activity: Dict) -> float:
        """Calculate confidence score for a whale activity"""
        try:
            confidence = 0.0
            
            # Base confidence on whale's success rate
            if activity['wallet'] in self.whale_stats:
                stats = self.whale_stats[activity['wallet']]
                confidence += stats['success_rate'] / 100

            # Adjust for trade size
            if activity['amount'] > settings.WHALE_LARGE_TRADE_THRESHOLD:
                confidence += 0.2

            # Adjust for recent activity
            if activity['last_activity']:
                time_diff = datetime.now() - datetime.fromisoformat(activity['last_activity'])
                if time_diff < timedelta(hours=1):
                    confidence += 0.1

            return min(confidence, 1.0)

        except Exception as e:
            error_msg = f"Error calculating confidence: {str(e)}"
            whale_logger.error(error_msg)
            raise NetworkError(error_msg)

    @handle_errors(whale_logger)
    def _calculate_risk_level(self, activity: Dict) -> str:
        """Calculate risk level for a whale activity"""
        try:
            if activity['wallet'] not in self.whale_stats:
                return 'unknown'

            stats = self.whale_stats[activity['wallet']]
            
            if stats['success_rate'] >= 80:
                return 'low'
            elif stats['success_rate'] >= 60:
                return 'medium'
            else:
                return 'high'

        except Exception as e:
            error_msg = f"Error calculating risk level: {str(e)}"
            whale_logger.error(error_msg)
            raise NetworkError(error_msg) 