"""
Transaction Warfare System - Battle-Hardened Transaction Execution

This module implements network-resistant transaction execution:
- Priority fee auto-boost (up to 0.1 SOL)
- Multi-RPC transaction broadcasting  
- Pre-signed cancellation orders
- Block height expiration triggers
- State tracking: TransactionStateMachine(status, retries, fallback_path)

SURVIVAL PROTOCOL: Execute transactions under extreme network conditions
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from collections import deque, defaultdict
import base58
import struct

logger = logging.getLogger(__name__)

class TransactionStatus(Enum):
    """Transaction execution states"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    ESCALATING = "escalating"

class NetworkCondition(Enum):
    """Network congestion levels"""
    CLEAR = "clear"         # <50% slot utilization
    BUSY = "busy"          # 50-70% slot utilization  
    CONGESTED = "congested" # 70-85% slot utilization
    JAMMED = "jammed"      # 85-95% slot utilization
    APOCALYPTIC = "apocalyptic"  # >95% slot utilization

class FailureReason(Enum):
    """Transaction failure types"""
    INSUFFICIENT_FUNDS = "insufficient_funds"
    NETWORK_TIMEOUT = "network_timeout"
    RPC_ERROR = "rpc_error"
    SLIPPAGE_EXCEEDED = "slippage_exceeded"
    BLOCKHASH_EXPIRED = "blockhash_expired"
    COMPUTE_LIMIT = "compute_limit"
    MEV_ATTACK = "mev_attack"
    PRIORITY_TOO_LOW = "priority_too_low"

@dataclass
class TransactionMetrics:
    """Real-time transaction execution metrics"""
    average_confirmation_time: float
    success_rate: float
    network_condition: NetworkCondition
    recommended_priority_fee: int
    rpc_latencies: Dict[str, float]
    slot_utilization: float
    mev_activity_level: float
    congestion_score: float
    measurement_time: float = field(default_factory=time.time)

@dataclass
class TransactionState:
    """Individual transaction state tracking"""
    transaction_id: str
    instruction_type: str  # "buy", "sell", "cancel"
    token_address: str
    amount: float
    
    # State tracking
    status: TransactionStatus
    submission_attempts: int = 0
    max_retries: int = 5
    
    # Network details
    signature: Optional[str] = None
    blockhash: Optional[str] = None
    block_height: Optional[int] = None
    priority_fee: int = 10000  # lamports
    
    # Timing
    created_at: float = field(default_factory=time.time)
    submitted_at: Optional[float] = None
    confirmed_at: Optional[float] = None
    deadline: float = field(default_factory=lambda: time.time() + 30)
    
    # Failure tracking
    failure_reasons: List[FailureReason] = field(default_factory=list)
    rpc_responses: List[Dict] = field(default_factory=list)
    
    # Escalation tracking
    original_priority_fee: int = 10000
    max_priority_fee: int = 100000  # 0.1 SOL max
    escalation_count: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if transaction has expired"""
        return time.time() > self.deadline
    
    @property
    def can_retry(self) -> bool:
        """Check if transaction can be retried"""
        return (self.submission_attempts < self.max_retries and 
                not self.is_expired and
                self.status not in [TransactionStatus.CONFIRMED, TransactionStatus.CANCELLED])
    
    @property
    def should_escalate(self) -> bool:
        """Check if priority fee should be escalated"""
        return (self.submission_attempts >= 2 and 
                self.priority_fee < self.max_priority_fee and
                time.time() - self.created_at > 10)  # 10 seconds before escalation

class RPCEndpoint:
    """RPC endpoint wrapper with health monitoring"""
    
    def __init__(self, url: str, name: str, priority: int = 1):
        self.url = url
        self.name = name
        self.priority = priority  # 1=highest, 5=lowest
        
        # Health metrics
        self.success_count = 0
        self.failure_count = 0
        self.total_requests = 0
        self.average_latency = 0.0
        self.last_success = 0.0
        self.last_failure = 0.0
        self.consecutive_failures = 0
        
        # Performance tracking
        self.latency_history = deque(maxlen=50)
        self.is_healthy = True
        
    def record_success(self, latency: float):
        """Record successful request"""
        self.success_count += 1
        self.total_requests += 1
        self.last_success = time.time()
        self.consecutive_failures = 0
        
        self.latency_history.append(latency)
        if self.latency_history:
            self.average_latency = sum(self.latency_history) / len(self.latency_history)
        
        # Update health status
        if self.success_count > 0:
            success_rate = self.success_count / self.total_requests
            self.is_healthy = success_rate > 0.8 and self.consecutive_failures < 3
    
    def record_failure(self, error: str):
        """Record failed request"""
        self.failure_count += 1
        self.total_requests += 1
        self.last_failure = time.time()
        self.consecutive_failures += 1
        
        # Update health status
        if self.consecutive_failures >= 5 or (self.total_requests > 10 and self.success_count / self.total_requests < 0.5):
            self.is_healthy = False
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 1.0
        return self.success_count / self.total_requests
    
    @property
    def health_score(self) -> float:
        """Calculate overall health score"""
        if not self.is_healthy:
            return 0.0
        
        success_weight = self.success_rate * 0.6
        latency_weight = max(0, (1000 - self.average_latency) / 1000) * 0.3  # Lower is better
        recency_weight = max(0, 1 - (time.time() - self.last_success) / 300) * 0.1  # Recent success
        
        return success_weight + latency_weight + recency_weight

class NetworkMonitor:
    """Monitors network conditions and congestion"""
    
    def __init__(self):
        self.slot_history = deque(maxlen=100)
        self.transaction_history = deque(maxlen=500)
        self.mev_activity = deque(maxlen=200)
        
    def add_slot_data(self, slot: int, transactions: int, compute_units: int, timestamp: float = None):
        """Add slot utilization data"""
        if timestamp is None:
            timestamp = time.time()
            
        # Estimate slot utilization (simplified)
        max_transactions_per_slot = 1500  # Approximate Solana limit
        max_compute_per_slot = 48_000_000  # Approximate compute limit
        
        tx_utilization = min(1.0, transactions / max_transactions_per_slot)
        compute_utilization = min(1.0, compute_units / max_compute_per_slot)
        
        slot_utilization = max(tx_utilization, compute_utilization)
        
        self.slot_history.append({
            'slot': slot,
            'utilization': slot_utilization,
            'transactions': transactions,
            'compute_units': compute_units,
            'timestamp': timestamp
        })
    
    def add_transaction_result(self, success: bool, confirmation_time: float, priority_fee: int):
        """Add transaction execution result"""
        self.transaction_history.append({
            'success': success,
            'confirmation_time': confirmation_time,
            'priority_fee': priority_fee,
            'timestamp': time.time()
        })
    
    def detect_mev_activity(self, failed_transactions: int, sandwich_attacks: int):
        """Detect MEV activity patterns"""
        mev_score = min(1.0, (failed_transactions * 0.1 + sandwich_attacks * 0.3))
        
        self.mev_activity.append({
            'mev_score': mev_score,
            'failed_txs': failed_transactions,
            'sandwich_attacks': sandwich_attacks,
            'timestamp': time.time()
        })
    
    def get_network_metrics(self) -> TransactionMetrics:
        """Calculate current network metrics"""
        try:
            current_time = time.time()
            
            # Calculate average confirmation time
            recent_transactions = [
                tx for tx in self.transaction_history 
                if current_time - tx['timestamp'] <= 300 and tx['success']  # Last 5 minutes
            ]
            
            if recent_transactions:
                avg_confirmation = sum(tx['confirmation_time'] for tx in recent_transactions) / len(recent_transactions)
                success_rate = len([tx for tx in self.transaction_history if tx['success']]) / max(1, len(self.transaction_history))
            else:
                avg_confirmation = 30.0  # Default assumption
                success_rate = 0.5
            
            # Calculate current slot utilization
            recent_slots = [
                slot for slot in self.slot_history
                if current_time - slot['timestamp'] <= 60  # Last minute
            ]
            
            if recent_slots:
                current_utilization = sum(slot['utilization'] for slot in recent_slots) / len(recent_slots)
            else:
                current_utilization = 0.5  # Default assumption
            
            # Determine network condition
            if current_utilization < 0.5:
                condition = NetworkCondition.CLEAR
            elif current_utilization < 0.7:
                condition = NetworkCondition.BUSY
            elif current_utilization < 0.85:
                condition = NetworkCondition.CONGESTED
            elif current_utilization < 0.95:
                condition = NetworkCondition.JAMMED
            else:
                condition = NetworkCondition.APOCALYPTIC
            
            # Calculate recommended priority fee
            base_fee = 10000  # 0.00001 SOL
            if condition == NetworkCondition.CLEAR:
                recommended_fee = base_fee
            elif condition == NetworkCondition.BUSY:
                recommended_fee = base_fee * 2
            elif condition == NetworkCondition.CONGESTED:
                recommended_fee = base_fee * 5
            elif condition == NetworkCondition.JAMMED:
                recommended_fee = base_fee * 10
            else:  # APOCALYPTIC
                recommended_fee = base_fee * 20
            
            # Calculate MEV activity level
            recent_mev = [
                mev for mev in self.mev_activity
                if current_time - mev['timestamp'] <= 300  # Last 5 minutes
            ]
            
            if recent_mev:
                mev_level = sum(mev['mev_score'] for mev in recent_mev) / len(recent_mev)
            else:
                mev_level = 0.1  # Low default
            
            # Calculate congestion score
            congestion_score = min(100, current_utilization * 100 + mev_level * 20)
            
            return TransactionMetrics(
                average_confirmation_time=avg_confirmation,
                success_rate=success_rate,
                network_condition=condition,
                recommended_priority_fee=int(recommended_fee),
                rpc_latencies={},  # Will be filled by RPC manager
                slot_utilization=current_utilization,
                mev_activity_level=mev_level,
                congestion_score=congestion_score
            )
            
        except Exception as e:
            logger.error(f"Error calculating network metrics: {str(e)}")
            return TransactionMetrics(
                average_confirmation_time=30.0,
                success_rate=0.5,
                network_condition=NetworkCondition.BUSY,
                recommended_priority_fee=20000,
                rpc_latencies={},
                slot_utilization=0.5,
                mev_activity_level=0.1,
                congestion_score=50.0
            )

class TransactionWarfareSystem:
    """Main warfare system coordinating all transaction execution"""
    
    def __init__(self):
        # RPC endpoint management
        self.rpc_endpoints = []
        self.primary_rpc = None
        self.fallback_rpcs = []
        
        # Network monitoring
        self.network_monitor = NetworkMonitor()
        
        # Active transaction tracking
        self.active_transactions: Dict[str, TransactionState] = {}
        self.transaction_history = deque(maxlen=1000)
        
        # Cancellation orders
        self.cancellation_orders: Dict[str, Dict] = {}
        
        # Performance tracking
        self.total_transactions = 0
        self.successful_transactions = 0
        self.failed_transactions = 0
        self.escalations_performed = 0
        
        # Configuration
        self.max_priority_fee_lamports = 100000  # 0.1 SOL
        self.default_timeout_seconds = 30
        self.max_retries = 5
        self.escalation_threshold_seconds = 10
        
        logger.info("⚔️ Transaction Warfare System initialized - Battle-ready")
    
    async def initialize(self):
        """Initialize the defense system (compatibility method)"""
        # Defense system is ready to use after __init__
        return True

    def add_rpc_endpoint(self, url: str, name: str, priority: int = 1, is_primary: bool = False):
        """Add RPC endpoint to the warfare arsenal"""
        endpoint = RPCEndpoint(url, name, priority)
        self.rpc_endpoints.append(endpoint)
        
        if is_primary or not self.primary_rpc:
            self.primary_rpc = endpoint
        else:
            self.fallback_rpcs.append(endpoint)
        
        # Sort fallback RPCs by priority
        self.fallback_rpcs.sort(key=lambda x: x.priority)
        
        logger.info(f"⚔️ Added RPC endpoint: {name} (Priority: {priority}, Primary: {is_primary})")
    
    async def execute_transaction(self, transaction_data: Dict, 
                                 instruction_type: str,
                                 token_address: str,
                                 amount: float,
                                 max_retries: int = None) -> TransactionState:
        """Execute transaction with full warfare protocols"""
        
        self.total_transactions += 1
        start_time = time.time()
        
        # Create transaction state
        tx_id = self._generate_transaction_id(transaction_data, instruction_type)
        
        tx_state = TransactionState(
            transaction_id=tx_id,
            instruction_type=instruction_type,
            token_address=token_address,
            amount=amount,
            status=TransactionStatus.PENDING,
            max_retries=max_retries or self.max_retries,
            deadline=time.time() + self.default_timeout_seconds
        )
        
        self.active_transactions[tx_id] = tx_state
        
        logger.info(f"⚔️ TRANSACTION WARFARE: Executing {instruction_type} for {amount} {token_address[:8]}... (ID: {tx_id[:8]})")
        
        try:
            # Get current network conditions
            network_metrics = self.network_monitor.get_network_metrics()
            
            # Set initial priority fee based on network conditions
            tx_state.priority_fee = max(
                network_metrics.recommended_priority_fee,
                tx_state.original_priority_fee
            )
            
            logger.debug(f"🎯 Network condition: {network_metrics.network_condition.value}, "
                        f"Recommended fee: {network_metrics.recommended_priority_fee}")
            
            # Prepare cancellation order
            await self._prepare_cancellation_order(tx_state)
            
            # Execute transaction with warfare protocols
            while tx_state.can_retry:
                tx_state.submission_attempts += 1
                tx_state.status = TransactionStatus.SUBMITTED
                
                logger.debug(f"⚔️ Attempt {tx_state.submission_attempts}/{tx_state.max_retries} "
                           f"(Priority: {tx_state.priority_fee} lamports)")
                
                # Try multi-RPC broadcasting
                success = await self._multi_rpc_broadcast(tx_state, transaction_data)
                
                if success:
                    # Transaction submitted, wait for confirmation
                    confirmed = await self._wait_for_confirmation(tx_state)
                    
                    if confirmed:
                        tx_state.status = TransactionStatus.CONFIRMED
                        tx_state.confirmed_at = time.time()
                        
                        confirmation_time = tx_state.confirmed_at - tx_state.created_at
                        self.network_monitor.add_transaction_result(True, confirmation_time, tx_state.priority_fee)
                        self.successful_transactions += 1
                        
                        logger.info(f"✅ TRANSACTION CONFIRMED: {tx_id[:8]} in {confirmation_time:.2f}s")
                        break
                    else:
                        # Confirmation failed, escalate or retry
                        if tx_state.should_escalate:
                            await self._escalate_priority_fee(tx_state)
                        
                        tx_state.status = TransactionStatus.RETRYING
                        continue
                else:
                    # Submission failed
                    tx_state.status = TransactionStatus.FAILED
                    if tx_state.should_escalate:
                        await self._escalate_priority_fee(tx_state)
                        tx_state.status = TransactionStatus.RETRYING
                    
                    # Brief pause before retry
                    await asyncio.sleep(1)
            
            # Handle final state
            if tx_state.status != TransactionStatus.CONFIRMED:
                if tx_state.is_expired:
                    tx_state.status = TransactionStatus.EXPIRED
                    logger.warning(f"⏰ TRANSACTION EXPIRED: {tx_id[:8]}")
                else:
                    tx_state.status = TransactionStatus.FAILED
                    logger.error(f"💥 TRANSACTION FAILED: {tx_id[:8]} after {tx_state.submission_attempts} attempts")
                
                self.failed_transactions += 1
                self.network_monitor.add_transaction_result(False, 0, tx_state.priority_fee)
            
            # Clean up
            if tx_id in self.cancellation_orders:
                del self.cancellation_orders[tx_id]
            
            self.transaction_history.append(tx_state)
            
            execution_time = time.time() - start_time
            logger.info(f"⚔️ WARFARE COMPLETE: {instruction_type} {tx_state.status.value} in {execution_time:.2f}s")
            
            return tx_state
            
        except Exception as e:
            logger.error(f"💥 WARFARE SYSTEM ERROR: {str(e)}")
            tx_state.status = TransactionStatus.FAILED
            tx_state.failure_reasons.append(FailureReason.RPC_ERROR)
            return tx_state
        
        finally:
            # Remove from active tracking
            if tx_id in self.active_transactions:
                del self.active_transactions[tx_id]
    
    async def _multi_rpc_broadcast(self, tx_state: TransactionState, transaction_data: Dict) -> bool:
        """Broadcast transaction to multiple RPCs simultaneously"""
        try:
            # Get healthy RPCs
            healthy_rpcs = [rpc for rpc in self.rpc_endpoints if rpc.is_healthy]
            
            if not healthy_rpcs:
                logger.error("💥 NO HEALTHY RPC ENDPOINTS AVAILABLE")
                return False
            
            # Sort by health score
            healthy_rpcs.sort(key=lambda x: x.health_score, reverse=True)
            
            # Use top 3 RPCs for broadcasting
            broadcast_rpcs = healthy_rpcs[:3]
            
            logger.debug(f"📡 Broadcasting to {len(broadcast_rpcs)} RPCs: {[rpc.name for rpc in broadcast_rpcs]}")
            
            # Create broadcast tasks
            broadcast_tasks = []
            for rpc in broadcast_rpcs:
                task = asyncio.create_task(
                    self._submit_to_rpc(rpc, tx_state, transaction_data)
                )
                broadcast_tasks.append(task)
            
            # Wait for first success or all failures
            try:
                # Use asyncio.wait with FIRST_COMPLETED
                done, pending = await asyncio.wait(
                    broadcast_tasks, 
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=10  # 10 second timeout
                )
                
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                
                # Check results
                for task in done:
                    try:
                        success, signature = await task
                        if success and signature:
                            tx_state.signature = signature
                            tx_state.submitted_at = time.time()
                            logger.debug(f"✅ Broadcast successful: {signature[:16]}...")
                            return True
                    except Exception as e:
                        logger.debug(f"🔸 Broadcast task error: {str(e)}")
                
                logger.warning("⚠️ All broadcast attempts failed")
                return False
                
            except asyncio.TimeoutError:
                logger.warning("⏰ Broadcast timeout - network too slow")
                return False
            
        except Exception as e:
            logger.error(f"💥 Multi-RPC broadcast error: {str(e)}")
            return False
    
    async def _submit_to_rpc(self, rpc: RPCEndpoint, tx_state: TransactionState, transaction_data: Dict) -> Tuple[bool, Optional[str]]:
        """Submit transaction to specific RPC endpoint"""
        start_time = time.time()
        
        try:
            # Simulate RPC submission (replace with actual Solana RPC call)
            # In production, this would use solana-py or similar library
            
            logger.debug(f"📡 Submitting to {rpc.name}...")
            
            # Simulate network call
            await asyncio.sleep(0.1 + rpc.average_latency / 1000)  # Convert ms to seconds
            
            # Simulate success/failure based on RPC health
            import random
            success_probability = rpc.health_score
            
            if random.random() < success_probability:
                # Simulate successful submission
                signature = f"sig_{tx_state.transaction_id[:8]}_{int(time.time())}"
                
                latency = (time.time() - start_time) * 1000  # Convert to ms
                rpc.record_success(latency)
                
                return True, signature
            else:
                # Simulate failure
                rpc.record_failure("Simulated RPC failure")
                return False, None
                
        except Exception as e:
            logger.debug(f"💥 RPC {rpc.name} submission error: {str(e)}")
            rpc.record_failure(str(e))
            return False, None
    
    async def _wait_for_confirmation(self, tx_state: TransactionState) -> bool:
        """Wait for transaction confirmation"""
        if not tx_state.signature:
            return False
        
        logger.debug(f"⏳ Waiting for confirmation: {tx_state.signature[:16]}...")
        
        # Simulate confirmation wait (replace with actual confirmation polling)
        confirmation_timeout = 20  # 20 seconds
        start_time = time.time()
        
        while time.time() - start_time < confirmation_timeout:
            await asyncio.sleep(1)
            
            # Simulate confirmation check
            import random
            network_metrics = self.network_monitor.get_network_metrics()
            
            # Higher chance of confirmation with better network conditions
            if network_metrics.network_condition == NetworkCondition.CLEAR:
                confirmation_chance = 0.3
            elif network_metrics.network_condition == NetworkCondition.BUSY:
                confirmation_chance = 0.2
            elif network_metrics.network_condition == NetworkCondition.CONGESTED:
                confirmation_chance = 0.1
            else:
                confirmation_chance = 0.05
            
            if random.random() < confirmation_chance:
                logger.debug(f"✅ Transaction confirmed: {tx_state.signature[:16]}...")
                return True
        
        logger.warning(f"⏰ Confirmation timeout: {tx_state.signature[:16]}...")
        return False
    
    async def _escalate_priority_fee(self, tx_state: TransactionState):
        """Escalate priority fee for better execution"""
        old_fee = tx_state.priority_fee
        
        # Double the priority fee, but cap at maximum
        new_fee = min(tx_state.priority_fee * 2, tx_state.max_priority_fee)
        
        if new_fee > old_fee:
            tx_state.priority_fee = new_fee
            tx_state.escalation_count += 1
            self.escalations_performed += 1
            
            logger.warning(f"⬆️ ESCALATING PRIORITY FEE: {old_fee} → {new_fee} lamports "
                          f"(Escalation #{tx_state.escalation_count})")
        else:
            logger.warning(f"🔴 MAXIMUM PRIORITY FEE REACHED: {tx_state.max_priority_fee} lamports")
    
    async def _prepare_cancellation_order(self, tx_state: TransactionState):
        """Prepare pre-signed cancellation order"""
        try:
            # In production, this would create an actual cancellation transaction
            cancellation_data = {
                'original_tx_id': tx_state.transaction_id,
                'instruction_type': 'cancel',
                'created_at': time.time(),
                'priority_fee': tx_state.priority_fee * 2  # Higher priority for cancellation
            }
            
            self.cancellation_orders[tx_state.transaction_id] = cancellation_data
            
            logger.debug(f"🚫 Prepared cancellation order for {tx_state.transaction_id[:8]}")
            
        except Exception as e:
            logger.error(f"💥 Error preparing cancellation order: {str(e)}")
    
    async def emergency_cancel_transaction(self, transaction_id: str) -> bool:
        """Emergency cancellation of active transaction"""
        try:
            if transaction_id not in self.active_transactions:
                logger.warning(f"⚠️ Transaction {transaction_id[:8]} not found for cancellation")
                return False
            
            tx_state = self.active_transactions[transaction_id]
            
            if transaction_id in self.cancellation_orders:
                cancel_order = self.cancellation_orders[transaction_id]
                
                logger.critical(f"🚫 EMERGENCY CANCEL: {transaction_id[:8]}")
                
                # Execute cancellation through multi-RPC broadcast
                # In production, submit the pre-signed cancellation transaction
                
                tx_state.status = TransactionStatus.CANCELLED
                
                # Clean up
                del self.active_transactions[transaction_id]
                del self.cancellation_orders[transaction_id]
                
                return True
            else:
                logger.error(f"💥 No cancellation order found for {transaction_id[:8]}")
                return False
                
        except Exception as e:
            logger.error(f"💥 Emergency cancellation error: {str(e)}")
            return False
    
    def _generate_transaction_id(self, transaction_data: Dict, instruction_type: str) -> str:
        """Generate unique transaction ID"""
        data_str = json.dumps(transaction_data, sort_keys=True)
        hash_input = f"{instruction_type}_{data_str}_{time.time()}".encode()
        return hashlib.sha256(hash_input).hexdigest()
    
    def get_warfare_status(self) -> Dict[str, Any]:
        """Get current warfare system status"""
        network_metrics = self.network_monitor.get_network_metrics()
        
        # Calculate RPC health
        rpc_health = {}
        for rpc in self.rpc_endpoints:
            rpc_health[rpc.name] = {
                'health_score': rpc.health_score,
                'success_rate': rpc.success_rate,
                'average_latency': rpc.average_latency,
                'is_healthy': rpc.is_healthy
            }
        
        success_rate = (self.successful_transactions / max(1, self.total_transactions)) * 100
        
        return {
            'total_transactions': self.total_transactions,
            'successful_transactions': self.successful_transactions,
            'failed_transactions': self.failed_transactions,
            'success_rate': success_rate,
            'escalations_performed': self.escalations_performed,
            'active_transactions': len(self.active_transactions),
            'network_condition': network_metrics.network_condition.value,
            'network_congestion': network_metrics.congestion_score,
            'recommended_priority_fee': network_metrics.recommended_priority_fee,
            'rpc_endpoints': rpc_health,
            'average_confirmation_time': network_metrics.average_confirmation_time
        }
    
    def emergency_warfare_mode(self):
        """Activate emergency warfare protocols"""
        logger.critical("🚨 EMERGENCY WARFARE MODE ACTIVATED")
        
        # Increase all priority fees to maximum
        for tx_state in self.active_transactions.values():
            tx_state.priority_fee = tx_state.max_priority_fee
            tx_state.max_retries = 10  # More aggressive retries
        
        # Reduce timeouts for faster failures
        self.default_timeout_seconds = 15
        
        logger.critical("⚔️ Emergency protocols engaged: Max priority fees, aggressive retries")
    
    async def network_recovery_mode(self):
        """Attempt to recover from network issues"""
        logger.warning("🔄 NETWORK RECOVERY MODE: Attempting to restore connectivity")
        
        # Test all RPC endpoints
        for rpc in self.rpc_endpoints:
            try:
                # Simulate health check
                await asyncio.sleep(0.1)
                rpc.consecutive_failures = 0
                rpc.is_healthy = True
                logger.info(f"✅ RPC {rpc.name} recovered")
            except Exception as e:
                logger.warning(f"❌ RPC {rpc.name} still failing: {str(e)}")
        
        # Reset escalation settings
        self.default_timeout_seconds = 30
        
        logger.info("🔄 Network recovery attempt completed") 