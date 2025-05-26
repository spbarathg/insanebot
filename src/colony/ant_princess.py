"""
Ant Princess - Accumulation wallet for withdrawals and reinvestment

Manages capital accumulation, withdrawal processing, and reinvestment decisions.
Serves as the final destination for profitable capital before external transfers.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .base_ant import BaseAnt, AntRole, AntStatus

logger = logging.getLogger(__name__)

@dataclass
class AccumulationBatch:
    """Represents a batch of accumulated capital"""
    batch_id: str
    amount: float
    source_queens: List[str]
    accumulated_at: float
    ready_for_withdrawal: bool = False
    withdrawal_threshold: float = 10.0

@dataclass
class WithdrawalRequest:
    """Represents a withdrawal request"""
    request_id: str
    amount: float
    destination: str
    requested_at: float
    status: str = "pending"  # pending, approved, executed, failed

class AntPrincess(BaseAnt):
    """Accumulation wallet for profitable capital management"""
    
    def __init__(self, ant_id: str = "ant_princess_0"):
        super().__init__(ant_id, AntRole.PRINCESS)
        
        # Princess-specific attributes
        self.accumulation_batches: Dict[str, AccumulationBatch] = {}
        self.withdrawal_requests: Dict[str, WithdrawalRequest] = {}
        self.pending_deposits: List[Dict] = []
        
        # Accumulation settings
        self.accumulation_threshold = self.config["accumulation_threshold"]  # 10 SOL default
        self.auto_reinvest_percentage = 0.3  # 30% auto-reinvest
        self.withdrawal_cooldown = 3600.0  # 1 hour between withdrawals
        
        # Princess metrics
        self.total_accumulated: float = 0.0
        self.total_withdrawn: float = 0.0
        self.total_reinvested: float = 0.0
        self.last_withdrawal: float = 0.0
        self.accumulation_efficiency: float = 1.0
        
        logger.info(f"AntPrincess {ant_id} created for capital accumulation")
    
    async def initialize(self) -> bool:
        """Initialize the Ant Princess"""
        try:
            # Initialize accumulation tracking
            await self._initialize_accumulation_system()
            
            # Initialize withdrawal system
            await self._initialize_withdrawal_system()
            
            logger.info(f"AntPrincess {self.ant_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AntPrincess {self.ant_id}: {e}")
            self.status = AntStatus.ERROR
            return False
    
    async def execute_cycle(self) -> Dict[str, Any]:
        """Execute Princess accumulation and withdrawal cycle"""
        if self.status != AntStatus.ACTIVE:
            return {"status": "inactive", "reason": f"Princess status: {self.status.value}"}
        
        try:
            self.update_activity()
            
            # Process pending deposits
            deposit_result = await self._process_pending_deposits()
            
            # Check accumulation batches for withdrawal readiness
            batch_assessment = await self._assess_accumulation_batches()
            
            # Process withdrawal requests
            withdrawal_result = await self._process_withdrawal_requests()
            
            # Evaluate reinvestment opportunities
            reinvestment_result = await self._evaluate_reinvestment()
            
            # Update accumulation metrics
            metrics_update = await self._update_accumulation_metrics()
            
            return {
                "status": "active",
                "deposit_processing": deposit_result,
                "batch_assessment": batch_assessment,
                "withdrawal_processing": withdrawal_result,
                "reinvestment_evaluation": reinvestment_result,
                "metrics_update": metrics_update,
                "princess_summary": self._get_princess_summary()
            }
            
        except Exception as e:
            logger.error(f"Error in AntPrincess {self.ant_id} cycle: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _initialize_accumulation_system(self):
        """Initialize the accumulation tracking system"""
        self.accumulation_batches = {}
        self.pending_deposits = []
    
    async def _initialize_withdrawal_system(self):
        """Initialize the withdrawal processing system"""
        self.withdrawal_requests = {}
    
    async def receive_deposit(self, amount: float, source_queen_id: str) -> bool:
        """Receive a deposit from a Queen"""
        try:
            deposit = {
                "amount": amount,
                "source_queen": source_queen_id,
                "received_at": time.time(),
                "processed": False
            }
            
            self.pending_deposits.append(deposit)
            
            logger.info(f"AntPrincess {self.ant_id} received deposit of {amount} SOL from {source_queen_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error receiving deposit: {e}")
            return False
    
    async def _process_pending_deposits(self) -> Dict[str, Any]:
        """Process all pending deposits"""
        try:
            processing_result = {
                "deposits_processed": 0,
                "total_amount_processed": 0.0,
                "new_batches_created": 0,
                "failed_deposits": 0
            }
            
            for deposit in self.pending_deposits[:]:  # Create copy to iterate
                if not deposit["processed"]:
                    success = await self._process_single_deposit(deposit)
                    if success:
                        deposit["processed"] = True
                        processing_result["deposits_processed"] += 1
                        processing_result["total_amount_processed"] += deposit["amount"]
                    else:
                        processing_result["failed_deposits"] += 1
            
            # Remove processed deposits
            self.pending_deposits = [d for d in self.pending_deposits if not d["processed"]]
            
            return processing_result
            
        except Exception as e:
            logger.error(f"Error processing deposits: {e}")
            return {"error": str(e)}
    
    async def _process_single_deposit(self, deposit: Dict) -> bool:
        """Process a single deposit"""
        try:
            amount = deposit["amount"]
            source_queen = deposit["source_queen"]
            
            # Update Princess capital
            self.capital.update_balance(self.capital.current_balance + amount)
            self.total_accumulated += amount
            
            # Create or update accumulation batch
            batch_id = f"batch_{int(time.time())}"
            
            accumulation_batch = AccumulationBatch(
                batch_id=batch_id,
                amount=amount,
                source_queens=[source_queen],
                accumulated_at=time.time(),
                withdrawal_threshold=self.accumulation_threshold
            )
            
            # Check if batch is ready for withdrawal
            if amount >= self.accumulation_threshold:
                accumulation_batch.ready_for_withdrawal = True
            
            self.accumulation_batches[batch_id] = accumulation_batch
            
            logger.debug(f"Processed deposit: {amount} SOL from {source_queen}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing single deposit: {e}")
            return False
    
    async def _assess_accumulation_batches(self) -> Dict[str, Any]:
        """Assess accumulation batches for withdrawal readiness"""
        try:
            assessment = {
                "total_batches": len(self.accumulation_batches),
                "ready_for_withdrawal": 0,
                "withdrawal_ready_amount": 0.0,
                "batch_details": []
            }
            
            for batch_id, batch in self.accumulation_batches.items():
                batch_info = {
                    "batch_id": batch_id,
                    "amount": batch.amount,
                    "ready": batch.ready_for_withdrawal,
                    "age_hours": (time.time() - batch.accumulated_at) / 3600
                }
                
                assessment["batch_details"].append(batch_info)
                
                if batch.ready_for_withdrawal:
                    assessment["ready_for_withdrawal"] += 1
                    assessment["withdrawal_ready_amount"] += batch.amount
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing accumulation batches: {e}")
            return {"error": str(e)}
    
    async def request_withdrawal(self, amount: float, destination: str = "external") -> str:
        """Request a withdrawal of accumulated capital"""
        try:
            # Check if we have enough capital
            if amount > self.capital.current_balance:
                raise ValueError(f"Insufficient capital for withdrawal: {amount} > {self.capital.current_balance}")
            
            # Check withdrawal cooldown
            if time.time() - self.last_withdrawal < self.withdrawal_cooldown:
                raise ValueError("Withdrawal cooldown period not met")
            
            request_id = f"withdrawal_{int(time.time())}_{len(self.withdrawal_requests)}"
            
            withdrawal_request = WithdrawalRequest(
                request_id=request_id,
                amount=amount,
                destination=destination,
                requested_at=time.time(),
                status="pending"
            )
            
            self.withdrawal_requests[request_id] = withdrawal_request
            
            logger.info(f"AntPrincess {self.ant_id} withdrawal requested: {amount} SOL to {destination}")
            return request_id
            
        except Exception as e:
            logger.error(f"Error requesting withdrawal: {e}")
            raise
    
    async def _process_withdrawal_requests(self) -> Dict[str, Any]:
        """Process pending withdrawal requests"""
        try:
            processing_result = {
                "requests_processed": 0,
                "total_withdrawn": 0.0,
                "failed_requests": 0,
                "pending_requests": 0
            }
            
            for request_id, request in self.withdrawal_requests.items():
                if request.status == "pending":
                    success = await self._execute_withdrawal(request)
                    if success:
                        request.status = "executed"
                        processing_result["requests_processed"] += 1
                        processing_result["total_withdrawn"] += request.amount
                        self.last_withdrawal = time.time()
                    else:
                        request.status = "failed"
                        processing_result["failed_requests"] += 1
                elif request.status == "pending":
                    processing_result["pending_requests"] += 1
            
            return processing_result
            
        except Exception as e:
            logger.error(f"Error processing withdrawal requests: {e}")
            return {"error": str(e)}
    
    async def _execute_withdrawal(self, request: WithdrawalRequest) -> bool:
        """Execute a withdrawal request"""
        try:
            # Check if we still have enough capital
            if request.amount > self.capital.current_balance:
                logger.warning(f"Insufficient capital for withdrawal {request.request_id}")
                return False
            
            # Update capital
            self.capital.update_balance(self.capital.current_balance - request.amount)
            self.total_withdrawn += request.amount
            
            # Remove corresponding accumulation batches
            await self._deduct_from_batches(request.amount)
            
            # In production, this would execute actual withdrawal to external wallet
            logger.info(f"Withdrawal executed: {request.amount} SOL to {request.destination}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing withdrawal: {e}")
            return False
    
    async def _deduct_from_batches(self, amount: float):
        """Deduct amount from accumulation batches (FIFO)"""
        remaining_amount = amount
        batches_to_remove = []
        
        # Sort batches by accumulation time (oldest first)
        sorted_batches = sorted(
            self.accumulation_batches.items(),
            key=lambda x: x[1].accumulated_at
        )
        
        for batch_id, batch in sorted_batches:
            if remaining_amount <= 0:
                break
            
            if batch.amount <= remaining_amount:
                # Remove entire batch
                remaining_amount -= batch.amount
                batches_to_remove.append(batch_id)
            else:
                # Partial deduction
                batch.amount -= remaining_amount
                remaining_amount = 0
        
        # Remove depleted batches
        for batch_id in batches_to_remove:
            del self.accumulation_batches[batch_id]
    
    async def _evaluate_reinvestment(self) -> Dict[str, Any]:
        """Evaluate opportunities for reinvestment"""
        try:
            evaluation = {
                "reinvestment_recommended": False,
                "recommended_amount": 0.0,
                "target_queens": [],
                "rationale": []
            }
            
            # Check if we have excess capital for reinvestment
            available_for_reinvestment = self.capital.current_balance * self.auto_reinvest_percentage
            
            if available_for_reinvestment >= 2.0:  # Minimum amount to create a Queen
                evaluation["reinvestment_recommended"] = True
                evaluation["recommended_amount"] = available_for_reinvestment
                evaluation["rationale"].append("Sufficient capital for Queen creation")
                
                # In production, this would analyze which Queens or system areas need capital
                evaluation["target_queens"] = ["new_queen_creation"]
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating reinvestment: {e}")
            return {"error": str(e)}
    
    async def execute_reinvestment(self, amount: float, target: str) -> bool:
        """Execute a reinvestment to the specified target"""
        try:
            if amount > self.capital.current_balance:
                return False
            
            # Update capital
            self.capital.update_balance(self.capital.current_balance - amount)
            self.total_reinvested += amount
            
            # In production, this would transfer capital back to the system
            logger.info(f"Reinvestment executed: {amount} SOL to {target}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing reinvestment: {e}")
            return False
    
    async def _update_accumulation_metrics(self) -> Dict[str, Any]:
        """Update accumulation efficiency and metrics"""
        try:
            # Calculate accumulation efficiency
            if self.total_accumulated > 0:
                self.accumulation_efficiency = (
                    (self.total_withdrawn + self.total_reinvested + self.capital.current_balance) / 
                    self.total_accumulated
                )
            else:
                self.accumulation_efficiency = 1.0
            
            return {
                "total_accumulated": self.total_accumulated,
                "total_withdrawn": self.total_withdrawn,
                "total_reinvested": self.total_reinvested,
                "current_balance": self.capital.current_balance,
                "accumulation_efficiency": self.accumulation_efficiency
            }
            
        except Exception as e:
            logger.error(f"Error updating accumulation metrics: {e}")
            return {"error": str(e)}
    
    def _get_princess_summary(self) -> Dict[str, Any]:
        """Get comprehensive Princess summary"""
        return {
            "capital_status": {
                "current_balance": self.capital.current_balance,
                "total_accumulated": self.total_accumulated,
                "total_withdrawn": self.total_withdrawn,
                "total_reinvested": self.total_reinvested
            },
            "accumulation_batches": {
                "total_batches": len(self.accumulation_batches),
                "ready_for_withdrawal": sum(1 for b in self.accumulation_batches.values() if b.ready_for_withdrawal),
                "total_batch_amount": sum(b.amount for b in self.accumulation_batches.values())
            },
            "withdrawal_status": {
                "pending_requests": sum(1 for r in self.withdrawal_requests.values() if r.status == "pending"),
                "total_requests": len(self.withdrawal_requests),
                "last_withdrawal_hours_ago": (time.time() - self.last_withdrawal) / 3600 if self.last_withdrawal > 0 else None
            },
            "efficiency_metrics": {
                "accumulation_efficiency": self.accumulation_efficiency,
                "pending_deposits": len(self.pending_deposits)
            }
        }
    
    def get_withdrawal_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific withdrawal request"""
        if request_id in self.withdrawal_requests:
            request = self.withdrawal_requests[request_id]
            return {
                "request_id": request_id,
                "amount": request.amount,
                "destination": request.destination,
                "status": request.status,
                "requested_at": request.requested_at
            }
        return None
    
    async def cleanup(self):
        """Cleanup Princess resources"""
        try:
            # Process any remaining withdrawal requests
            for request in self.withdrawal_requests.values():
                if request.status == "pending":
                    await self._execute_withdrawal(request)
            
            logger.info(f"AntPrincess {self.ant_id} cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during AntPrincess cleanup: {e}") 