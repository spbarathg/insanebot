"""
Splitting Logic - Handles ant division when reaching capital thresholds

When an ant reaches 2 SOL in capital, it splits into 5 new worker ants,
each inheriting 0.4 SOL and knowledge from the parent.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SplitRequest:
    """Request for ant splitting"""
    parent_ant_id: str
    parent_capital: float
    split_reason: str
    num_children: int = 5
    capital_per_child: float = 0.4
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass 
class SplitResult:
    """Result of ant splitting operation"""
    success: bool
    parent_ant_id: str
    child_ant_ids: List[str]
    capital_distributed: float
    errors: List[str] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.errors is None:
            self.errors = []

class SplittingLogic:
    """Handles the logic for splitting ants when they reach thresholds"""
    
    def __init__(self):
        self.split_threshold = 2.0  # SOL
        self.children_per_split = 5
        self.capital_per_child = 0.4  # SOL
        self.min_parent_retention = 0.0  # Parent retires after split
        
        # Track splits for analytics
        self.split_history: List[SplitResult] = []
        
    def should_split(self, ant_capital: float, ant_age: float = 0, 
                    performance_score: float = 0) -> bool:
        """Determine if an ant should split based on criteria"""
        try:
            # Primary criterion: capital threshold
            if ant_capital < self.split_threshold:
                return False
                
            # Additional checks could be added here:
            # - Minimum age requirement
            # - Performance thresholds
            # - Market conditions
            
            return True
            
        except Exception as e:
            logger.error(f"Error determining split eligibility: {e}")
            return False
    
    def create_split_request(self, ant_id: str, current_capital: float, 
                           performance_metrics: Dict = None) -> Optional[SplitRequest]:
        """Create a split request for an ant"""
        try:
            if not self.should_split(current_capital):
                return None
                
            # Calculate how many children we can create
            available_capital = current_capital - self.min_parent_retention
            max_children = int(available_capital / self.capital_per_child)
            
            # Use default or calculated number of children
            num_children = min(self.children_per_split, max_children)
            
            if num_children < 1:
                logger.warning(f"Insufficient capital for split: {available_capital}")
                return None
                
            reason = f"Capital threshold reached: {current_capital:.2f} SOL"
            if performance_metrics:
                reason += f", Performance: {performance_metrics.get('score', 'N/A')}"
                
            return SplitRequest(
                parent_ant_id=ant_id,
                parent_capital=current_capital,
                split_reason=reason,
                num_children=num_children,
                capital_per_child=self.capital_per_child
            )
            
        except Exception as e:
            logger.error(f"Error creating split request for ant {ant_id}: {e}")
            return None
    
    def validate_split_request(self, request: SplitRequest) -> bool:
        """Validate that a split request is feasible"""
        try:
            # Check capital requirements
            total_required = request.num_children * request.capital_per_child
            if request.parent_capital < total_required:
                logger.error(f"Insufficient capital for split: {request.parent_capital} < {total_required}")
                return False
                
            # Check reasonable limits
            if request.num_children > 10 or request.num_children < 1:
                logger.error(f"Invalid number of children: {request.num_children}")
                return False
                
            if request.capital_per_child <= 0 or request.capital_per_child > 1.0:
                logger.error(f"Invalid capital per child: {request.capital_per_child}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating split request: {e}")
            return False
    
    def generate_child_ids(self, parent_id: str, num_children: int) -> List[str]:
        """Generate unique IDs for child ants"""
        timestamp = int(time.time() * 1000)
        child_ids = []
        
        for i in range(num_children):
            child_id = f"{parent_id}_child_{i+1}_{timestamp}"
            child_ids.append(child_id)
            
        return child_ids
    
    def calculate_inheritance(self, parent_data: Dict, child_id: str) -> Dict[str, Any]:
        """Calculate what each child inherits from parent"""
        try:
            inheritance = {
                "parent_id": parent_data.get("ant_id"),
                "parent_generation": parent_data.get("generation", 0) + 1,
                "inherited_capital": self.capital_per_child,
                "inherited_strategies": parent_data.get("successful_strategies", []),
                "inherited_experience": {
                    "coins_traded": parent_data.get("coins_traded", []),
                    "performance_metrics": parent_data.get("performance_summary", {}),
                    "risk_preferences": parent_data.get("risk_preferences", {})
                },
                "creation_timestamp": time.time(),
                "child_id": child_id
            }
            
            return inheritance
            
        except Exception as e:
            logger.error(f"Error calculating inheritance for {child_id}: {e}")
            return {"child_id": child_id, "inherited_capital": self.capital_per_child}
    
    def record_split(self, result: SplitResult):
        """Record a split operation for analytics"""
        try:
            self.split_history.append(result)
            
            # Keep only recent history to prevent memory bloat
            if len(self.split_history) > 1000:
                self.split_history = self.split_history[-500:]
                
            logger.info(f"Recorded split: {result.parent_ant_id} -> {len(result.child_ant_ids)} children")
            
        except Exception as e:
            logger.error(f"Error recording split: {e}")
    
    def get_split_statistics(self) -> Dict[str, Any]:
        """Get statistics about splitting operations"""
        try:
            if not self.split_history:
                return {"total_splits": 0, "total_children_created": 0}
                
            total_splits = len(self.split_history)
            total_children = sum(len(split.child_ant_ids) for split in self.split_history)
            success_rate = sum(1 for split in self.split_history if split.success) / total_splits
            
            return {
                "total_splits": total_splits,
                "total_children_created": total_children,
                "success_rate": success_rate,
                "average_children_per_split": total_children / total_splits if total_splits > 0 else 0,
                "recent_splits": len([s for s in self.split_history if time.time() - s.timestamp < 3600])
            }
            
        except Exception as e:
            logger.error(f"Error calculating split statistics: {e}")
            return {"error": str(e)} 