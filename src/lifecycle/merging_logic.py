"""
Merging Logic - Handles ant consolidation when underperforming

When ants consistently underperform or have low capital, they can merge
with other ants to pool resources and knowledge.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MergeRequest:
    """Request for ant merging"""
    primary_ant_id: str
    merge_candidate_ids: List[str]
    merge_reason: str
    total_capital: float
    expected_performance_boost: float = 0.0
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass 
class MergeResult:
    """Result of ant merging operation"""
    success: bool
    merged_ant_id: str
    absorbed_ant_ids: List[str]
    combined_capital: float
    combined_experience: Dict[str, Any]
    errors: List[str] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.errors is None:
            self.errors = []

class MergingLogic:
    """Handles the logic for merging underperforming ants"""
    
    def __init__(self):
        self.merge_capital_threshold = 0.1  # SOL - below this, consider merging
        self.min_performance_score = 0.3  # Below this score, consider merging
        self.max_merge_group_size = 5  # Maximum ants in a merge
        self.merge_cooldown = 3600  # 1 hour between merge attempts
        
        # Track merges for analytics
        self.merge_history: List[MergeResult] = []
        self.last_merge_attempts: Dict[str, float] = {}
        
    def should_merge(self, ant_id: str, ant_capital: float, performance_score: float = 0.5, 
                    ant_age: float = 0) -> bool:
        """Determine if an ant should consider merging"""
        try:
            # Check cooldown
            if ant_id in self.last_merge_attempts:
                time_since_last = time.time() - self.last_merge_attempts[ant_id]
                if time_since_last < self.merge_cooldown:
                    return False
            
            # Primary criteria: low capital or poor performance
            capital_trigger = ant_capital < self.merge_capital_threshold
            performance_trigger = performance_score < self.min_performance_score
            
            # Age consideration - very young ants might not merge immediately
            age_threshold = 300  # 5 minutes minimum age
            too_young = ant_age < age_threshold
            
            return (capital_trigger or performance_trigger) and not too_young
            
        except Exception as e:
            logger.error(f"Error determining merge eligibility for {ant_id}: {e}")
            return False
    
    def find_merge_candidates(self, primary_ant_id: str, available_ants: List[Dict], 
                            primary_ant_data: Dict) -> List[str]:
        """Find suitable ants for merging with the primary ant"""
        try:
            candidates = []
            primary_capital = primary_ant_data.get("capital", 0)
            primary_performance = primary_ant_data.get("performance_score", 0)
            
            for ant in available_ants:
                ant_id = ant.get("ant_id")
                
                # Skip self
                if ant_id == primary_ant_id:
                    continue
                    
                # Skip if ant is not eligible for merging
                if not self.should_merge(
                    ant_id, 
                    ant.get("capital", 0), 
                    ant.get("performance_score", 0),
                    ant.get("age", 0)
                ):
                    continue
                
                # Check compatibility (similar performance tiers)
                ant_performance = ant.get("performance_score", 0)
                performance_diff = abs(primary_performance - ant_performance)
                
                if performance_diff > 0.4:  # Too different in performance
                    continue
                
                # Check if combined capital makes sense
                combined_capital = primary_capital + ant.get("capital", 0)
                if combined_capital > 1.5:  # Don't create overly large ants
                    continue
                    
                candidates.append(ant_id)
                
                # Limit merge group size
                if len(candidates) >= self.max_merge_group_size - 1:
                    break
                    
            return candidates
            
        except Exception as e:
            logger.error(f"Error finding merge candidates for {primary_ant_id}: {e}")
            return []
    
    def create_merge_request(self, primary_ant_id: str, primary_ant_data: Dict,
                           candidate_ants: List[Dict]) -> Optional[MergeRequest]:
        """Create a merge request"""
        try:
            if not candidate_ants:
                return None
                
            candidate_ids = [ant.get("ant_id") for ant in candidate_ants]
            total_capital = primary_ant_data.get("capital", 0)
            total_capital += sum(ant.get("capital", 0) for ant in candidate_ants)
            
            # Calculate expected performance boost
            num_ants = len(candidate_ants) + 1
            avg_performance = (primary_ant_data.get("performance_score", 0) + 
                             sum(ant.get("performance_score", 0) for ant in candidate_ants)) / num_ants
            
            # Boost expected from combined knowledge and capital
            expected_boost = min(0.2, total_capital * 0.1)  # Cap at 20% boost
            
            reason = f"Low capital/performance merger: {num_ants} ants, {total_capital:.3f} SOL combined"
            
            return MergeRequest(
                primary_ant_id=primary_ant_id,
                merge_candidate_ids=candidate_ids,
                merge_reason=reason,
                total_capital=total_capital,
                expected_performance_boost=expected_boost
            )
            
        except Exception as e:
            logger.error(f"Error creating merge request for {primary_ant_id}: {e}")
            return None
    
    def validate_merge_request(self, request: MergeRequest) -> bool:
        """Validate that a merge request is feasible"""
        try:
            # Check reasonable limits
            if len(request.merge_candidate_ids) < 1:
                logger.error("No merge candidates specified")
                return False
                
            if len(request.merge_candidate_ids) > self.max_merge_group_size - 1:
                logger.error(f"Too many merge candidates: {len(request.merge_candidate_ids)}")
                return False
                
            # Check capital makes sense
            if request.total_capital <= 0:
                logger.error("Invalid total capital for merge")
                return False
                
            # Check for duplicate IDs
            all_ids = [request.primary_ant_id] + request.merge_candidate_ids
            if len(set(all_ids)) != len(all_ids):
                logger.error("Duplicate ant IDs in merge request")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating merge request: {e}")
            return False
    
    def combine_ant_data(self, primary_ant_data: Dict, candidate_ants_data: List[Dict]) -> Dict[str, Any]:
        """Combine the knowledge and experience from multiple ants"""
        try:
            combined_data = {
                "ant_id": primary_ant_data.get("ant_id"),
                "generation": max(primary_ant_data.get("generation", 0),
                                max(ant.get("generation", 0) for ant in candidate_ants_data) if candidate_ants_data else 0),
                "merged_from": [ant.get("ant_id") for ant in candidate_ants_data],
                "merge_timestamp": time.time()
            }
            
            # Combine capital
            combined_data["capital"] = primary_ant_data.get("capital", 0)
            combined_data["capital"] += sum(ant.get("capital", 0) for ant in candidate_ants_data)
            
            # Combine trading experience
            all_coins_traded = set(primary_ant_data.get("coins_traded", []))
            for ant in candidate_ants_data:
                all_coins_traded.update(ant.get("coins_traded", []))
            combined_data["coins_traded"] = list(all_coins_traded)
            
            # Combine successful strategies
            all_strategies = primary_ant_data.get("successful_strategies", [])
            for ant in candidate_ants_data:
                all_strategies.extend(ant.get("successful_strategies", []))
            combined_data["successful_strategies"] = list(set(all_strategies))
            
            # Average performance metrics
            all_ants = [primary_ant_data] + candidate_ants_data
            performance_metrics = {}
            
            # Calculate weighted averages based on number of trades
            total_trades = sum(ant.get("total_trades", 1) for ant in all_ants)
            
            if total_trades > 0:
                for metric in ["win_rate", "avg_return", "sharpe_ratio"]:
                    weighted_sum = sum(
                        ant.get(metric, 0) * ant.get("total_trades", 1) 
                        for ant in all_ants
                    )
                    performance_metrics[metric] = weighted_sum / total_trades
                    
            combined_data["performance_metrics"] = performance_metrics
            combined_data["total_trades"] = total_trades
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error combining ant data: {e}")
            return primary_ant_data
    
    def record_merge_attempt(self, ant_id: str):
        """Record that an ant attempted to merge (for cooldown tracking)"""
        self.last_merge_attempts[ant_id] = time.time()
    
    def record_merge(self, result: MergeResult):
        """Record a merge operation for analytics"""
        try:
            self.merge_history.append(result)
            
            # Keep only recent history to prevent memory bloat
            if len(self.merge_history) > 1000:
                self.merge_history = self.merge_history[-500:]
                
            logger.info(f"Recorded merge: {len(result.absorbed_ant_ids)} ants -> {result.merged_ant_id}")
            
        except Exception as e:
            logger.error(f"Error recording merge: {e}")
    
    def get_merge_statistics(self) -> Dict[str, Any]:
        """Get statistics about merging operations"""
        try:
            if not self.merge_history:
                return {"total_merges": 0, "total_ants_merged": 0}
                
            total_merges = len(self.merge_history)
            total_ants_merged = sum(len(merge.absorbed_ant_ids) for merge in self.merge_history)
            success_rate = sum(1 for merge in self.merge_history if merge.success) / total_merges
            
            return {
                "total_merges": total_merges,
                "total_ants_merged": total_ants_merged,
                "success_rate": success_rate,
                "average_ants_per_merge": total_ants_merged / total_merges if total_merges > 0 else 0,
                "recent_merges": len([m for m in self.merge_history if time.time() - m.timestamp < 3600])
            }
            
        except Exception as e:
            logger.error(f"Error calculating merge statistics: {e}")
            return {"error": str(e)} 