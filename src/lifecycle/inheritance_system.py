"""
Inheritance System - Handles knowledge and capital transfer between ant generations

When ants split or retire, their knowledge, strategies, and capital need to be
properly transferred to the next generation of ants.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class InheritancePackage:
    """Package of inheritance data for a new ant"""
    parent_id: str
    inherited_capital: float
    inherited_strategies: List[Dict[str, Any]]
    performance_history: Dict[str, Any]
    trading_preferences: Dict[str, Any]
    risk_profile: Dict[str, Any]
    learned_patterns: List[Dict[str, Any]]
    generation: int
    creation_timestamp: float = None
    
    def __post_init__(self):
        if self.creation_timestamp is None:
            self.creation_timestamp = time.time()

@dataclass
class KnowledgeTransfer:
    """Represents a knowledge transfer between ants"""
    source_ant_id: str
    target_ant_id: str
    knowledge_type: str
    knowledge_data: Dict[str, Any]
    transfer_reason: str
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class InheritanceSystem:
    """Manages inheritance and knowledge transfer between ants"""
    
    def __init__(self):
        self.inheritance_history: List[InheritancePackage] = []
        self.knowledge_transfers: List[KnowledgeTransfer] = []
        self.generation_performance: Dict[int, Dict[str, Any]] = {}
        
        # Configuration
        self.max_inherited_strategies = 10
        self.knowledge_decay_factor = 0.9  # Knowledge becomes less relevant over time
        self.performance_weight_recent = 0.7  # Weight recent performance more heavily
        
    def create_inheritance_package(self, parent_ant_data: Dict[str, Any], 
                                 child_capital: float = 0.4) -> InheritancePackage:
        """Create an inheritance package for a new ant"""
        try:
            # Extract parent's successful strategies
            all_strategies = parent_ant_data.get("successful_strategies", [])
            
            # Filter and rank strategies by success rate
            top_strategies = self._filter_top_strategies(all_strategies)
            
            # Extract performance history
            performance_history = self._extract_performance_history(parent_ant_data)
            
            # Extract trading preferences
            trading_preferences = self._extract_trading_preferences(parent_ant_data)
            
            # Extract risk profile
            risk_profile = self._extract_risk_profile(parent_ant_data)
            
            # Extract learned patterns
            learned_patterns = self._extract_learned_patterns(parent_ant_data)
            
            package = InheritancePackage(
                parent_id=parent_ant_data.get("ant_id", "unknown"),
                inherited_capital=child_capital,
                inherited_strategies=top_strategies,
                performance_history=performance_history,
                trading_preferences=trading_preferences,
                risk_profile=risk_profile,
                learned_patterns=learned_patterns,
                generation=parent_ant_data.get("generation", 0) + 1
            )
            
            # Record the inheritance
            self.inheritance_history.append(package)
            
            logger.info(f"Created inheritance package from {package.parent_id} "
                       f"with {len(top_strategies)} strategies")
            
            return package
            
        except Exception as e:
            logger.error(f"Error creating inheritance package: {e}")
            # Return minimal package
            return InheritancePackage(
                parent_id=parent_ant_data.get("ant_id", "unknown"),
                inherited_capital=child_capital,
                inherited_strategies=[],
                performance_history={},
                trading_preferences={},
                risk_profile={},
                learned_patterns=[],
                generation=parent_ant_data.get("generation", 0) + 1
            )
    
    def _filter_top_strategies(self, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and return the top performing strategies"""
        try:
            if not strategies:
                return []
            
            # Sort strategies by success rate and profit
            sorted_strategies = sorted(
                strategies,
                key=lambda s: (s.get("success_rate", 0), s.get("avg_profit", 0)),
                reverse=True
            )
            
            # Take the top strategies up to the limit
            return sorted_strategies[:self.max_inherited_strategies]
            
        except Exception as e:
            logger.error(f"Error filtering strategies: {e}")
            return strategies[:self.max_inherited_strategies] if strategies else []
    
    def _extract_performance_history(self, ant_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant performance history"""
        try:
            performance = ant_data.get("performance_metrics", {})
            
            return {
                "total_trades": performance.get("total_trades", 0),
                "win_rate": performance.get("win_rate", 0),
                "avg_return": performance.get("avg_return", 0),
                "max_drawdown": performance.get("max_drawdown", 0),
                "sharpe_ratio": performance.get("sharpe_ratio", 0),
                "best_performing_coins": ant_data.get("best_performing_coins", []),
                "worst_performing_coins": ant_data.get("worst_performing_coins", [])
            }
            
        except Exception as e:
            logger.error(f"Error extracting performance history: {e}")
            return {}
    
    def _extract_trading_preferences(self, ant_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract trading preferences and patterns"""
        try:
            return {
                "preferred_trade_size": ant_data.get("avg_trade_size", 0.1),
                "preferred_holding_time": ant_data.get("avg_holding_time", 3600),
                "risk_tolerance": ant_data.get("risk_tolerance", 0.5),
                "sector_preferences": ant_data.get("sector_preferences", {}),
                "time_of_day_preferences": ant_data.get("time_preferences", {}),
                "market_condition_preferences": ant_data.get("market_preferences", {})
            }
            
        except Exception as e:
            logger.error(f"Error extracting trading preferences: {e}")
            return {}
    
    def _extract_risk_profile(self, ant_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract risk management profile"""
        try:
            return {
                "max_position_size": ant_data.get("max_position_size", 0.2),
                "stop_loss_threshold": ant_data.get("stop_loss_threshold", 0.1),
                "take_profit_threshold": ant_data.get("take_profit_threshold", 0.3),
                "max_concurrent_positions": ant_data.get("max_concurrent_positions", 3),
                "volatility_tolerance": ant_data.get("volatility_tolerance", 0.5)
            }
            
        except Exception as e:
            logger.error(f"Error extracting risk profile: {e}")
            return {}
    
    def _extract_learned_patterns(self, ant_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract learned market patterns and insights"""
        try:
            patterns = ant_data.get("learned_patterns", [])
            
            # Filter patterns by confidence and recency
            current_time = time.time()
            
            filtered_patterns = []
            for pattern in patterns:
                # Check if pattern is recent enough to be relevant
                pattern_age = current_time - pattern.get("learned_timestamp", 0)
                max_age = 86400 * 7  # 7 days
                
                if pattern_age > max_age:
                    continue
                    
                # Check confidence threshold
                if pattern.get("confidence", 0) < 0.6:
                    continue
                    
                # Apply decay factor based on age
                decay_factor = max(0.1, self.knowledge_decay_factor ** (pattern_age / 3600))
                pattern["confidence"] *= decay_factor
                
                filtered_patterns.append(pattern)
            
            # Sort by confidence and return top patterns
            filtered_patterns.sort(key=lambda p: p.get("confidence", 0), reverse=True)
            return filtered_patterns[:20]  # Limit to top 20 patterns
            
        except Exception as e:
            logger.error(f"Error extracting learned patterns: {e}")
            return []
    
    def apply_inheritance(self, ant_data: Dict[str, Any], 
                         inheritance_package: InheritancePackage) -> Dict[str, Any]:
        """Apply inheritance package to a new ant's data"""
        try:
            # Start with the ant's base data
            enhanced_data = ant_data.copy()
            
            # Apply inherited capital
            enhanced_data["capital"] = inheritance_package.inherited_capital
            enhanced_data["initial_capital"] = inheritance_package.inherited_capital
            
            # Apply inherited strategies
            enhanced_data["inherited_strategies"] = inheritance_package.inherited_strategies
            enhanced_data["available_strategies"] = inheritance_package.inherited_strategies.copy()
            
            # Apply performance insights
            enhanced_data["parent_performance"] = inheritance_package.performance_history
            
            # Apply trading preferences
            enhanced_data.update(inheritance_package.trading_preferences)
            
            # Apply risk profile
            enhanced_data["risk_profile"] = inheritance_package.risk_profile
            
            # Apply learned patterns
            enhanced_data["inherited_patterns"] = inheritance_package.learned_patterns
            
            # Set generation
            enhanced_data["generation"] = inheritance_package.generation
            enhanced_data["parent_id"] = inheritance_package.parent_id
            
            # Record knowledge transfer
            transfer = KnowledgeTransfer(
                source_ant_id=inheritance_package.parent_id,
                target_ant_id=ant_data.get("ant_id", "unknown"),
                knowledge_type="full_inheritance",
                knowledge_data={
                    "num_strategies": len(inheritance_package.inherited_strategies),
                    "num_patterns": len(inheritance_package.learned_patterns),
                    "capital": inheritance_package.inherited_capital
                },
                transfer_reason="ant_birth_inheritance"
            )
            
            self.knowledge_transfers.append(transfer)
            
            logger.info(f"Applied inheritance to ant {ant_data.get('ant_id')}: "
                       f"{len(inheritance_package.inherited_strategies)} strategies, "
                       f"{inheritance_package.inherited_capital} SOL")
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Error applying inheritance: {e}")
            return ant_data
    
    def transfer_knowledge(self, source_ant_data: Dict[str, Any], 
                          target_ant_data: Dict[str, Any], 
                          knowledge_type: str = "strategy_sharing") -> bool:
        """Transfer specific knowledge between living ants"""
        try:
            source_id = source_ant_data.get("ant_id")
            target_id = target_ant_data.get("ant_id")
            
            if knowledge_type == "strategy_sharing":
                # Share successful strategies
                source_strategies = source_ant_data.get("successful_strategies", [])
                target_strategies = target_ant_data.get("available_strategies", [])
                
                # Filter strategies not already known
                new_strategies = []
                for strategy in source_strategies:
                    if not any(s.get("name") == strategy.get("name") for s in target_strategies):
                        new_strategies.append(strategy)
                
                if new_strategies:
                    target_ant_data.setdefault("available_strategies", []).extend(new_strategies[:3])
                    
                    transfer = KnowledgeTransfer(
                        source_ant_id=source_id,
                        target_ant_id=target_id,
                        knowledge_type=knowledge_type,
                        knowledge_data={"strategies_shared": len(new_strategies)},
                        transfer_reason="peer_learning"
                    )
                    
                    self.knowledge_transfers.append(transfer)
                    
                    logger.info(f"Transferred {len(new_strategies)} strategies from {source_id} to {target_id}")
                    return True
                    
            elif knowledge_type == "market_insight":
                # Share market patterns and insights
                source_patterns = source_ant_data.get("learned_patterns", [])
                target_patterns = target_ant_data.setdefault("inherited_patterns", [])
                
                # Share top performing patterns
                for pattern in source_patterns[:5]:
                    if pattern.get("confidence", 0) > 0.7:
                        target_patterns.append(pattern)
                
                if source_patterns:
                    transfer = KnowledgeTransfer(
                        source_ant_id=source_id,
                        target_ant_id=target_id,
                        knowledge_type=knowledge_type,
                        knowledge_data={"patterns_shared": len(source_patterns[:5])},
                        transfer_reason="market_insight_sharing"
                    )
                    
                    self.knowledge_transfers.append(transfer)
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error transferring knowledge: {e}")
            return False
    
    def update_generation_performance(self, generation: int, performance_data: Dict[str, Any]):
        """Update performance statistics for a generation"""
        try:
            if generation not in self.generation_performance:
                self.generation_performance[generation] = {
                    "total_ants": 0,
                    "avg_performance": 0,
                    "best_performance": 0,
                    "total_capital": 0,
                    "successful_splits": 0,
                    "successful_merges": 0
                }
            
            gen_data = self.generation_performance[generation]
            gen_data["total_ants"] += 1
            
            # Update averages
            current_avg = gen_data["avg_performance"]
            new_performance = performance_data.get("performance_score", 0)
            gen_data["avg_performance"] = (current_avg + new_performance) / 2
            
            # Update best performance
            gen_data["best_performance"] = max(gen_data["best_performance"], new_performance)
            
            # Update capital
            gen_data["total_capital"] += performance_data.get("capital", 0)
            
        except Exception as e:
            logger.error(f"Error updating generation performance: {e}")
    
    def get_inheritance_statistics(self) -> Dict[str, Any]:
        """Get statistics about inheritance and knowledge transfer"""
        try:
            total_inheritances = len(self.inheritance_history)
            total_transfers = len(self.knowledge_transfers)
            
            if total_inheritances == 0:
                return {"total_inheritances": 0, "total_knowledge_transfers": 0}
            
            # Calculate average inheritance values
            avg_capital = sum(pkg.inherited_capital for pkg in self.inheritance_history) / total_inheritances
            avg_strategies = sum(len(pkg.inherited_strategies) for pkg in self.inheritance_history) / total_inheritances
            
            # Calculate generation statistics
            generations = list(self.generation_performance.keys())
            max_generation = max(generations) if generations else 0
            
            return {
                "total_inheritances": total_inheritances,
                "total_knowledge_transfers": total_transfers,
                "average_inherited_capital": avg_capital,
                "average_inherited_strategies": avg_strategies,
                "max_generation_reached": max_generation,
                "active_generations": len(generations),
                "recent_inheritances": len([pkg for pkg in self.inheritance_history 
                                          if time.time() - pkg.creation_timestamp < 3600])
            }
            
        except Exception as e:
            logger.error(f"Error calculating inheritance statistics: {e}")
            return {"error": str(e)} 