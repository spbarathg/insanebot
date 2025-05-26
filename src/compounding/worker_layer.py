"""
Worker Layer - Worker Ant multiplication compounding

Implements worker compounding effects through worker multiplication,
efficiency improvements, and skill development that create exponential
workforce growth and capability enhancement.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import deque
import math

logger = logging.getLogger(__name__)

@dataclass
class WorkerMultiplicationRecord:
    """Represents a worker multiplication event"""
    record_id: str
    timestamp: float
    parent_worker_id: str
    new_worker_count: int
    multiplication_efficiency: float
    inherited_skills: Dict[str, float]
    success_probability: float

@dataclass
class WorkerEfficiencyMetrics:
    """Tracks worker efficiency compounding"""
    total_workers_created: int = 0
    average_worker_efficiency: float = 1.0
    efficiency_compound_rate: float = 1.02  # 2% efficiency growth
    skill_transfer_rate: float = 0.8  # 80% skill inheritance
    multiplication_success_rate: float = 0.75  # 75% success rate
    total_efficiency_gained: float = 0.0

class WorkerLayer:
    """Implements worker multiplication and efficiency compounding"""
    
    def __init__(self):
        # Worker compounding configuration
        self.base_multiplication_rate = 5  # Base 5 workers per split
        self.max_workers_per_split = 10
        self.min_workers_per_split = 2
        self.efficiency_decay_rate = 0.95  # 5% efficiency decay without training
        
        # Worker tracking
        self.metrics = WorkerEfficiencyMetrics()
        self.worker_multiplication_history: deque = deque(maxlen=1000)
        self.worker_efficiency_pools: Dict[str, float] = {}
        
        # Skill development tracking
        self.skill_categories = [
            "trading_accuracy", "risk_management", "market_analysis", 
            "execution_speed", "profit_optimization", "pattern_recognition"
        ]
        self.global_skill_levels: Dict[str, float] = {}
        self.skill_transfer_matrix: Dict[str, Dict[str, float]] = {}
        
        # Efficiency multiplication
        self.efficiency_multipliers: Dict[str, float] = {}
        self.worker_lineages: Dict[str, List[str]] = {}  # Track worker family trees
        self.generation_bonuses: Dict[int, float] = {}
        
        # Cycle management
        self.last_efficiency_update = 0.0
        self.efficiency_update_interval = 1800.0  # 30 minutes
        self.total_workers_managed = 0
        
        logger.info("WorkerLayer initialized")
    
    async def initialize(self) -> bool:
        """Initialize the worker compounding layer"""
        try:
            # Initialize skill systems
            await self._initialize_skill_systems()
            
            # Initialize efficiency tracking
            await self._initialize_efficiency_tracking()
            
            # Initialize worker lineage tracking
            await self._initialize_lineage_tracking()
            
            logger.info("WorkerLayer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize WorkerLayer: {e}")
            return False
    
    async def multiply_workers(
        self, 
        parent_worker_id: str,
        parent_efficiency: float,
        parent_skills: Dict[str, float],
        target_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """Multiply workers from a parent worker with skill inheritance"""
        try:
            # Calculate multiplication count
            multiplication_count = target_count or await self._calculate_multiplication_count(
                parent_efficiency, parent_skills
            )
            
            # Calculate inheritance efficiency
            inheritance_efficiency = await self._calculate_inheritance_efficiency(
                parent_worker_id, parent_skills
            )
            
            # Generate inherited skills for new workers
            inherited_skills = await self._generate_inherited_skills(
                parent_skills, inheritance_efficiency
            )
            
            # Create multiplication record
            record_id = f"mult_{int(time.time())}_{parent_worker_id}"
            multiplication_record = WorkerMultiplicationRecord(
                record_id=record_id,
                timestamp=time.time(),
                parent_worker_id=parent_worker_id,
                new_worker_count=multiplication_count,
                multiplication_efficiency=inheritance_efficiency,
                inherited_skills=inherited_skills,
                success_probability=self._calculate_multiplication_success_probability(
                    parent_efficiency, inheritance_efficiency
                )
            )
            
            self.worker_multiplication_history.append(multiplication_record)
            
            # Update metrics
            self.metrics.total_workers_created += multiplication_count
            self.total_workers_managed += multiplication_count
            
            # Update lineage tracking
            await self._update_worker_lineage(parent_worker_id, multiplication_count)
            
            # Apply efficiency compounding
            efficiency_gain = await self._apply_efficiency_compounding(
                multiplication_count, inheritance_efficiency
            )
            
            # Update global skill levels
            await self._update_global_skills(inherited_skills, multiplication_count)
            
            return {
                "multiplication_successful": True,
                "new_worker_count": multiplication_count,
                "inherited_skills": inherited_skills,
                "inheritance_efficiency": inheritance_efficiency,
                "efficiency_gain": efficiency_gain,
                "record_id": record_id
            }
            
        except Exception as e:
            logger.error(f"Error multiplying workers: {e}")
            return {
                "multiplication_successful": False,
                "error": str(e)
            }
    
    async def execute_worker_cycle(self) -> Dict[str, Any]:
        """Execute worker layer compounding cycle"""
        try:
            cycle_results = {
                "cycle_timestamp": time.time(),
                "efficiency_updates": 0,
                "skill_developments": 0,
                "generation_bonuses_applied": 0,
                "compound_efficiency_gain": 0.0
            }
            
            # Check if it's time for efficiency update cycle
            if time.time() - self.last_efficiency_update < self.efficiency_update_interval:
                return cycle_results
            
            # Update worker efficiencies
            efficiency_updates = await self._update_worker_efficiencies()
            cycle_results["efficiency_updates"] = efficiency_updates["workers_updated"]
            
            # Develop global skills
            skill_development = await self._develop_global_skills()
            cycle_results["skill_developments"] = skill_development["skills_developed"]
            
            # Apply generation bonuses
            generation_bonuses = await self._apply_generation_bonuses()
            cycle_results["generation_bonuses_applied"] = generation_bonuses["bonuses_applied"]
            
            # Calculate compound efficiency gain
            compound_gain = await self._calculate_compound_efficiency_gain()
            cycle_results["compound_efficiency_gain"] = compound_gain
            
            self.last_efficiency_update = time.time()
            
            return cycle_results
            
        except Exception as e:
            logger.error(f"Error in worker cycle: {e}")
            return {"error": str(e)}
    
    async def _calculate_multiplication_count(
        self, 
        parent_efficiency: float, 
        parent_skills: Dict[str, float]
    ) -> int:
        """Calculate optimal number of workers to create"""
        try:
            # Base multiplication from configuration
            base_count = self.base_multiplication_rate
            
            # Efficiency bonus (higher efficiency = more workers)
            efficiency_bonus = max(1.0, parent_efficiency) - 1.0
            efficiency_multiplier = 1.0 + (efficiency_bonus * 0.5)  # Up to 50% bonus
            
            # Skill bonus (average skill level bonus)
            if parent_skills:
                avg_skill_level = sum(parent_skills.values()) / len(parent_skills)
                skill_multiplier = 1.0 + (avg_skill_level - 1.0) * 0.3  # Up to 30% bonus
            else:
                skill_multiplier = 1.0
            
            # Generation bonus (later generations are more efficient)
            generation_bonus = 1.0  # Placeholder, would calculate from lineage
            
            # Calculate final count
            multiplication_count = base_count * efficiency_multiplier * skill_multiplier * generation_bonus
            
            # Apply bounds
            multiplication_count = max(
                self.min_workers_per_split, 
                min(self.max_workers_per_split, int(multiplication_count))
            )
            
            return multiplication_count
            
        except Exception as e:
            logger.error(f"Error calculating multiplication count: {e}")
            return self.base_multiplication_rate
    
    async def _calculate_inheritance_efficiency(
        self, 
        parent_worker_id: str, 
        parent_skills: Dict[str, float]
    ) -> float:
        """Calculate efficiency of skill inheritance"""
        try:
            # Base inheritance rate
            base_efficiency = self.metrics.skill_transfer_rate
            
            # Parent skill quality bonus
            if parent_skills:
                skill_quality = sum(parent_skills.values()) / len(parent_skills)
                quality_bonus = min(0.2, (skill_quality - 1.0) * 0.1)  # Up to 20% bonus
                base_efficiency += quality_bonus
            
            # Lineage depth bonus (experienced bloodlines transfer better)
            lineage_depth = len(self.worker_lineages.get(parent_worker_id, []))
            lineage_bonus = min(0.15, lineage_depth * 0.02)  # Up to 15% bonus
            base_efficiency += lineage_bonus
            
            # Global skill pool bonus
            global_skill_bonus = self._calculate_global_skill_bonus()
            base_efficiency += global_skill_bonus
            
            return min(1.0, base_efficiency)  # Cap at 100%
            
        except Exception as e:
            logger.error(f"Error calculating inheritance efficiency: {e}")
            return self.metrics.skill_transfer_rate
    
    async def _generate_inherited_skills(
        self, 
        parent_skills: Dict[str, float], 
        inheritance_efficiency: float
    ) -> Dict[str, float]:
        """Generate inherited skills for new workers"""
        try:
            inherited_skills = {}
            
            for skill, level in parent_skills.items():
                # Apply inheritance efficiency
                inherited_level = level * inheritance_efficiency
                
                # Add small random variation
                variation = 0.95 + (0.1 * (time.time() % 1))  # Simple randomness
                inherited_level *= variation
                
                # Add global skill pool contribution
                global_contribution = self.global_skill_levels.get(skill, 1.0) * 0.1
                inherited_level += global_contribution
                
                # Ensure minimum competency
                inherited_level = max(0.5, inherited_level)
                
                inherited_skills[skill] = inherited_level
            
            # Add any missing core skills
            for skill in self.skill_categories:
                if skill not in inherited_skills:
                    # New workers get basic level in missing skills
                    basic_level = 0.5 + (self.global_skill_levels.get(skill, 1.0) * 0.2)
                    inherited_skills[skill] = basic_level
            
            return inherited_skills
            
        except Exception as e:
            logger.error(f"Error generating inherited skills: {e}")
            return {skill: 1.0 for skill in self.skill_categories}
    
    def _calculate_multiplication_success_probability(
        self, 
        parent_efficiency: float, 
        inheritance_efficiency: float
    ) -> float:
        """Calculate probability of successful worker multiplication"""
        # Base success rate
        base_probability = self.metrics.multiplication_success_rate
        
        # Efficiency bonuses
        efficiency_bonus = (parent_efficiency - 1.0) * 0.1  # 10% per efficiency point
        inheritance_bonus = (inheritance_efficiency - 0.5) * 0.2  # 20% per inheritance point
        
        # System experience bonus
        experience_bonus = min(0.2, self.metrics.total_workers_created * 0.01)  # Up to 20%
        
        total_probability = base_probability + efficiency_bonus + inheritance_bonus + experience_bonus
        return min(1.0, max(0.1, total_probability))  # Keep between 10% and 100%
    
    async def _apply_efficiency_compounding(
        self, 
        worker_count: int, 
        inheritance_efficiency: float
    ) -> float:
        """Apply efficiency compounding from worker multiplication"""
        try:
            # Base efficiency gain from creating workers
            base_gain = worker_count * 0.01  # 1% per worker
            
            # Inheritance efficiency multiplier
            inheritance_multiplier = 1.0 + (inheritance_efficiency - 0.5)
            
            # Compound rate application
            compound_multiplier = self.metrics.efficiency_compound_rate
            
            # Calculate total efficiency gain
            efficiency_gain = base_gain * inheritance_multiplier * compound_multiplier
            
            # Apply to metrics
            self.metrics.total_efficiency_gained += efficiency_gain
            self.metrics.average_worker_efficiency += efficiency_gain * 0.1  # 10% contribution
            
            return efficiency_gain
            
        except Exception as e:
            logger.error(f"Error applying efficiency compounding: {e}")
            return 0.0
    
    async def _update_worker_lineage(self, parent_worker_id: str, new_worker_count: int):
        """Update worker lineage tracking"""
        try:
            if parent_worker_id not in self.worker_lineages:
                self.worker_lineages[parent_worker_id] = []
            
            # Add new workers to lineage
            for i in range(new_worker_count):
                new_worker_id = f"{parent_worker_id}_child_{i}_{int(time.time())}"
                self.worker_lineages[parent_worker_id].append(new_worker_id)
                
                # Initialize lineage for new worker
                self.worker_lineages[new_worker_id] = []
            
        except Exception as e:
            logger.error(f"Error updating worker lineage: {e}")
    
    async def _update_global_skills(self, inherited_skills: Dict[str, float], worker_count: int):
        """Update global skill levels based on new workers"""
        try:
            for skill, level in inherited_skills.items():
                if skill not in self.global_skill_levels:
                    self.global_skill_levels[skill] = 1.0
                
                # Weighted update based on worker count and skill level
                contribution = (level * worker_count) * 0.001  # Small contribution per worker
                self.global_skill_levels[skill] += contribution
                
                # Apply skill decay to prevent infinite growth
                self.global_skill_levels[skill] *= 0.999
                
                # Keep skills within reasonable bounds
                self.global_skill_levels[skill] = max(0.5, min(3.0, self.global_skill_levels[skill]))
            
        except Exception as e:
            logger.error(f"Error updating global skills: {e}")
    
    async def _update_worker_efficiencies(self) -> Dict[str, Any]:
        """Update worker efficiency pools and multipliers"""
        try:
            update_results = {
                "workers_updated": 0,
                "efficiency_improvements": 0,
                "skill_transfers": 0
            }
            
            # Update efficiency pools
            for worker_id, efficiency in self.worker_efficiency_pools.items():
                # Apply efficiency decay
                decayed_efficiency = efficiency * self.efficiency_decay_rate
                
                # Apply skill-based improvements
                skill_bonus = self._calculate_skill_based_efficiency_bonus(worker_id)
                improved_efficiency = decayed_efficiency + skill_bonus
                
                self.worker_efficiency_pools[worker_id] = improved_efficiency
                update_results["workers_updated"] += 1
                
                if skill_bonus > 0:
                    update_results["efficiency_improvements"] += 1
            
            return update_results
            
        except Exception as e:
            logger.error(f"Error updating worker efficiencies: {e}")
            return {"workers_updated": 0}
    
    def _calculate_skill_based_efficiency_bonus(self, worker_id: str) -> float:
        """Calculate efficiency bonus based on worker skills and lineage"""
        # Base bonus from global skill levels
        avg_global_skill = sum(self.global_skill_levels.values()) / max(1, len(self.global_skill_levels))
        global_bonus = (avg_global_skill - 1.0) * 0.05  # 5% per skill point above 1.0
        
        # Lineage bonus
        lineage_bonus = len(self.worker_lineages.get(worker_id, [])) * 0.01  # 1% per descendant
        
        return max(0.0, global_bonus + lineage_bonus)
    
    async def _develop_global_skills(self) -> Dict[str, Any]:
        """Develop global skill levels through collective learning"""
        try:
            development_results = {
                "skills_developed": 0,
                "total_skill_improvement": 0.0
            }
            
            for skill in self.skill_categories:
                if skill not in self.global_skill_levels:
                    self.global_skill_levels[skill] = 1.0
                
                # Calculate development based on worker activity
                worker_activity_factor = min(1.0, self.total_workers_managed / 100.0)
                skill_development = worker_activity_factor * 0.01  # 1% development per 100 workers
                
                # Apply compound learning
                compound_development = skill_development * self.metrics.efficiency_compound_rate
                
                self.global_skill_levels[skill] += compound_development
                development_results["total_skill_improvement"] += compound_development
                development_results["skills_developed"] += 1
            
            return development_results
            
        except Exception as e:
            logger.error(f"Error developing global skills: {e}")
            return {"skills_developed": 0}
    
    async def _apply_generation_bonuses(self) -> Dict[str, Any]:
        """Apply bonuses based on worker generations"""
        try:
            bonus_results = {
                "bonuses_applied": 0,
                "total_bonus_value": 0.0
            }
            
            # Calculate generation depths
            generation_counts = {}
            for worker_id, children in self.worker_lineages.items():
                generation = self._calculate_generation_depth(worker_id)
                if generation not in generation_counts:
                    generation_counts[generation] = 0
                generation_counts[generation] += len(children)
            
            # Apply generation bonuses
            for generation, worker_count in generation_counts.items():
                if generation > 0:  # Skip generation 0 (original workers)
                    bonus_multiplier = 1.0 + (generation * 0.05)  # 5% per generation
                    bonus_value = worker_count * bonus_multiplier * 0.01  # 1% base bonus
                    
                    self.generation_bonuses[generation] = bonus_value
                    bonus_results["bonuses_applied"] += 1
                    bonus_results["total_bonus_value"] += bonus_value
            
            return bonus_results
            
        except Exception as e:
            logger.error(f"Error applying generation bonuses: {e}")
            return {"bonuses_applied": 0}
    
    def _calculate_generation_depth(self, worker_id: str) -> int:
        """Calculate generation depth of a worker"""
        # Simple generation calculation based on ID structure
        if "_child_" in worker_id:
            return worker_id.count("_child_")
        return 0
    
    async def _calculate_compound_efficiency_gain(self) -> float:
        """Calculate total compound efficiency gain"""
        try:
            # Base efficiency from worker count
            worker_count_efficiency = min(2.0, self.total_workers_managed / 100.0)
            
            # Skill development efficiency
            avg_skill_efficiency = sum(self.global_skill_levels.values()) / max(1, len(self.global_skill_levels))
            
            # Generation bonus efficiency
            generation_efficiency = sum(self.generation_bonuses.values())
            
            # Compound all efficiencies
            compound_efficiency = (
                worker_count_efficiency * 
                avg_skill_efficiency * 
                (1.0 + generation_efficiency)
            )
            
            # Update metrics
            self.metrics.average_worker_efficiency = compound_efficiency
            
            return compound_efficiency
            
        except Exception as e:
            logger.error(f"Error calculating compound efficiency gain: {e}")
            return 1.0
    
    def _calculate_global_skill_bonus(self) -> float:
        """Calculate bonus from global skill pool"""
        if not self.global_skill_levels:
            return 0.0
        
        avg_skill = sum(self.global_skill_levels.values()) / len(self.global_skill_levels)
        return min(0.1, (avg_skill - 1.0) * 0.05)  # Up to 10% bonus
    
    async def _initialize_skill_systems(self):
        """Initialize skill development systems"""
        # Initialize global skill levels
        self.global_skill_levels = {skill: 1.0 for skill in self.skill_categories}
        
        # Initialize skill transfer matrix (how skills reinforce each other)
        self.skill_transfer_matrix = {}
        for skill in self.skill_categories:
            self.skill_transfer_matrix[skill] = {
                other_skill: 0.1 if other_skill != skill else 1.0
                for other_skill in self.skill_categories
            }
    
    async def _initialize_efficiency_tracking(self):
        """Initialize efficiency tracking systems"""
        self.worker_efficiency_pools = {}
        self.efficiency_multipliers = {}
    
    async def _initialize_lineage_tracking(self):
        """Initialize worker lineage tracking"""
        self.worker_lineages = {}
        self.generation_bonuses = {}
    
    def get_worker_summary(self) -> Dict[str, Any]:
        """Get comprehensive worker layer summary"""
        return {
            "worker_metrics": {
                "total_workers_created": self.metrics.total_workers_created,
                "average_worker_efficiency": self.metrics.average_worker_efficiency,
                "efficiency_compound_rate": self.metrics.efficiency_compound_rate,
                "skill_transfer_rate": self.metrics.skill_transfer_rate,
                "multiplication_success_rate": self.metrics.multiplication_success_rate,
                "total_efficiency_gained": self.metrics.total_efficiency_gained
            },
            "global_skills": self.global_skill_levels.copy(),
            "lineage_tracking": {
                "total_lineages": len(self.worker_lineages),
                "generation_bonuses": self.generation_bonuses.copy(),
                "total_workers_managed": self.total_workers_managed
            },
            "efficiency_pools": {
                "active_pools": len(self.worker_efficiency_pools),
                "efficiency_multipliers": len(self.efficiency_multipliers)
            }
        }
    
    def get_worker_efficiency_for_component(self, component: str) -> float:
        """Get current worker efficiency for a specific component"""
        return self.worker_efficiency_pools.get(component, self.metrics.average_worker_efficiency)
    
    async def cleanup(self):
        """Cleanup worker layer resources"""
        try:
            # Clear large data structures
            self.worker_multiplication_history.clear()
            self.worker_lineages.clear()
            
            logger.info("WorkerLayer cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during WorkerLayer cleanup: {e}") 