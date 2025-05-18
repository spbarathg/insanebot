from typing import List, Dict, Any
from .ant_princess import AntPrincess

class AntQueen:
    def __init__(self):
        self.ant_princesses: List[AntPrincess] = []
        self.global_experience_pool: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Dict[str, float]] = {}
    
    def register_ant_princess(self, ant_princess: AntPrincess):
        """Register a new Ant Princess with the Queen"""
        self.ant_princesses.append(ant_princess)
        self.performance_metrics[id(ant_princess)] = {}
    
    def collect_experience(self, ant_princess: AntPrincess, experience: Dict[str, Any]):
        """Collect experience from an Ant Princess"""
        self.global_experience_pool.update(experience)
        # Update performance metrics
        if "performance" in experience:
            self.performance_metrics[id(ant_princess)].update(experience["performance"])
    
    def distribute_experience(self):
        """Distribute collected experience to all Ant Princesses"""
        for ant_princess in self.ant_princesses:
            ant_princess.receive_experience(self.global_experience_pool)
    
    def analyze_colony_performance(self) -> Dict[str, Any]:
        """Analyze overall colony performance and identify areas for improvement"""
        performance_summary = {
            "average_performance": self._calculate_average_performance(),
            "best_performers": self._identify_best_performers(),
            "improvement_areas": self._identify_improvement_areas()
        }
        return performance_summary
    
    def _calculate_average_performance(self) -> float:
        """Calculate average performance across all Ant Princesses"""
        if not self.performance_metrics:
            return 0.0
        all_scores = [metrics.get("score", 0.0) for metrics in self.performance_metrics.values()]
        return sum(all_scores) / len(all_scores)
    
    def _identify_best_performers(self) -> List[Dict[str, Any]]:
        """Identify the best performing Ant Princesses"""
        sorted_princesses = sorted(
            self.performance_metrics.items(),
            key=lambda x: x[1].get("score", 0.0),
            reverse=True
        )
        return [{"id": id, "metrics": metrics} for id, metrics in sorted_princesses[:3]]
    
    def _identify_improvement_areas(self) -> List[str]:
        """Identify areas where the colony can improve"""
        # Implementation of improvement area identification
        return []
    
    def optimize_colony(self):
        """Optimize the colony based on performance analysis"""
        performance_summary = self.analyze_colony_performance()
        
        # Implement optimization strategies based on performance summary
        if performance_summary["average_performance"] < 0.7:
            self._trigger_colony_adaptation()
        
        # Share optimization insights with all Ant Princesses
        self.distribute_experience()
    
    def _trigger_colony_adaptation(self):
        """Trigger adaptation mechanisms in the colony"""
        # Implementation of colony adaptation logic
        pass 