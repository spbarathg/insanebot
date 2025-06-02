#!/usr/bin/env python3
"""
Smart API Scaling Manager

Automatically monitors trading performance and suggests/enables API upgrades
when they become cost-efficient based on actual profit generation.

This ensures maximum ROI while scaling infrastructure costs intelligently.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class APIService:
    """API service configuration"""
    name: str
    current_tier: str
    monthly_cost: float
    upgrade_tier: str
    upgrade_cost: float
    signal_weight: float
    performance_impact: float
    required_monthly_profit: float

@dataclass
class PerformanceMetrics:
    """Trading performance metrics"""
    total_profit_usd: float
    monthly_profit_usd: float
    win_rate: float
    total_trades: int
    successful_trades: int
    avg_profit_per_trade: float
    api_costs_monthly: float
    net_profit_monthly: float
    roi_percentage: float

class SmartAPIScaler:
    """
    Intelligent API scaling based on actual trading performance and ROI
    """
    
    def __init__(self):
        self.api_services = {
            'helius': APIService(
                name='Helius',
                current_tier='free',
                monthly_cost=0,
                upgrade_tier='pro',
                upgrade_cost=29,
                signal_weight=0.30,  # 30% of signal comes from technical analysis (via Helius)
                performance_impact=0.15,  # 15% performance boost expected
                required_monthly_profit=100  # Need $100/month profit to justify
            ),
            'quicknode': APIService(
                name='QuickNode',
                current_tier='free',
                monthly_cost=0,
                upgrade_tier='build',
                upgrade_cost=9,
                signal_weight=0.05,  # Backup RPC, minimal signal impact
                performance_impact=0.05,  # 5% reliability boost
                required_monthly_profit=30  # Need $30/month profit to justify
            ),
            'twitter': APIService(
                name='Twitter',
                current_tier='free',
                monthly_cost=0,
                upgrade_tier='basic',
                upgrade_cost=100,
                signal_weight=0.15,  # 15% signal weight for social sentiment
                performance_impact=0.10,  # 10% performance boost from better sentiment
                required_monthly_profit=300  # Need $300/month profit to justify
            ),
            'grok': APIService(
                name='Grok AI',
                current_tier='disabled',
                monthly_cost=0,
                upgrade_tier='premium',
                upgrade_cost=16,
                signal_weight=0.05,  # Only 5% signal weight for AI analysis
                performance_impact=0.03,  # 3% performance boost
                required_monthly_profit=60  # Need $60/month profit to justify
            )
        }
        
        self.scaling_thresholds = {
            'conservative': 3.0,  # 3x cost in monthly profit to upgrade
            'balanced': 2.0,      # 2x cost in monthly profit to upgrade
            'aggressive': 1.5     # 1.5x cost in monthly profit to upgrade
        }
        
        self.current_strategy = 'balanced'  # Default scaling strategy
        
        # Performance tracking
        self.performance_history: List[PerformanceMetrics] = []
        self.last_performance_check = time.time()
        
    async def analyze_scaling_opportunities(self) -> Dict[str, any]:
        """Analyze current performance and suggest scaling opportunities"""
        try:
            # Get current performance metrics
            current_metrics = await self._get_current_performance()
            
            if not current_metrics:
                return {"error": "Unable to retrieve performance metrics"}
            
            # Calculate scaling recommendations
            recommendations = await self._calculate_scaling_recommendations(current_metrics)
            
            # Generate upgrade suggestions
            upgrade_suggestions = await self._generate_upgrade_suggestions(current_metrics, recommendations)
            
            # Calculate ROI projections
            roi_projections = await self._calculate_roi_projections(current_metrics, recommendations)
            
            return {
                "current_performance": asdict(current_metrics),
                "scaling_recommendations": recommendations,
                "upgrade_suggestions": upgrade_suggestions,
                "roi_projections": roi_projections,
                "next_check_in": 3600,  # Check again in 1 hour
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing scaling opportunities: {e}")
            return {"error": str(e)}
    
    async def _get_current_performance(self) -> Optional[PerformanceMetrics]:
        """Get current trading performance metrics"""
        try:
            # This would read from actual performance logs/database
            # For now, simulating with sample data structure
            
            performance_file = "logs/performance_metrics.json"
            if os.path.exists(performance_file):
                with open(performance_file, 'r') as f:
                    data = json.load(f)
                
                return PerformanceMetrics(
                    total_profit_usd=data.get('total_profit_usd', 0),
                    monthly_profit_usd=data.get('monthly_profit_usd', 0),
                    win_rate=data.get('win_rate', 0),
                    total_trades=data.get('total_trades', 0),
                    successful_trades=data.get('successful_trades', 0),
                    avg_profit_per_trade=data.get('avg_profit_per_trade', 0),
                    api_costs_monthly=data.get('api_costs_monthly', 0),
                    net_profit_monthly=data.get('net_profit_monthly', 0),
                    roi_percentage=data.get('roi_percentage', 0)
                )
            else:
                # Create sample performance for demonstration
                return PerformanceMetrics(
                    total_profit_usd=0,
                    monthly_profit_usd=0,
                    win_rate=0,
                    total_trades=0,
                    successful_trades=0,
                    avg_profit_per_trade=0,
                    api_costs_monthly=10,  # SOL gas fees only
                    net_profit_monthly=0,
                    roi_percentage=0
                )
                
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return None
    
    async def _calculate_scaling_recommendations(self, metrics: PerformanceMetrics) -> Dict[str, any]:
        """Calculate which APIs should be upgraded based on performance"""
        recommendations = {}
        
        threshold_multiplier = self.scaling_thresholds[self.current_strategy]
        
        for service_id, service in self.api_services.items():
            if service.current_tier == 'free' or service.current_tier == 'disabled':
                required_profit = service.upgrade_cost * threshold_multiplier
                
                can_afford = metrics.monthly_profit_usd >= required_profit
                roi_positive = (service.performance_impact * metrics.monthly_profit_usd) > service.upgrade_cost
                
                recommendations[service_id] = {
                    'current_tier': service.current_tier,
                    'suggested_tier': service.upgrade_tier if (can_afford or roi_positive) else service.current_tier,
                    'monthly_cost': service.upgrade_cost if (can_afford or roi_positive) else service.monthly_cost,
                    'can_afford': can_afford,
                    'roi_positive': roi_positive,
                    'required_monthly_profit': required_profit,
                    'current_monthly_profit': metrics.monthly_profit_usd,
                    'performance_impact': service.performance_impact,
                    'priority': self._calculate_upgrade_priority(service, metrics)
                }
        
        return recommendations
    
    def _calculate_upgrade_priority(self, service: APIService, metrics: PerformanceMetrics) -> str:
        """Calculate upgrade priority based on ROI and impact"""
        if metrics.monthly_profit_usd == 0:
            return 'wait'
        
        roi_ratio = (service.performance_impact * metrics.monthly_profit_usd) / service.upgrade_cost
        
        if roi_ratio >= 2.0:
            return 'high'
        elif roi_ratio >= 1.5:
            return 'medium'
        elif roi_ratio >= 1.0:
            return 'low'
        else:
            return 'wait'
    
    async def _generate_upgrade_suggestions(self, metrics: PerformanceMetrics, recommendations: Dict) -> List[Dict]:
        """Generate specific upgrade suggestions with justification"""
        suggestions = []
        
        # Sort by priority and ROI
        sorted_services = sorted(
            [(k, v) for k, v in recommendations.items()],
            key=lambda x: (x[1]['priority'] == 'high', x[1]['roi_positive'], -x[1]['monthly_cost'])
        )
        
        for service_id, rec in sorted_services:
            service = self.api_services[service_id]
            
            if rec['suggested_tier'] != rec['current_tier']:
                suggestions.append({
                    'service': service.name,
                    'action': f"Upgrade from {rec['current_tier']} to {rec['suggested_tier']}",
                    'monthly_cost': rec['monthly_cost'],
                    'expected_performance_boost': f"{service.performance_impact*100:.1f}%",
                    'justification': self._generate_justification(service, rec, metrics),
                    'priority': rec['priority'],
                    'immediate_action': rec['priority'] in ['high', 'medium'] and rec['roi_positive']
                })
        
        return suggestions
    
    def _generate_justification(self, service: APIService, rec: Dict, metrics: PerformanceMetrics) -> str:
        """Generate human-readable justification for upgrade"""
        if rec['roi_positive']:
            expected_additional_profit = service.performance_impact * metrics.monthly_profit_usd
            roi_months = service.upgrade_cost / expected_additional_profit if expected_additional_profit > 0 else float('inf')
            return f"Expected ${expected_additional_profit:.0f}/month additional profit. ROI break-even in {roi_months:.1f} months."
        elif rec['can_afford']:
            return f"Can afford upgrade. Currently profitable enough to support ${rec['monthly_cost']}/month cost."
        else:
            return f"Wait until monthly profit reaches ${rec['required_monthly_profit']:.0f} before upgrading."
    
    async def _calculate_roi_projections(self, metrics: PerformanceMetrics, recommendations: Dict) -> Dict:
        """Calculate ROI projections for different upgrade scenarios"""
        scenarios = {}
        
        # Current scenario (no upgrades)
        scenarios['current'] = {
            'monthly_costs': sum(service.monthly_cost for service in self.api_services.values()),
            'expected_performance': 1.0,
            'projected_monthly_profit': metrics.monthly_profit_usd,
            'net_profit': metrics.monthly_profit_usd - sum(service.monthly_cost for service in self.api_services.values())
        }
        
        # Recommended upgrades only
        recommended_cost = sum(rec['monthly_cost'] for rec in recommendations.values() if rec['suggested_tier'] != rec['current_tier'])
        recommended_performance_boost = sum(
            self.api_services[service_id].performance_impact 
            for service_id, rec in recommendations.items() 
            if rec['suggested_tier'] != rec['current_tier']
        )
        
        scenarios['recommended'] = {
            'monthly_costs': recommended_cost,
            'expected_performance': 1.0 + recommended_performance_boost,
            'projected_monthly_profit': metrics.monthly_profit_usd * (1.0 + recommended_performance_boost),
            'net_profit': metrics.monthly_profit_usd * (1.0 + recommended_performance_boost) - recommended_cost
        }
        
        # All upgrades scenario
        all_upgrades_cost = sum(service.upgrade_cost for service in self.api_services.values())
        all_upgrades_performance = sum(service.performance_impact for service in self.api_services.values())
        
        scenarios['all_upgrades'] = {
            'monthly_costs': all_upgrades_cost,
            'expected_performance': 1.0 + all_upgrades_performance,
            'projected_monthly_profit': metrics.monthly_profit_usd * (1.0 + all_upgrades_performance),
            'net_profit': metrics.monthly_profit_usd * (1.0 + all_upgrades_performance) - all_upgrades_cost
        }
        
        return scenarios
    
    async def auto_upgrade_apis(self, dry_run: bool = True) -> Dict:
        """Automatically upgrade APIs based on performance (with safety checks)"""
        try:
            analysis = await self.analyze_scaling_opportunities()
            
            if 'error' in analysis:
                return analysis
            
            upgrades_made = []
            
            for suggestion in analysis['upgrade_suggestions']:
                if suggestion['immediate_action'] and suggestion['priority'] == 'high':
                    if not dry_run:
                        # Here you would actually update environment variables
                        # For now, just log the action
                        logger.info(f"AUTO-UPGRADE: {suggestion['action']} - {suggestion['justification']}")
                        upgrades_made.append(suggestion)
                    else:
                        logger.info(f"DRY-RUN: Would upgrade {suggestion['service']}")
                        upgrades_made.append(suggestion)
            
            return {
                'upgrades_made': upgrades_made,
                'total_new_monthly_cost': sum(u['monthly_cost'] for u in upgrades_made),
                'dry_run': dry_run,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in auto-upgrade: {e}")
            return {"error": str(e)}
    
    def update_performance_metrics(self, new_metrics: Dict):
        """Update performance metrics from trading bot"""
        try:
            # Save to file for persistence
            performance_file = "logs/performance_metrics.json"
            os.makedirs(os.path.dirname(performance_file), exist_ok=True)
            
            with open(performance_file, 'w') as f:
                json.dump(new_metrics, f, indent=2)
            
            logger.info("Performance metrics updated")
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

async def main():
    """Main function for testing the smart scaler"""
    scaler = SmartAPIScaler()
    
    print("🎯 Smart API Scaling Analysis")
    print("=" * 50)
    
    # Analyze current situation
    analysis = await scaler.analyze_scaling_opportunities()
    
    if 'error' in analysis:
        print(f"Error: {analysis['error']}")
        return
    
    print(f"📊 Current Performance:")
    metrics = analysis['current_performance']
    print(f"   Monthly Profit: ${metrics['monthly_profit_usd']:.2f}")
    print(f"   API Costs: ${metrics['api_costs_monthly']:.2f}")
    print(f"   Net Profit: ${metrics['net_profit_monthly']:.2f}")
    print(f"   Win Rate: {metrics['win_rate']:.1%}")
    
    print(f"\n💡 Upgrade Suggestions:")
    for suggestion in analysis['upgrade_suggestions']:
        priority_emoji = {"high": "🔥", "medium": "⚡", "low": "💡", "wait": "⏳"}
        print(f"   {priority_emoji.get(suggestion['priority'], '💡')} {suggestion['action']}")
        print(f"      Cost: ${suggestion['monthly_cost']}/month")
        print(f"      Impact: {suggestion['expected_performance_boost']} performance boost")
        print(f"      Justification: {suggestion['justification']}")
        print()
    
    print(f"📈 ROI Projections:")
    for scenario, data in analysis['roi_projections'].items():
        print(f"   {scenario.title()}: ${data['net_profit']:.2f}/month net profit")
    
    # Test auto-upgrade (dry run)
    print(f"\n🤖 Auto-Upgrade Analysis (Dry Run):")
    auto_result = await scaler.auto_upgrade_apis(dry_run=True)
    
    if auto_result.get('upgrades_made'):
        print(f"   Would make {len(auto_result['upgrades_made'])} upgrades")
        for upgrade in auto_result['upgrades_made']:
            print(f"   - {upgrade['action']}")
    else:
        print("   No automatic upgrades recommended at this time")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main()) 