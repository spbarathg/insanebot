#!/usr/bin/env python3
"""
AI-Powered Trading Bot Analyzer

This tool analyzes comprehensive bot logs to provide intelligent insights,
recommendations, and code improvements. It acts as an AI assistant that
can understand what your bot is doing and suggest optimizations.

Features:
- Pattern recognition in bot behavior
- Performance optimization suggestions
- Error pattern analysis and solutions
- Trading strategy evaluation
- Code improvement recommendations
- Automated issue detection
- Behavioral anomaly detection
- Configuration optimization

Usage:
    python monitoring/bot_analyzer_ai.py --analyze-recent
    python monitoring/bot_analyzer_ai.py --full-analysis
    python monitoring/bot_analyzer_ai.py --live-monitoring
"""

import argparse
import asyncio
import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

console = Console()

@dataclass
class BotBehaviorAnalysis:
    """Comprehensive bot behavior analysis results"""
    timestamp: float
    analysis_period: str
    performance_score: float  # 0-100
    efficiency_score: float   # 0-100
    reliability_score: float  # 0-100
    
    # Function analysis
    function_performance: Dict[str, Any]
    slow_functions: List[Dict[str, Any]]
    error_prone_functions: List[Dict[str, Any]]
    
    # Decision analysis
    decision_quality: Dict[str, Any]
    low_confidence_decisions: List[Dict[str, Any]]
    decision_patterns: Dict[str, Any]
    
    # API analysis
    api_performance: Dict[str, Any]
    failing_apis: List[Dict[str, Any]]
    
    # Error analysis
    error_patterns: Dict[str, Any]
    critical_errors: List[Dict[str, Any]]
    
    # Recommendations
    immediate_actions: List[str]
    optimization_suggestions: List[str]
    code_improvements: List[str]
    configuration_changes: List[str]

class BotAnalyzerAI:
    """AI-powered trading bot analyzer"""
    
    def __init__(self):
        self.console = Console()
        self.db_path = Path("data/comprehensive_logs.db")
        
        # Analysis thresholds
        self.performance_thresholds = {
            'slow_function_ms': 1000,
            'very_slow_function_ms': 5000,
            'high_error_rate': 0.1,
            'low_confidence': 0.5,
            'slow_api_ms': 2000
        }
        
        # Pattern recognition weights
        self.pattern_weights = {
            'performance': 0.4,
            'reliability': 0.3,
            'efficiency': 0.3
        }
        
    async def analyze_bot_behavior(self, hours_back: int = 24) -> BotBehaviorAnalysis:
        """Perform comprehensive bot behavior analysis"""
        console.print(f"[bold blue]🤖 Analyzing bot behavior (last {hours_back} hours)[/bold blue]")
        
        # Connect to database
        if not self.db_path.exists():
            console.print("[red]❌ No comprehensive logs database found. Run with comprehensive logging first.[/red]")
            return None
            
        conn = sqlite3.connect(str(self.db_path))
        start_time = time.time() - (hours_back * 3600)
        
        # Analyze different aspects
        function_analysis = await self._analyze_function_performance(conn, start_time)
        decision_analysis = await self._analyze_decision_quality(conn, start_time)
        api_analysis = await self._analyze_api_performance(conn, start_time)
        error_analysis = await self._analyze_error_patterns(conn, start_time)
        
        # Calculate overall scores
        performance_score = self._calculate_performance_score(function_analysis, api_analysis)
        efficiency_score = self._calculate_efficiency_score(function_analysis, decision_analysis)
        reliability_score = self._calculate_reliability_score(error_analysis, api_analysis)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(
            function_analysis, decision_analysis, api_analysis, error_analysis
        )
        
        conn.close()
        
        return BotBehaviorAnalysis(
            timestamp=time.time(),
            analysis_period=f"last_{hours_back}_hours",
            performance_score=performance_score,
            efficiency_score=efficiency_score,
            reliability_score=reliability_score,
            function_performance=function_analysis,
            slow_functions=function_analysis.get('slow_functions', []),
            error_prone_functions=function_analysis.get('error_prone_functions', []),
            decision_quality=decision_analysis,
            low_confidence_decisions=decision_analysis.get('low_confidence_decisions', []),
            decision_patterns=decision_analysis.get('patterns', {}),
            api_performance=api_analysis,
            failing_apis=api_analysis.get('failing_apis', []),
            error_patterns=error_analysis,
            critical_errors=error_analysis.get('critical_errors', []),
            immediate_actions=recommendations['immediate'],
            optimization_suggestions=recommendations['optimization'],
            code_improvements=recommendations['code'],
            configuration_changes=recommendations['config']
        )
        
    async def _analyze_function_performance(self, conn: sqlite3.Connection, start_time: float) -> Dict[str, Any]:
        """Analyze function call performance patterns"""
        cursor = conn.cursor()
        
        # Get function performance statistics
        cursor.execute("""
            SELECT function_name, module_name, class_name,
                   COUNT(*) as call_count,
                   AVG(execution_time_ms) as avg_time,
                   MAX(execution_time_ms) as max_time,
                   MIN(execution_time_ms) as min_time,
                   SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
                   AVG(memory_usage_mb) as avg_memory,
                   AVG(cpu_usage_percent) as avg_cpu
            FROM function_calls 
            WHERE timestamp > ?
            GROUP BY function_name, module_name, class_name
            ORDER BY avg_time DESC
        """, (start_time,))
        
        function_stats = cursor.fetchall()
        
        # Analyze performance patterns
        slow_functions = []
        error_prone_functions = []
        memory_intensive_functions = []
        
        for stats in function_stats:
            (func_name, module, class_name, call_count, avg_time, max_time, 
             min_time, success_count, avg_memory, avg_cpu) = stats
            
            success_rate = success_count / call_count if call_count > 0 else 0
            
            # Identify slow functions
            if avg_time > self.performance_thresholds['slow_function_ms']:
                slow_functions.append({
                    'function': f"{module}.{class_name or ''}.{func_name}".replace('..', '.'),
                    'avg_execution_time_ms': avg_time,
                    'max_execution_time_ms': max_time,
                    'call_count': call_count,
                    'performance_impact': call_count * avg_time / 1000  # Total seconds spent
                })
                
            # Identify error-prone functions
            if success_rate < (1 - self.performance_thresholds['high_error_rate']):
                error_prone_functions.append({
                    'function': f"{module}.{class_name or ''}.{func_name}".replace('..', '.'),
                    'success_rate': success_rate,
                    'call_count': call_count,
                    'failure_count': call_count - success_count
                })
                
            # Identify memory-intensive functions
            if avg_memory and avg_memory > 10:  # More than 10MB average
                memory_intensive_functions.append({
                    'function': f"{module}.{class_name or ''}.{func_name}".replace('..', '.'),
                    'avg_memory_usage_mb': avg_memory,
                    'call_count': call_count
                })
        
        return {
            'total_functions_analyzed': len(function_stats),
            'slow_functions': sorted(slow_functions, key=lambda x: x['performance_impact'], reverse=True)[:10],
            'error_prone_functions': sorted(error_prone_functions, key=lambda x: x['failure_count'], reverse=True)[:10],
            'memory_intensive_functions': memory_intensive_functions[:5],
            'performance_summary': {
                'avg_execution_time': sum(s[4] for s in function_stats) / len(function_stats) if function_stats else 0,
                'total_function_calls': sum(s[3] for s in function_stats),
                'overall_success_rate': sum(s[7] for s in function_stats) / sum(s[3] for s in function_stats) if function_stats else 0
            }
        }
        
    async def _analyze_decision_quality(self, conn: sqlite3.Connection, start_time: float) -> Dict[str, Any]:
        """Analyze decision point quality and patterns"""
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT decision_type, confidence, reasoning, inputs, output,
                   context, alternatives_considered, risk_factors
            FROM decision_points 
            WHERE timestamp > ?
            ORDER BY timestamp DESC
        """, (start_time,))
        
        decisions = cursor.fetchall()
        
        if not decisions:
            return {'status': 'no_decisions', 'message': 'No decision points found'}
        
        # Analyze decision patterns
        decision_types = Counter(d[0] for d in decisions)
        confidence_levels = [d[1] for d in decisions if d[1] is not None]
        
        # Find low confidence decisions
        low_confidence_decisions = []
        for decision in decisions:
            if decision[1] and decision[1] < self.performance_thresholds['low_confidence']:
                low_confidence_decisions.append({
                    'decision_type': decision[0],
                    'confidence': decision[1],
                    'reasoning': decision[2]
                })
        
        # Analyze decision patterns
        patterns = {
            'decision_frequency': dict(decision_types),
            'avg_confidence': sum(confidence_levels) / len(confidence_levels) if confidence_levels else 0,
            'confidence_distribution': {
                'high': sum(1 for c in confidence_levels if c > 0.8),
                'medium': sum(1 for c in confidence_levels if 0.5 <= c <= 0.8),
                'low': sum(1 for c in confidence_levels if c < 0.5)
            }
        }
        
        return {
            'total_decisions': len(decisions),
            'low_confidence_decisions': low_confidence_decisions[:10],
            'patterns': patterns,
            'decision_quality_score': patterns['avg_confidence'] * 100
        }
        
    async def _analyze_api_performance(self, conn: sqlite3.Connection, start_time: float) -> Dict[str, Any]:
        """Analyze API call performance and reliability"""
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT api_name, endpoint, method,
                   COUNT(*) as call_count,
                   AVG(response_time_ms) as avg_response_time,
                   MAX(response_time_ms) as max_response_time,
                   SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
                   AVG(status_code) as avg_status_code
            FROM api_calls 
            WHERE timestamp > ?
            GROUP BY api_name, endpoint, method
            ORDER BY avg_response_time DESC
        """, (start_time,))
        
        api_stats = cursor.fetchall()
        
        slow_apis = []
        failing_apis = []
        
        for stats in api_stats:
            (api_name, endpoint, method, call_count, avg_response_time, 
             max_response_time, success_count, avg_status_code) = stats
            
            success_rate = success_count / call_count if call_count > 0 else 0
            
            # Identify slow APIs
            if avg_response_time and avg_response_time > self.performance_thresholds['slow_api_ms']:
                slow_apis.append({
                    'api': f"{api_name} {method} {endpoint}",
                    'avg_response_time_ms': avg_response_time,
                    'max_response_time_ms': max_response_time,
                    'call_count': call_count
                })
                
            # Identify failing APIs
            if success_rate < 0.9:  # Less than 90% success rate
                failing_apis.append({
                    'api': f"{api_name} {method} {endpoint}",
                    'success_rate': success_rate,
                    'call_count': call_count,
                    'avg_status_code': avg_status_code
                })
        
        return {
            'total_apis_analyzed': len(api_stats),
            'slow_apis': slow_apis[:5],
            'failing_apis': failing_apis[:5],
            'api_performance_summary': {
                'avg_response_time': sum(s[4] or 0 for s in api_stats) / len(api_stats) if api_stats else 0,
                'total_api_calls': sum(s[3] for s in api_stats),
                'overall_api_success_rate': sum(s[6] for s in api_stats) / sum(s[3] for s in api_stats) if api_stats else 0
            }
        }
        
    async def _analyze_error_patterns(self, conn: sqlite3.Connection, start_time: float) -> Dict[str, Any]:
        """Analyze error patterns and recurring issues"""
        cursor = conn.cursor()
        
        # Get function call errors
        cursor.execute("""
            SELECT error_type, error_message, function_name, module_name,
                   COUNT(*) as error_count,
                   MAX(timestamp) as last_occurrence
            FROM function_calls 
            WHERE timestamp > ? AND success = 0
            GROUP BY error_type, error_message, function_name, module_name
            ORDER BY error_count DESC
        """, (start_time,))
        
        function_errors = cursor.fetchall()
        
        # Get system events with errors
        cursor.execute("""
            SELECT event_type, component, event_data, severity,
                   COUNT(*) as event_count,
                   MAX(timestamp) as last_occurrence
            FROM system_events 
            WHERE timestamp > ? AND severity IN ('ERROR', 'CRITICAL')
            GROUP BY event_type, component, severity
            ORDER BY event_count DESC
        """, (start_time,))
        
        system_errors = cursor.fetchall()
        
        # Categorize errors
        critical_errors = []
        recurring_errors = []
        error_types = Counter()
        
        for error in function_errors:
            error_type, error_msg, func_name, module_name, count, last_time = error
            error_types[error_type] += count
            
            if count >= 5:  # Recurring error
                recurring_errors.append({
                    'error_type': error_type,
                    'function': f"{module_name}.{func_name}",
                    'count': count,
                    'last_occurrence': datetime.fromtimestamp(last_time).isoformat(),
                    'message': error_msg[:200]  # Truncate long messages
                })
                
            if error_type in ['ConnectionError', 'TimeoutError', 'CriticalError']:
                critical_errors.append({
                    'error_type': error_type,
                    'function': f"{module_name}.{func_name}",
                    'count': count,
                    'message': error_msg[:200]
                })
        
        return {
            'total_errors': sum(e[4] for e in function_errors),
            'unique_error_types': len(error_types),
            'error_type_distribution': dict(error_types.most_common(10)),
            'critical_errors': critical_errors[:5],
            'recurring_errors': recurring_errors[:10],
            'system_error_events': len(system_errors)
        }
        
    def _calculate_performance_score(self, function_analysis: Dict, api_analysis: Dict) -> float:
        """Calculate overall performance score (0-100)"""
        score = 100
        
        # Penalize slow functions
        slow_functions_count = len(function_analysis.get('slow_functions', []))
        score -= min(slow_functions_count * 10, 30)  # Max 30 point penalty
        
        # Penalize slow APIs
        slow_apis_count = len(api_analysis.get('slow_apis', []))
        score -= min(slow_apis_count * 15, 25)  # Max 25 point penalty
        
        # Reward good average performance
        avg_time = function_analysis.get('performance_summary', {}).get('avg_execution_time', 1000)
        if avg_time < 100:
            score += 10
        elif avg_time > 1000:
            score -= 15
        
        return max(0, min(100, score))
        
    def _calculate_efficiency_score(self, function_analysis: Dict, decision_analysis: Dict) -> float:
        """Calculate efficiency score (0-100)"""
        score = 100
        
        # Factor in decision confidence
        avg_confidence = decision_analysis.get('patterns', {}).get('avg_confidence', 0.5)
        score = score * avg_confidence
        
        # Penalize memory-intensive functions
        memory_functions = len(function_analysis.get('memory_intensive_functions', []))
        score -= min(memory_functions * 5, 20)
        
        # Factor in function success rate
        success_rate = function_analysis.get('performance_summary', {}).get('overall_success_rate', 1.0)
        score = score * success_rate
        
        return max(0, min(100, score))
        
    def _calculate_reliability_score(self, error_analysis: Dict, api_analysis: Dict) -> float:
        """Calculate reliability score (0-100)"""
        score = 100
        
        # Penalize critical errors
        critical_errors = len(error_analysis.get('critical_errors', []))
        score -= min(critical_errors * 20, 50)
        
        # Penalize recurring errors
        recurring_errors = len(error_analysis.get('recurring_errors', []))
        score -= min(recurring_errors * 10, 30)
        
        # Factor in API reliability
        api_success_rate = api_analysis.get('api_performance_summary', {}).get('overall_api_success_rate', 1.0)
        score = score * api_success_rate
        
        return max(0, min(100, score))
        
    async def _generate_recommendations(self, function_analysis: Dict, decision_analysis: Dict, 
                                      api_analysis: Dict, error_analysis: Dict) -> Dict[str, List[str]]:
        """Generate intelligent recommendations based on analysis"""
        recommendations = {
            'immediate': [],
            'optimization': [],
            'code': [],
            'config': []
        }
        
        # Immediate actions (critical issues)
        critical_errors = error_analysis.get('critical_errors', [])
        if critical_errors:
            recommendations['immediate'].append(f"🚨 Fix {len(critical_errors)} critical errors immediately")
            for error in critical_errors[:3]:
                recommendations['immediate'].append(f"   - Fix {error['error_type']} in {error['function']}")
        
        failing_apis = api_analysis.get('failing_apis', [])
        if failing_apis:
            recommendations['immediate'].append(f"🔧 Fix {len(failing_apis)} failing API endpoints")
            for api in failing_apis[:2]:
                recommendations['immediate'].append(f"   - Fix {api['api']} (success rate: {api['success_rate']:.1%})")
        
        # Optimization suggestions
        slow_functions = function_analysis.get('slow_functions', [])
        if slow_functions:
            recommendations['optimization'].append(f"⚡ Optimize {len(slow_functions)} slow functions")
            top_slow = slow_functions[0]
            recommendations['optimization'].append(f"   - Priority: {top_slow['function']} ({top_slow['avg_execution_time_ms']:.0f}ms avg)")
        
        slow_apis = api_analysis.get('slow_apis', [])
        if slow_apis:
            recommendations['optimization'].append(f"🌐 Optimize {len(slow_apis)} slow API calls")
            recommendations['optimization'].append("   - Consider caching, retry logic, or alternative endpoints")
        
        # Code improvements
        error_prone_functions = function_analysis.get('error_prone_functions', [])
        if error_prone_functions:
            recommendations['code'].append("🛠️ Improve error handling in functions:")
            for func in error_prone_functions[:3]:
                recommendations['code'].append(f"   - {func['function']} (success rate: {func['success_rate']:.1%})")
        
        memory_functions = function_analysis.get('memory_intensive_functions', [])
        if memory_functions:
            recommendations['code'].append("💾 Optimize memory usage in:")
            for func in memory_functions[:2]:
                recommendations['code'].append(f"   - {func['function']} ({func['avg_memory_usage_mb']:.1f}MB avg)")
        
        # Configuration changes
        low_confidence = decision_analysis.get('low_confidence_decisions', [])
        if low_confidence:
            recommendations['config'].append("⚙️ Improve decision confidence:")
            recommendations['config'].append(f"   - {len(low_confidence)} decisions have low confidence")
            recommendations['config'].append("   - Review signal weights and thresholds")
        
        avg_confidence = decision_analysis.get('patterns', {}).get('avg_confidence', 1.0)
        if avg_confidence < 0.7:
            recommendations['config'].append(f"📊 Overall decision confidence is {avg_confidence:.1%}")
            recommendations['config'].append("   - Consider adjusting decision logic parameters")
        
        return recommendations
        
    def display_analysis_report(self, analysis: BotBehaviorAnalysis):
        """Display comprehensive analysis report"""
        console.print("\n" + "="*80)
        console.print("[bold green]🤖 AI BOT BEHAVIOR ANALYSIS REPORT[/bold green]")
        console.print("="*80)
        
        # Overall scores
        scores_table = Table(title="Overall Performance Scores")
        scores_table.add_column("Metric", style="cyan")
        scores_table.add_column("Score", style="green")
        scores_table.add_column("Assessment", style="yellow")
        
        def get_assessment(score):
            if score >= 90: return "Excellent ✅"
            elif score >= 75: return "Good 👍"
            elif score >= 60: return "Fair ⚠️"
            else: return "Needs Improvement ❌"
        
        scores_table.add_row("Performance", f"{analysis.performance_score:.1f}/100", get_assessment(analysis.performance_score))
        scores_table.add_row("Efficiency", f"{analysis.efficiency_score:.1f}/100", get_assessment(analysis.efficiency_score))
        scores_table.add_row("Reliability", f"{analysis.reliability_score:.1f}/100", get_assessment(analysis.reliability_score))
        
        console.print(scores_table)
        
        # Immediate actions
        if analysis.immediate_actions:
            console.print("\n[bold red]🚨 IMMEDIATE ACTIONS REQUIRED[/bold red]")
            for action in analysis.immediate_actions:
                console.print(f"  {action}")
        
        # Performance issues
        if analysis.slow_functions:
            console.print("\n[bold yellow]⚡ SLOW FUNCTIONS[/bold yellow]")
            for func in analysis.slow_functions[:5]:
                console.print(f"  • {func['function']} - {func['avg_execution_time_ms']:.0f}ms avg ({func['call_count']} calls)")
        
        # Error analysis
        if analysis.critical_errors:
            console.print("\n[bold red]🚨 CRITICAL ERRORS[/bold red]")
            for error in analysis.critical_errors[:5]:
                console.print(f"  • {error['error_type']} in {error['function']} ({error['count']} times)")
        
        # Optimization suggestions
        if analysis.optimization_suggestions:
            console.print("\n[bold blue]💡 OPTIMIZATION SUGGESTIONS[/bold blue]")
            for suggestion in analysis.optimization_suggestions:
                console.print(f"  {suggestion}")
        
        # Code improvements
        if analysis.code_improvements:
            console.print("\n[bold green]🛠️ CODE IMPROVEMENTS[/bold green]")
            for improvement in analysis.code_improvements:
                console.print(f"  {improvement}")
        
        # Configuration changes
        if analysis.configuration_changes:
            console.print("\n[bold cyan]⚙️ CONFIGURATION CHANGES[/bold cyan]")
            for change in analysis.configuration_changes:
                console.print(f"  {change}")
        
        # Decision quality
        decision_quality = analysis.decision_quality.get('decision_quality_score', 0)
        console.print(f"\n[bold magenta]🧠 DECISION QUALITY: {decision_quality:.1f}/100[/bold magenta]")
        
        console.print("\n" + "="*80)
        console.print("[dim]Use this analysis to improve your bot's performance and reliability[/dim]")

async def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description="AI-Powered Trading Bot Analyzer")
    parser.add_argument("--analyze-recent", action="store_true", help="Analyze recent behavior (6 hours)")
    parser.add_argument("--full-analysis", action="store_true", help="Full analysis (24 hours)")
    parser.add_argument("--hours", type=int, default=24, help="Hours to analyze")
    parser.add_argument("--export", type=str, help="Export analysis to JSON file")
    
    args = parser.parse_args()
    
    analyzer = BotAnalyzerAI()
    
    if args.analyze_recent:
        hours = 6
    elif args.full_analysis:
        hours = 24
    else:
        hours = args.hours
    
    console.print("[bold blue]🤖 Starting AI Bot Analysis...[/bold blue]")
    
    analysis = await analyzer.analyze_bot_behavior(hours)
    
    if analysis:
        analyzer.display_analysis_report(analysis)
        
        if args.export:
            with open(args.export, 'w') as f:
                # Convert dataclass to dict for JSON serialization
                analysis_dict = {
                    'timestamp': analysis.timestamp,
                    'analysis_period': analysis.analysis_period,
                    'performance_score': analysis.performance_score,
                    'efficiency_score': analysis.efficiency_score,
                    'reliability_score': analysis.reliability_score,
                    'function_performance': analysis.function_performance,
                    'decision_quality': analysis.decision_quality,
                    'api_performance': analysis.api_performance,
                    'error_patterns': analysis.error_patterns,
                    'recommendations': {
                        'immediate_actions': analysis.immediate_actions,
                        'optimization_suggestions': analysis.optimization_suggestions,
                        'code_improvements': analysis.code_improvements,
                        'configuration_changes': analysis.configuration_changes
                    }
                }
                json.dump(analysis_dict, f, indent=2, default=str)
            console.print(f"[green]✅ Analysis exported to {args.export}[/green]")
    else:
        console.print("[red]❌ Analysis failed. Make sure comprehensive logging is enabled.[/red]")

if __name__ == "__main__":
    asyncio.run(main()) 