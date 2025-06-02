#!/usr/bin/env python3
"""
Main Trading Bot with Integrated Comprehensive Monitoring

This is your main bot runner that automatically includes:
- Comprehensive logging of everything the bot does
- Real-time performance monitoring and analysis
- Automatic issue detection and alerts
- Background AI analysis with recommendations
- Live dashboard and status reporting

Usage:
    python main_trading_bot_with_monitoring.py

The bot will:
1. Start with comprehensive logging enabled automatically
2. Run continuous background monitoring and analysis
3. Display real-time status and recommendations
4. Generate periodic analysis reports
5. Alert you to any issues that need attention

Everything is integrated - just run this one script and get complete visibility.
"""

import asyncio
import sys
import signal
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import comprehensive logging and monitoring
from monitoring.comprehensive_bot_logger import (
    get_bot_logger, 
    log_all_methods, 
    BotActivityLogger
)
from monitoring.bot_analyzer_ai import BotAnalyzerAI

# Import your actual trading bot components
# from enhanced_trading_main import AntBotSystem  # Uncomment to use your real bot

# Rich for beautiful terminal output
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import print as rprint

console = Console()

class TradingBotWithIntegratedMonitoring:
    """Main trading bot class with integrated comprehensive monitoring"""
    
    def __init__(self):
        # Initialize comprehensive logging FIRST
        self.logger = get_bot_logger("main_trading_bot")
        
        # Initialize AI analyzer
        self.analyzer = BotAnalyzerAI()
        
        # Bot status
        self.is_running = False
        self.start_time = None
        self.last_analysis_time = None
        self.current_analysis = None
        
        # Performance tracking
        self.trades_today = 0
        self.profit_today = 0.0
        self.successful_trades = 0
        self.total_api_calls = 0
        self.avg_response_time = 0
        self.current_issues = []
        
        # Real-time stats
        self.recent_decisions = []
        self.recent_errors = []
        self.system_status = "starting"
        
        # Your actual trading bot instance
        # self.trading_bot = AntBotSystem()  # Uncomment for real bot
        self.trading_bot = SimulatedTradingBot()  # Remove this line when using real bot
        
        # Monitoring tasks
        self.monitoring_tasks = []
        
    async def start(self):
        """Start the trading bot with integrated monitoring"""
        console.print("\n🚀 [bold green]Starting Trading Bot with Comprehensive Monitoring[/bold green]")
        console.print("=" * 70)
        
        self.is_running = True
        self.start_time = time.time()
        self.system_status = "initializing"
        
        # Log bot startup
        await self.logger.log_system_event(
            event_type="bot_startup",
            component="main_system",
            event_data={
                "startup_time": datetime.now().isoformat(),
                "monitoring_enabled": True,
                "comprehensive_logging": True
            },
            severity="INFO"
        )
        
        try:
            # Start background monitoring tasks
            self.monitoring_tasks = [
                asyncio.create_task(self.run_trading_loop()),
                asyncio.create_task(self.run_continuous_analysis()),
                asyncio.create_task(self.monitor_system_health()),
                asyncio.create_task(self.display_live_dashboard()),
                asyncio.create_task(self.auto_issue_detection())
            ]
            
            console.print("✅ All systems started successfully")
            console.print("📊 Comprehensive monitoring active")
            console.print("🤖 AI analysis running in background")
            console.print("📈 Live dashboard updating...")
            console.print("\n[dim]Press Ctrl+C to stop[/dim]")
            
            # Wait for all tasks
            await asyncio.gather(*self.monitoring_tasks)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]🛑 Shutdown requested by user[/yellow]")
        except Exception as e:
            console.print(f"\n[red]❌ Critical error: {e}[/red]")
            await self.logger.log_system_event(
                "critical_error",
                "main_system",
                {"error": str(e)},
                "CRITICAL",
                "Bot stopped due to critical error"
            )
        finally:
            await self.shutdown()
    
    async def run_trading_loop(self):
        """Main trading loop with comprehensive logging"""
        self.system_status = "trading"
        
        while self.is_running:
            try:
                # Use monitoring session for the trading cycle
                async with self.logger.monitoring_session("trading_cycle"):
                    
                    # Run one trading cycle
                    cycle_result = await self.trading_bot.run_trading_cycle()
                    
                    # Update performance metrics
                    if cycle_result and cycle_result.get('executed'):
                        self.trades_today += 1
                        if cycle_result.get('profit', 0) > 0:
                            self.successful_trades += 1
                            self.profit_today += cycle_result.get('profit', 0)
                    
                    # Track decisions
                    if cycle_result and cycle_result.get('decision'):
                        self.recent_decisions.append({
                            'time': datetime.now().strftime('%H:%M:%S'),
                            'decision': cycle_result['decision']['decision'],
                            'confidence': cycle_result['decision'].get('confidence', 0),
                            'symbol': cycle_result.get('symbol', 'UNKNOWN')
                        })
                        # Keep only last 10 decisions
                        if len(self.recent_decisions) > 10:
                            self.recent_decisions.pop(0)
                
                # Wait between cycles
                await asyncio.sleep(30)  # 30 second intervals
                
            except Exception as e:
                error_info = {
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'error': str(e)[:100],
                    'component': 'trading_loop'
                }
                self.recent_errors.append(error_info)
                if len(self.recent_errors) > 5:
                    self.recent_errors.pop(0)
                
                # Log the error (automatically handled by comprehensive logging)
                await asyncio.sleep(60)  # Wait longer after errors
    
    async def run_continuous_analysis(self):
        """Run AI analysis continuously in the background"""
        while self.is_running:
            try:
                # Run analysis every 10 minutes
                if (self.last_analysis_time is None or 
                    time.time() - self.last_analysis_time > 600):
                    
                    # Run AI analysis
                    self.current_analysis = await self.analyzer.analyze_bot_behavior(hours_back=1)
                    self.last_analysis_time = time.time()
                    
                    # Check for critical issues
                    if self.current_analysis:
                        await self.process_analysis_results()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                await self.logger.log_system_event(
                    "analysis_error",
                    "ai_analyzer",
                    {"error": str(e)},
                    "ERROR"
                )
                await asyncio.sleep(300)  # Wait 5 minutes after error
    
    async def process_analysis_results(self):
        """Process AI analysis results and take action if needed"""
        if not self.current_analysis:
            return
        
        # Clear previous issues
        self.current_issues = []
        
        # Check for immediate actions needed
        if self.current_analysis.immediate_actions:
            for action in self.current_analysis.immediate_actions:
                self.current_issues.append({
                    'type': 'CRITICAL',
                    'message': action,
                    'time': datetime.now().strftime('%H:%M:%S')
                })
        
        # Check for performance issues
        if self.current_analysis.slow_functions:
            self.current_issues.append({
                'type': 'PERFORMANCE',
                'message': f"{len(self.current_analysis.slow_functions)} slow functions detected",
                'time': datetime.now().strftime('%H:%M:%S')
            })
        
        # Log analysis summary
        await self.logger.log_system_event(
            "ai_analysis_complete",
            "ai_analyzer",
            {
                "performance_score": self.current_analysis.performance_score,
                "efficiency_score": self.current_analysis.efficiency_score,
                "reliability_score": self.current_analysis.reliability_score,
                "issues_found": len(self.current_issues)
            },
            "INFO"
        )
    
    async def monitor_system_health(self):
        """Monitor system health and resources"""
        import psutil
        
        while self.is_running:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                
                # Log system health
                await self.logger.log_system_event(
                    "system_health_check",
                    "system_monitor",
                    {
                        "cpu_usage": cpu_percent,
                        "memory_usage": memory_percent,
                        "uptime_hours": (time.time() - self.start_time) / 3600
                    },
                    "INFO"
                )
                
                # Alert if resources high
                if cpu_percent > 80 or memory_percent > 85:
                    self.current_issues.append({
                        'type': 'SYSTEM',
                        'message': f"High resource usage: CPU {cpu_percent}%, RAM {memory_percent}%",
                        'time': datetime.now().strftime('%H:%M:%S')
                    })
                
                await asyncio.sleep(120)  # Every 2 minutes
                
            except Exception as e:
                await asyncio.sleep(120)
    
    async def auto_issue_detection(self):
        """Automatically detect and alert on issues"""
        while self.is_running:
            try:
                # Check for error patterns
                if len(self.recent_errors) >= 3:
                    self.current_issues.append({
                        'type': 'ERROR_PATTERN',
                        'message': f"{len(self.recent_errors)} recent errors detected",
                        'time': datetime.now().strftime('%H:%M:%S')
                    })
                
                # Check trading performance
                if self.trades_today > 10:
                    success_rate = self.successful_trades / self.trades_today
                    if success_rate < 0.6:  # Less than 60% success
                        self.current_issues.append({
                            'type': 'PERFORMANCE',
                            'message': f"Low success rate: {success_rate:.1%}",
                            'time': datetime.now().strftime('%H:%M:%S')
                        })
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                await asyncio.sleep(300)
    
    async def display_live_dashboard(self):
        """Display live dashboard with real-time information"""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=8)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        with Live(layout, refresh_per_second=1, screen=True):
            while self.is_running:
                try:
                    # Update header
                    uptime = timedelta(seconds=int(time.time() - self.start_time))
                    layout["header"].update(Panel(
                        f"[bold blue]🤖 Trading Bot with Comprehensive Monitoring[/bold blue] | "
                        f"Status: [green]{self.system_status}[/green] | "
                        f"Uptime: {uptime} | "
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        style="blue"
                    ))
                    
                    # Update main panels
                    layout["left"].update(self.create_performance_panel())
                    layout["right"].update(self.create_analysis_panel())
                    
                    # Update footer
                    layout["footer"].update(self.create_status_panel())
                    
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    # Don't let dashboard errors crash the bot
                    await asyncio.sleep(1)
    
    def create_performance_panel(self) -> Panel:
        """Create performance metrics panel"""
        table = Table(title="🚀 Live Performance Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Calculate success rate
        success_rate = (self.successful_trades / max(self.trades_today, 1)) * 100
        
        table.add_row("Trades Today", str(self.trades_today))
        table.add_row("Success Rate", f"{success_rate:.1f}%")
        table.add_row("Profit Today", f"{self.profit_today:.4f} SOL")
        table.add_row("System Status", self.system_status.title())
        
        # Add recent decisions
        if self.recent_decisions:
            table.add_row("", "")
            table.add_row("[bold]Recent Decisions", "")
            for decision in self.recent_decisions[-3:]:
                table.add_row(
                    f"  {decision['time']}",
                    f"{decision['decision']} {decision['symbol']} ({decision['confidence']:.1%})"
                )
        
        return Panel(table)
    
    def create_analysis_panel(self) -> Panel:
        """Create AI analysis panel"""
        if not self.current_analysis:
            return Panel(
                Text("🤖 AI Analysis\n\nAnalyzing bot behavior...\nFirst analysis in progress."),
                title="AI Analysis"
            )
        
        table = Table(title="🤖 AI Analysis Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="green")
        
        table.add_row("Performance", f"{self.current_analysis.performance_score:.1f}/100")
        table.add_row("Efficiency", f"{self.current_analysis.efficiency_score:.1f}/100")
        table.add_row("Reliability", f"{self.current_analysis.reliability_score:.1f}/100")
        
        # Add analysis timestamp
        if self.last_analysis_time:
            analysis_age = int((time.time() - self.last_analysis_time) / 60)
            table.add_row("Last Analysis", f"{analysis_age} minutes ago")
        
        return Panel(table)
    
    def create_status_panel(self) -> Panel:
        """Create status and issues panel"""
        content = ""
        
        # Show current issues
        if self.current_issues:
            content += "[bold red]🚨 Active Issues:[/bold red]\n"
            for issue in self.current_issues[-5:]:  # Show last 5 issues
                content += f"  • [{issue['time']}] {issue['type']}: {issue['message']}\n"
        else:
            content += "[bold green]✅ No active issues detected[/bold green]\n"
        
        # Show recent errors
        if self.recent_errors:
            content += "\n[bold yellow]⚠️ Recent Errors:[/bold yellow]\n"
            for error in self.recent_errors[-3:]:
                content += f"  • [{error['time']}] {error['component']}: {error['error']}\n"
        
        # Show recommendations
        if (self.current_analysis and 
            self.current_analysis.optimization_suggestions):
            content += "\n[bold blue]💡 Recommendations:[/bold blue]\n"
            for rec in self.current_analysis.optimization_suggestions[:2]:
                content += f"  • {rec}\n"
        
        # Show controls
        content += "\n[dim]Controls: Ctrl+C to stop | Analysis runs every 10 min[/dim]"
        
        return Panel(content, title="🔍 System Status & Issues")
    
    async def shutdown(self):
        """Gracefully shutdown the bot and monitoring"""
        console.print("\n[yellow]🔄 Shutting down...[/yellow]")
        
        self.is_running = False
        
        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Log shutdown
        await self.logger.log_system_event(
            "bot_shutdown",
            "main_system",
            {
                "shutdown_time": datetime.now().isoformat(),
                "total_trades": self.trades_today,
                "total_profit": self.profit_today,
                "uptime_hours": (time.time() - self.start_time) / 3600
            },
            "INFO"
        )
        
        # Generate final analysis report
        if self.current_analysis:
            console.print("\n📊 [bold blue]Final Analysis Report:[/bold blue]")
            console.print(f"Performance Score: {self.current_analysis.performance_score:.1f}/100")
            console.print(f"Efficiency Score: {self.current_analysis.efficiency_score:.1f}/100")
            console.print(f"Reliability Score: {self.current_analysis.reliability_score:.1f}/100")
            
            if self.current_analysis.immediate_actions:
                console.print("\n🚨 [red]Actions needed:[/red]")
                for action in self.current_analysis.immediate_actions[:3]:
                    console.print(f"  • {action}")
        
        console.print("\n✅ [green]Shutdown complete[/green]")
        console.print("📁 All logs saved to: logs/comprehensive/")
        console.print("📊 Run analysis anytime: python monitoring/bot_analyzer_ai.py --full-analysis")

# Simulated trading bot for demo (remove when using real bot)
@log_all_methods  # This decorator logs everything automatically
class SimulatedTradingBot:
    """Simulated trading bot for demonstration"""
    
    def __init__(self):
        self.logger = get_bot_logger("simulated_bot")
        self.cycle_count = 0
    
    async def run_trading_cycle(self):
        """Simulate a trading cycle"""
        self.cycle_count += 1
        
        # Simulate market analysis
        market_data = await self.get_market_data()
        
        # Simulate decision making
        decision = await self.make_trading_decision(market_data)
        
        # Simulate trade execution
        execution_result = None
        if decision['decision'] != 'hold':
            execution_result = await self.execute_trade(decision)
        
        return {
            'executed': execution_result is not None,
            'decision': decision,
            'market_data': market_data,
            'execution': execution_result,
            'symbol': 'BONK',
            'profit': 0.001 if execution_result else 0
        }
    
    async def get_market_data(self):
        """Simulate getting market data"""
        await asyncio.sleep(0.1)  # Simulate API delay
        
        # Log API call
        await self.logger.log_api_call(
            "market_api",
            "/api/v1/market_data",
            "GET",
            request_data={"symbol": "BONK"},
            response_data={"price": 1.25, "volume": 1000000},
            response_time_ms=100,
            success=True
        )
        
        return {"price": 1.25, "volume": 1000000}
    
    async def make_trading_decision(self, market_data):
        """Simulate trading decision"""
        import random
        
        confidence = random.uniform(0.4, 0.9)
        decision = "buy" if confidence > 0.7 else "hold" if confidence > 0.5 else "sell"
        
        # Log decision point
        await self.logger.log_decision_point(
            "trading_decision",
            context={"market_data": market_data, "cycle": self.cycle_count},
            inputs={"price": market_data["price"], "volume": market_data["volume"]},
            output={"decision": decision, "confidence": confidence},
            confidence=confidence,
            reasoning=f"Market analysis shows {confidence:.1%} confidence for {decision}"
        )
        
        return {"decision": decision, "confidence": confidence}
    
    async def execute_trade(self, decision):
        """Simulate trade execution"""
        await asyncio.sleep(0.05)  # Simulate execution delay
        
        # Log system event
        await self.logger.log_system_event(
            "trade_executed",
            "execution_engine",
            {"decision": decision["decision"], "confidence": decision["confidence"]},
            "INFO"
        )
        
        return {"status": "executed", "price": 1.25}

async def main():
    """Main function to start the integrated trading bot"""
    console.print("[bold green]🤖 Trading Bot with Integrated Comprehensive Monitoring[/bold green]")
    console.print("=" * 70)
    console.print()
    console.print("This system includes:")
    console.print("✅ Comprehensive logging of all bot activities")
    console.print("✅ Real-time performance monitoring")
    console.print("✅ Continuous AI analysis and recommendations")
    console.print("✅ Live dashboard with key metrics")
    console.print("✅ Automatic issue detection and alerts")
    console.print("✅ Complete audit trail for analysis")
    console.print()
    
    # Handle graceful shutdown
    def signal_handler(signum, frame):
        console.print("\n[yellow]Received shutdown signal[/yellow]")
        # The event loop will handle the KeyboardInterrupt
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the integrated bot
    bot = TradingBotWithIntegratedMonitoring()
    await bot.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # Handled gracefully in the main loop 