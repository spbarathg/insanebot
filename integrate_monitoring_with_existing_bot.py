#!/usr/bin/env python3
"""
Integrate Comprehensive Monitoring with Your Existing Trading Bot

This script shows how to modify your existing enhanced_trading_main.py
to include comprehensive monitoring and run everything as one integrated system.

Usage:
    python integrate_monitoring_with_existing_bot.py

This will:
1. Import your existing AntBotSystem from enhanced_trading_main.py
2. Add comprehensive logging to it automatically  
3. Run it with the integrated monitoring dashboard
4. Provide real-time analysis and recommendations

Your existing bot logic stays exactly the same - we just add monitoring on top.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the integrated monitoring system
from main_trading_bot_with_monitoring import TradingBotWithIntegratedMonitoring

# Import comprehensive logging components
from monitoring.comprehensive_bot_logger import (
    get_bot_logger, 
    log_all_methods, 
    log_function_calls
)

from rich.console import Console

console = Console()

class RealTradingBotWithMonitoring(TradingBotWithIntegratedMonitoring):
    """
    Your actual trading bot with integrated comprehensive monitoring
    """
    
    def __init__(self):
        # Initialize the base monitoring system
        super().__init__()
        
        # Replace the simulated bot with your real bot
        try:
            # Import your actual trading bot
            from enhanced_trading_main import AntBotSystem
            
            # Add comprehensive logging to your existing bot
            self.trading_bot = self.wrap_existing_bot_with_logging(AntBotSystem())
            
            console.print("✅ [green]Successfully loaded your real trading bot (AntBotSystem)[/green]")
            console.print("✅ [green]Comprehensive logging added automatically[/green]")
            
        except ImportError as e:
            console.print(f"[red]❌ Could not import your trading bot: {e}[/red]")
            console.print("[yellow]Using simulated bot for demo purposes[/yellow]")
            console.print("[dim]Make sure enhanced_trading_main.py exists and AntBotSystem is properly defined[/dim]")
            
            # Fallback to simulated bot
            from main_trading_bot_with_monitoring import SimulatedTradingBot
            self.trading_bot = SimulatedTradingBot()
    
    def wrap_existing_bot_with_logging(self, bot_instance):
        """
        Add comprehensive logging to your existing bot instance
        """
        # Apply the @log_all_methods decorator to the bot's class
        # This logs all method calls automatically
        bot_class = bot_instance.__class__
        
        # Get all methods and wrap them with logging
        for attr_name in dir(bot_class):
            if not attr_name.startswith('__'):
                attr = getattr(bot_class, attr_name)
                if callable(attr):
                    # Wrap method with comprehensive logging
                    wrapped_method = log_function_calls(attr)
                    setattr(bot_class, attr_name, wrapped_method)
        
        # Add logger instance to your bot
        if not hasattr(bot_instance, 'logger'):
            bot_instance.logger = get_bot_logger("ant_bot_system")
        
        # Wrap the main trading methods if they exist
        if hasattr(bot_instance, 'run'):
            original_run = bot_instance.run
            
            async def logged_run(*args, **kwargs):
                """Wrapped run method with session logging"""
                async with bot_instance.logger.monitoring_session("main_trading_run"):
                    return await original_run(*args, **kwargs)
            
            bot_instance.run = logged_run
        
        # Wrap other key methods that might exist
        key_methods = [
            'process_signals', 'execute_trades', 'analyze_market', 
            'check_positions', 'manage_risk', 'update_strategy'
        ]
        
        for method_name in key_methods:
            if hasattr(bot_instance, method_name):
                method = getattr(bot_instance, method_name)
                if callable(method):
                    wrapped_method = log_function_calls(method)
                    setattr(bot_instance, method_name, wrapped_method)
        
        return bot_instance
    
    async def run_trading_loop(self):
        """
        Override trading loop to use your actual bot's methods
        """
        self.system_status = "trading"
        
        while self.is_running:
            try:
                # Use monitoring session for the trading cycle
                async with self.logger.monitoring_session("real_trading_cycle"):
                    
                    # Check if your bot has a specific run method
                    if hasattr(self.trading_bot, 'run'):
                        # Use your bot's main run method
                        cycle_result = await self.trading_bot.run()
                    elif hasattr(self.trading_bot, 'trading_cycle'):
                        # Alternative method name
                        cycle_result = await self.trading_bot.trading_cycle()
                    elif hasattr(self.trading_bot, 'execute_cycle'):
                        # Another alternative
                        cycle_result = await self.trading_bot.execute_cycle()
                    else:
                        # Fallback to the simulated method
                        cycle_result = await self.trading_bot.run_trading_cycle()
                    
                    # Process the results (adapt based on your bot's return format)
                    await self.process_trading_cycle_result(cycle_result)
                
                # Wait between cycles (adjust timing as needed)
                await asyncio.sleep(30)  # 30 second intervals
                
            except Exception as e:
                error_info = {
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'error': str(e)[:100],
                    'component': 'real_trading_loop'
                }
                self.recent_errors.append(error_info)
                if len(self.recent_errors) > 5:
                    self.recent_errors.pop(0)
                
                # Log the error (automatically handled by comprehensive logging)
                await asyncio.sleep(60)  # Wait longer after errors
    
    async def process_trading_cycle_result(self, cycle_result):
        """
        Process the results from your trading bot
        Adapt this based on what your bot returns
        """
        from datetime import datetime
        
        # Adapt this section based on your bot's actual return format
        if cycle_result:
            # Example processing - modify based on your bot's actual structure
            executed = False
            decision_data = None
            profit = 0
            symbol = 'UNKNOWN'
            
            # Try to extract information from your bot's results
            if isinstance(cycle_result, dict):
                executed = cycle_result.get('executed', False)
                decision_data = cycle_result.get('decision', {})
                profit = cycle_result.get('profit', 0)
                symbol = cycle_result.get('symbol', 'UNKNOWN')
            
            # Update performance metrics
            if executed:
                self.trades_today += 1
                if profit > 0:
                    self.successful_trades += 1
                    self.profit_today += profit
            
            # Track decisions
            if decision_data:
                decision_entry = {
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'decision': decision_data.get('action', 'unknown'),
                    'confidence': decision_data.get('confidence', 0),
                    'symbol': symbol
                }
                self.recent_decisions.append(decision_entry)
                
                # Keep only last 10 decisions
                if len(self.recent_decisions) > 10:
                    self.recent_decisions.pop(0)
            
            # Log the cycle completion
            await self.logger.log_system_event(
                "trading_cycle_complete",
                "real_trading_bot",
                {
                    "executed": executed,
                    "profit": profit,
                    "symbol": symbol,
                    "decision": decision_data
                },
                "INFO"
            )

# Modified startup script for your real bot
async def start_real_bot_with_monitoring():
    """Start your real trading bot with comprehensive monitoring"""
    console.print("[bold green]🤖 Starting Your Real Trading Bot with Comprehensive Monitoring[/bold green]")
    console.print("=" * 70)
    console.print()
    console.print("Loading your existing trading bot from enhanced_trading_main.py...")
    console.print()
    
    # Create and start the integrated system
    bot = RealTradingBotWithMonitoring()
    await bot.start()

def show_integration_instructions():
    """Show instructions for integrating with existing bot"""
    console.print("\n[bold blue]📋 Integration Instructions[/bold blue]")
    console.print("=" * 50)
    console.print()
    console.print("To integrate monitoring with your existing bot:")
    console.print()
    console.print("1. [bold]Make sure your enhanced_trading_main.py exists[/bold]")
    console.print("   - Should contain your AntBotSystem class")
    console.print("   - Should have a main trading method (run, trading_cycle, etc.)")
    console.print()
    console.print("2. [bold]Modify the integration if needed[/bold]")
    console.print("   - Edit this file to match your bot's method names")
    console.print("   - Adjust the result processing in process_trading_cycle_result()")
    console.print()
    console.print("3. [bold]Run the integrated system[/bold]")
    console.print("   - python integrate_monitoring_with_existing_bot.py")
    console.print()
    console.print("4. [bold]Customize monitoring[/bold]")
    console.print("   - Add manual logging for specific events")
    console.print("   - Adjust analysis frequency in the monitoring system")
    console.print()
    console.print("[bold green]Your bot logic stays exactly the same - we just add monitoring![/bold green]")
    console.print()
    console.print("Example of manual logging you can add to your existing bot:")
    console.print()
    console.print("""[dim]
# In your existing bot methods, add logging like this:
async def your_existing_method(self):
    # Log decision points
    await self.logger.log_decision_point(
        "signal_analysis",
        context={"market_conditions": "bullish"},
        inputs={"rsi": 70, "macd": 0.5},
        output={"action": "buy", "amount": 1000},
        confidence=0.85,
        reasoning="Strong technical signals"
    )
    
    # Log API calls
    await self.logger.log_api_call(
        "solana_rpc",
        "/getAccountInfo",
        "POST", 
        request_data=request,
        response_data=response,
        response_time_ms=150,
        success=True
    )
[/dim]""")

async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrate monitoring with existing trading bot")
    parser.add_argument("--instructions", action="store_true", help="Show integration instructions")
    parser.add_argument("--start", action="store_true", help="Start the integrated bot")
    
    args = parser.parse_args()
    
    if args.instructions:
        show_integration_instructions()
    elif args.start:
        await start_real_bot_with_monitoring()
    else:
        console.print("[bold blue]🤖 Real Trading Bot Integration[/bold blue]")
        console.print("=" * 40)
        console.print()
        console.print("Options:")
        console.print("  --instructions  Show integration instructions")
        console.print("  --start        Start your bot with monitoring")
        console.print()
        console.print("Quick start:")
        console.print("  python integrate_monitoring_with_existing_bot.py --start")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass 