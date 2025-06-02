#!/usr/bin/env python3
"""
🤖 Trading Bot Startup Script

This is your main startup script. Just run this and everything works together:
- Your trading bot runs automatically
- Comprehensive logging captures everything
- Real-time monitoring and analysis 
- Live dashboard with performance metrics
- AI-powered recommendations

Usage:
    python start_bot.py

That's it! Everything else is automatic.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel

console = Console()

async def main():
    """Main startup function"""
    console.print("\n🤖 [bold green]Trading Bot Startup[/bold green]")
    console.print("=" * 50)
    
    # Try to use your real bot first
    try:
        console.print("🔍 [blue]Checking for your trading bot (enhanced_trading_main.py)...[/blue]")
        
        # Check if enhanced_trading_main.py exists
        if Path("enhanced_trading_main.py").exists():
            console.print("✅ [green]Found your trading bot! Starting with comprehensive monitoring...[/green]")
            
            # Import and start with your real bot
            from integrate_monitoring_with_existing_bot import start_real_bot_with_monitoring
            await start_real_bot_with_monitoring()
            
        else:
            console.print("⚠️  [yellow]enhanced_trading_main.py not found[/yellow]")
            console.print("🎭 [blue]Starting with demo bot to show how the system works...[/blue]")
            
            # Start with demo bot
            from main_trading_bot_with_monitoring import main as start_demo_bot
            await start_demo_bot()
            
    except ImportError as e:
        console.print(f"⚠️  [yellow]Could not load your bot: {e}[/yellow]")
        console.print("🎭 [blue]Starting with demo bot...[/blue]")
        
        # Start with demo bot
        from main_trading_bot_with_monitoring import main as start_demo_bot
        await start_demo_bot()
        
    except Exception as e:
        console.print(f"❌ [red]Error starting bot: {e}[/red]")
        console.print("\n[bold]Troubleshooting:[/bold]")
        console.print("1. Make sure enhanced_trading_main.py exists")
        console.print("2. Check that AntBotSystem class is properly defined")
        console.print("3. Verify all dependencies are installed")

def show_startup_info():
    """Show information about what this script does"""
    info_text = """
🤖 This script automatically starts your trading bot with comprehensive monitoring:

✅ Loads your existing bot from enhanced_trading_main.py
✅ Adds comprehensive logging automatically (logs everything)
✅ Runs continuous AI analysis in the background
✅ Shows live dashboard with real-time metrics
✅ Detects issues and provides recommendations
✅ Generates analysis reports periodically

Your bot logic stays exactly the same - we just add monitoring and analysis on top!

Features running automatically:
- Function call logging with timing and memory usage
- API request/response logging with performance tracking
- Decision point logging with full context and reasoning
- Error capture with stack traces and pattern analysis
- System health monitoring (CPU, memory, uptime)
- Real-time performance analysis and scoring
- Automatic issue detection and alerts
- AI-powered optimization recommendations

Dashboard shows:
- Live performance metrics (trades, success rate, profit)
- Recent trading decisions with confidence levels
- AI analysis scores (Performance, Efficiency, Reliability)
- Active issues and recommendations
- System status and health indicators

Just run: python start_bot.py
Everything else is automatic!
    """
    
    console.print(Panel(info_text.strip(), title="🤖 Trading Bot with Comprehensive Monitoring", style="blue"))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start trading bot with comprehensive monitoring")
    parser.add_argument("--info", action="store_true", help="Show information about the system")
    
    args = parser.parse_args()
    
    if args.info:
        show_startup_info()
    else:
        console.print("\n[dim]Starting trading bot with comprehensive monitoring...[/dim]")
        console.print("[dim]Press Ctrl+C to stop[/dim]\n")
        
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            console.print("\n[yellow]Bot stopped by user[/yellow]")
            console.print("📁 All logs saved to: logs/comprehensive/")
            console.print("📊 View analysis: python monitoring/bot_analyzer_ai.py --full-analysis")
        except Exception as e:
            console.print(f"\n[red]Unexpected error: {e}[/red]") 