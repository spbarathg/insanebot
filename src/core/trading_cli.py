#!/usr/bin/env python3
"""
📊💹 Trading Bot CLI Interface

Real-time monitoring and control interface for your Solana memecoin trading bot.
Monitor positions, performance, recent trades, and interact with the bot.
"""

import os
import json
import asyncio
import datetime
import time
import subprocess
import sys
from typing import Dict, List, Optional, Any
from pathlib import Path
import threading
import signal

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class TradingBotCLI:
    """Comprehensive CLI interface for trading bot monitoring and control"""
    
    def __init__(self):
        self.bot_status = "Unknown"
        self.performance_data = {}
        self.positions = []
        self.recent_trades = []
        self.market_data = {}
        self.alerts = []
        self.refresh_interval = 10  # Increased from 2 to 10 seconds
        self.auto_refresh = True
        self.last_update = None
        self.last_data_hash = None  # Track data changes
        
        # Load initial data
        self.load_all_data()
    
    def load_all_data(self):
        """Load all trading bot data from files"""
        try:
            # Load performance metrics
            if os.path.exists("data/performance_metrics.json"):
                with open("data/performance_metrics.json", 'r') as f:
                    self.performance_data = json.load(f)
            
            # Load current positions
            if os.path.exists("data/positions.json"):
                with open("data/positions.json", 'r') as f:
                    self.positions = json.load(f)
            
            # Load recent trades
            if os.path.exists("logs/recent_trades.json"):
                with open("logs/recent_trades.json", 'r') as f:
                    self.recent_trades = json.load(f)
            
            # Load market data
            if os.path.exists("data/market_analysis.json"):
                with open("data/market_analysis.json", 'r') as f:
                    self.market_data = json.load(f)
            
            # Load alerts
            if os.path.exists("logs/alerts.json"):
                with open("logs/alerts.json", 'r') as f:
                    self.alerts = json.load(f)
            
            # Check bot status
            self.check_bot_status()
            
            self.last_update = datetime.datetime.now()
            
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def has_data_changed(self) -> bool:
        """Check if data has changed since last update"""
        try:
            current_hash = hash(str(self.performance_data) + str(self.positions) + 
                              str(self.recent_trades) + str(self.market_data) + str(self.alerts))
            
            if self.last_data_hash is None or current_hash != self.last_data_hash:
                self.last_data_hash = current_hash
                return True
            return False
        except:
            return True  # Assume changed if error
    
    def check_bot_status(self):
        """Check if trading bot is running"""
        try:
            # Check for bot process or status file
            if os.path.exists("data/bot_status.json"):
                with open("data/bot_status.json", 'r') as f:
                    status_data = json.load(f)
                    self.bot_status = status_data.get("status", "Unknown")
            else:
                # Try to detect if main bot is running
                if os.path.exists("logs/bot.log"):
                    # Check if log was updated recently (within 30 seconds)
                    log_time = os.path.getmtime("logs/bot.log")
                    if time.time() - log_time < 30:
                        self.bot_status = "Running"
                    else:
                        self.bot_status = "Inactive"
                else:
                    self.bot_status = "Stopped"
        except:
            self.bot_status = "Unknown"
    
    def clear_screen(self):
        """Clear terminal screen"""
        try:
            if os.name == 'nt':  # Windows
                subprocess.run(['cmd', '/c', 'cls'], check=False, capture_output=True)
            else:  # Unix/Linux/macOS
                subprocess.run(['clear'], check=False, capture_output=True)
        except Exception:
            # Fallback: print empty lines if subprocess fails
            print('\n' * 50)
    
    def print_header(self):
        """Print main header"""
        self.clear_screen()
        print(f"{Colors.CYAN}{Colors.BOLD}{'='*80}")
        print("🚀 ADVANCED SOLANA MEMECOIN TRADING BOT - CONTROL CENTER 🚀")
        print(f"{'='*80}{Colors.END}")
        
        # Status bar
        status_color = Colors.GREEN if self.bot_status == "Running" else Colors.RED
        update_time = self.last_update.strftime("%H:%M:%S") if self.last_update else "Never"
        
        print(f"{Colors.WHITE}Status: {status_color}{self.bot_status}{Colors.WHITE} | "
              f"Last Update: {update_time} | "
              f"Auto-refresh: {'ON' if self.auto_refresh else 'OFF'}{Colors.END}\n")
    
    def print_performance_summary(self):
        """Print performance summary"""
        print(f"{Colors.YELLOW}{Colors.BOLD}📊 PERFORMANCE OVERVIEW{Colors.END}")
        print("─" * 50)
        
        total_profit = self.performance_data.get("total_profit_pct", 0)
        today_profit = self.performance_data.get("today_profit_pct", 0)
        win_rate = self.performance_data.get("win_rate", 0)
        total_trades = self.performance_data.get("total_trades", 0)
        
        # Color code profits
        total_color = Colors.GREEN if total_profit > 0 else Colors.RED
        today_color = Colors.GREEN if today_profit > 0 else Colors.RED
        
        print(f"Total P&L:     {total_color}{total_profit:+.2f}%{Colors.END}")
        print(f"Today P&L:     {today_color}{today_profit:+.2f}%{Colors.END}")
        print(f"Win Rate:      {Colors.CYAN}{win_rate:.1f}%{Colors.END}")
        print(f"Total Trades:  {Colors.WHITE}{total_trades}{Colors.END}")
        print(f"Portfolio:     {Colors.MAGENTA}${self.performance_data.get('portfolio_value', 0):,.2f}{Colors.END}")
        print()
    
    def print_active_positions(self):
        """Print active positions"""
        print(f"{Colors.BLUE}{Colors.BOLD}📈 ACTIVE POSITIONS ({len(self.positions)}){Colors.END}")
        print("─" * 70)
        
        if not self.positions:
            print(f"{Colors.YELLOW}No active positions{Colors.END}")
        else:
            print(f"{'Token':<12} {'Side':<6} {'Size':<12} {'Entry':<10} {'Current':<10} {'P&L':<8}")
            print("─" * 70)
            
            for pos in self.positions:
                token = pos.get("token", "")[:11]
                side = pos.get("side", "")
                size = f"${pos.get('size', 0):,.0f}"
                entry = f"${pos.get('entry_price', 0):.4f}"
                current = f"${pos.get('current_price', 0):.4f}"
                pnl = pos.get("unrealized_pnl_pct", 0)
                pnl_color = Colors.GREEN if pnl > 0 else Colors.RED
                
                print(f"{token:<12} {side:<6} {size:<12} {entry:<10} {current:<10} "
                      f"{pnl_color}{pnl:+.1f}%{Colors.END}")
        print()
    
    def print_recent_trades(self):
        """Print recent trades"""
        print(f"{Colors.MAGENTA}{Colors.BOLD}🔄 RECENT TRADES (Last 5){Colors.END}")
        print("─" * 80)
        
        if not self.recent_trades:
            print(f"{Colors.YELLOW}No recent trades{Colors.END}")
        else:
            print(f"{'Time':<8} {'Token':<12} {'Side':<6} {'Size':<12} {'Price':<10} {'P&L':<8}")
            print("─" * 80)
            
            for trade in self.recent_trades[-5:]:  # Last 5 trades
                time_str = trade.get("timestamp", "")[-8:]  # Last 8 chars (time)
                token = trade.get("token", "")[:11]
                side = trade.get("side", "")
                size = f"${trade.get('size', 0):,.0f}"
                price = f"${trade.get('price', 0):.4f}"
                pnl = trade.get("realized_pnl_pct", 0)
                pnl_color = Colors.GREEN if pnl > 0 else Colors.RED
                
                print(f"{time_str:<8} {token:<12} {side:<6} {size:<12} {price:<10} "
                      f"{pnl_color}{pnl:+.1f}%{Colors.END}")
        print()
    
    def print_market_overview(self):
        """Print market overview"""
        print(f"{Colors.GREEN}{Colors.BOLD}🌍 MARKET OVERVIEW{Colors.END}")
        print("─" * 40)
        
        if not self.market_data:
            print(f"{Colors.YELLOW}Market data not available{Colors.END}")
        else:
            sentiment = self.market_data.get("sentiment", "Neutral")
            vol_score = self.market_data.get("volatility_score", 0)
            trending = self.market_data.get("trending_tokens", [])
            
            sentiment_color = (Colors.GREEN if sentiment == "Bullish" 
                             else Colors.RED if sentiment == "Bearish" 
                             else Colors.YELLOW)
            
            print(f"Sentiment:     {sentiment_color}{sentiment}{Colors.END}")
            print(f"Volatility:    {Colors.CYAN}{vol_score:.1f}/10{Colors.END}")
            print(f"SOL Price:     {Colors.WHITE}${self.market_data.get('sol_price', 0):.2f}{Colors.END}")
            
            if trending:
                print(f"Trending:      {Colors.MAGENTA}{', '.join(trending[:3])}{Colors.END}")
        print()
    
    def print_alerts(self):
        """Print recent alerts"""
        if self.alerts:
            print(f"{Colors.RED}{Colors.BOLD}🚨 RECENT ALERTS{Colors.END}")
            print("─" * 50)
            
            for alert in self.alerts[-3:]:  # Last 3 alerts
                timestamp = alert.get("timestamp", "")[-8:]
                message = alert.get("message", "")
                alert_type = alert.get("type", "INFO")
                
                type_color = (Colors.RED if alert_type == "ERROR" 
                            else Colors.YELLOW if alert_type == "WARNING"
                            else Colors.CYAN)
                
                print(f"{timestamp} [{type_color}{alert_type}{Colors.END}] {message}")
            print()
    
    def print_commands(self):
        """Print available commands"""
        print(f"{Colors.CYAN}{Colors.BOLD}🎮 AVAILABLE COMMANDS{Colors.END}")
        print("─" * 50)
        commands = [
            ("r", "Refresh data manually"),
            ("p", "Pause/Resume bot"),
            ("pos", "View detailed positions"),
            ("trades", "View trade history"),
            ("chat", "Open AI chat interface"),
            ("alerts", "View all alerts"),
            ("config", "Bot configuration"),
            ("logs", "View bot logs"),
            ("auto", "Toggle auto-refresh"),
            ("interval", f"Set refresh interval (current: {self.refresh_interval}s)"),
            ("clear", "Clear screen"),
            ("q", "Quit")
        ]
        
        for cmd, desc in commands:
            print(f"{Colors.WHITE}{cmd:<8}{Colors.END} {desc}")
        print()
    
    def handle_command(self, command: str):
        """Handle user commands"""
        cmd = command.lower().strip()
        
        if cmd == 'r':
            self.load_all_data()
            print(f"{Colors.GREEN}✅ Data refreshed{Colors.END}")
            time.sleep(1)
        
        elif cmd == 'p':
            self.toggle_bot_status()
        
        elif cmd == 'pos':
            self.show_detailed_positions()
        
        elif cmd == 'trades':
            self.show_trade_history()
        
        elif cmd == 'chat':
            self.open_chat_interface()
        
        elif cmd == 'alerts':
            self.show_all_alerts()
        
        elif cmd == 'config':
            self.show_bot_config()
        
        elif cmd == 'logs':
            self.show_bot_logs()
        
        elif cmd == 'auto':
            self.auto_refresh = not self.auto_refresh
            status = "enabled" if self.auto_refresh else "disabled"
            print(f"{Colors.CYAN}Auto-refresh {status}{Colors.END}")
            time.sleep(1)
        
        elif cmd == 'interval':
            new_interval = int(input(f"Enter new refresh interval (in seconds): "))
            self.refresh_interval = new_interval
            print(f"{Colors.CYAN}Refresh interval set to {self.refresh_interval}s{Colors.END}")
            time.sleep(1)
        
        elif cmd == 'clear':
            pass  # Will be handled by main loop
        
        elif cmd == 'q':
            return False
        
        else:
            print(f"{Colors.RED}Unknown command: {cmd}{Colors.END}")
            time.sleep(1)
        
        return True
    
    def toggle_bot_status(self):
        """Toggle bot pause/resume"""
        try:
            if self.bot_status == "Running":
                # Send pause signal
                with open("data/bot_commands.json", 'w') as f:
                    json.dump({"command": "pause", "timestamp": datetime.datetime.now().isoformat()}, f)
                print(f"{Colors.YELLOW}⏸️  Pause signal sent to bot{Colors.END}")
            else:
                # Send resume signal
                with open("data/bot_commands.json", 'w') as f:
                    json.dump({"command": "resume", "timestamp": datetime.datetime.now().isoformat()}, f)
                print(f"{Colors.GREEN}▶️  Resume signal sent to bot{Colors.END}")
            
            time.sleep(2)
            self.load_all_data()
            
        except Exception as e:
            print(f"{Colors.RED}Error toggling bot status: {e}{Colors.END}")
            time.sleep(2)
    
    def show_detailed_positions(self):
        """Show detailed position information"""
        self.clear_screen()
        print(f"{Colors.BLUE}{Colors.BOLD}📊 DETAILED POSITIONS{Colors.END}")
        print("="*80)
        
        if not self.positions:
            print(f"{Colors.YELLOW}No active positions{Colors.END}")
        else:
            for i, pos in enumerate(self.positions, 1):
                pnl = pos.get("unrealized_pnl_pct", 0)
                pnl_color = Colors.GREEN if pnl > 0 else Colors.RED
                
                print(f"\n{Colors.BOLD}Position {i}: {pos.get('token', 'Unknown')}{Colors.END}")
                print(f"Side:         {pos.get('side', 'Unknown')}")
                print(f"Size:         ${pos.get('size', 0):,.2f}")
                print(f"Entry Price:  ${pos.get('entry_price', 0):.6f}")
                print(f"Current:      ${pos.get('current_price', 0):.6f}")
                print(f"P&L:          {pnl_color}{pnl:+.2f}%{Colors.END}")
                print(f"Duration:     {pos.get('duration', 'Unknown')}")
                print("-" * 40)
        
        input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.END}")
    
    def show_trade_history(self):
        """Show detailed trade history"""
        self.clear_screen()
        print(f"{Colors.MAGENTA}{Colors.BOLD}📈 TRADE HISTORY{Colors.END}")
        print("="*80)
        
        if not self.recent_trades:
            print(f"{Colors.YELLOW}No trades available{Colors.END}")
        else:
            for trade in self.recent_trades[-10:]:  # Last 10 trades
                pnl = trade.get("realized_pnl_pct", 0)
                pnl_color = Colors.GREEN if pnl > 0 else Colors.RED
                
                print(f"\n{Colors.BOLD}{trade.get('timestamp', '')}{Colors.END}")
                print(f"Token: {trade.get('token', 'Unknown')} | "
                      f"Side: {trade.get('side', 'Unknown')} | "
                      f"Size: ${trade.get('size', 0):,.2f}")
                print(f"Price: ${trade.get('price', 0):.6f} | "
                      f"P&L: {pnl_color}{pnl:+.2f}%{Colors.END}")
                print("-" * 60)
        
        input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.END}")
    
    def open_chat_interface(self):
        """Open the AI chat interface"""
        print(f"{Colors.GREEN}🤖 Opening AI chat interface...{Colors.END}")
        time.sleep(1)
        
        try:
            # Run the chat interface
            subprocess.run([sys.executable, "src/core/local_llm_chat.py"])
        except Exception as e:
            print(f"{Colors.RED}Error opening chat: {e}{Colors.END}")
            input("Press Enter to continue...")
    
    def show_all_alerts(self):
        """Show all alerts"""
        self.clear_screen()
        print(f"{Colors.RED}{Colors.BOLD}🚨 ALL ALERTS{Colors.END}")
        print("="*80)
        
        if not self.alerts:
            print(f"{Colors.YELLOW}No alerts{Colors.END}")
        else:
            for alert in self.alerts:
                alert_type = alert.get("type", "INFO")
                type_color = (Colors.RED if alert_type == "ERROR" 
                            else Colors.YELLOW if alert_type == "WARNING"
                            else Colors.CYAN)
                
                print(f"\n{Colors.BOLD}{alert.get('timestamp', '')}{Colors.END}")
                print(f"Type: {type_color}{alert_type}{Colors.END}")
                print(f"Message: {alert.get('message', '')}")
                print("-" * 60)
        
        input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.END}")
    
    def show_bot_config(self):
        """Show bot configuration"""
        self.clear_screen()
        print(f"{Colors.YELLOW}{Colors.BOLD}⚙️  BOT CONFIGURATION{Colors.END}")
        print("="*80)
        
        try:
            if os.path.exists("config/bot_config.json"):
                with open("config/bot_config.json", 'r') as f:
                    config = json.load(f)
                
                for key, value in config.items():
                    print(f"{Colors.CYAN}{key}:{Colors.END} {value}")
            else:
                print(f"{Colors.YELLOW}Configuration file not found{Colors.END}")
        
        except Exception as e:
            print(f"{Colors.RED}Error loading config: {e}{Colors.END}")
        
        input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.END}")
    
    def show_bot_logs(self):
        """Show recent bot logs"""
        self.clear_screen()
        print(f"{Colors.WHITE}{Colors.BOLD}📄 BOT LOGS (Last 20 lines){Colors.END}")
        print("="*80)
        
        try:
            if os.path.exists("logs/bot.log"):
                with open("logs/bot.log", 'r') as f:
                    lines = f.readlines()
                    for line in lines[-20:]:  # Last 20 lines
                        # Color code log levels
                        if "ERROR" in line:
                            print(f"{Colors.RED}{line.strip()}{Colors.END}")
                        elif "WARNING" in line:
                            print(f"{Colors.YELLOW}{line.strip()}{Colors.END}")
                        elif "INFO" in line:
                            print(f"{Colors.CYAN}{line.strip()}{Colors.END}")
                        else:
                            print(line.strip())
            else:
                print(f"{Colors.YELLOW}Log file not found{Colors.END}")
        
        except Exception as e:
            print(f"{Colors.RED}Error reading logs: {e}{Colors.END}")
        
        input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.END}")
    
    def run_dashboard(self):
        """Main dashboard loop"""
        try:
            # Initial display
            self.print_header()
            self.print_performance_summary()
            self.print_active_positions()
            self.print_recent_trades()
            self.print_market_overview()
            self.print_alerts()
            self.print_commands()
            
            while True:
                # Get user input with timeout if auto-refresh is on
                if self.auto_refresh:
                    print(f"{Colors.CYAN}Command (auto-refresh in {self.refresh_interval}s):{Colors.END} ", end="", flush=True)
                    
                    # Windows-compatible non-blocking input
                    if os.name == 'nt':  # Windows
                        import msvcrt
                        start_time = time.time()
                        command = ""
                        
                        while time.time() - start_time < self.refresh_interval:
                            if msvcrt.kbhit():
                                char = msvcrt.getch().decode('utf-8')
                                if char == '\r':  # Enter key
                                    print()  # New line
                                    break
                                elif char == '\b':  # Backspace
                                    if command:
                                        command = command[:-1]
                                        print('\b \b', end='', flush=True)
                                else:
                                    command += char
                                    print(char, end='', flush=True)
                            time.sleep(0.1)
                        else:
                            # Timeout reached - check for data changes
                            old_data_hash = self.last_data_hash
                            self.load_all_data()
                            
                            if self.has_data_changed() or old_data_hash != self.last_data_hash:
                                # Only refresh screen if data changed
                                print(f"\n{Colors.GREEN}📊 Data updated!{Colors.END}")
                                time.sleep(0.5)
                                self.print_header()
                                self.print_performance_summary()
                                self.print_active_positions()
                                self.print_recent_trades()
                                self.print_market_overview()
                                self.print_alerts()
                                self.print_commands()
                            else:
                                # Just show a subtle indicator that system is alive
                                print(f"\r{Colors.CYAN}Command (auto-refresh in {self.refresh_interval}s): {Colors.YELLOW}●{Colors.END} ", end="", flush=True)
                                time.sleep(0.5)
                                print(f"\r{Colors.CYAN}Command (auto-refresh in {self.refresh_interval}s):{Colors.END} ", end="", flush=True)
                            continue
                        
                        if command.strip():
                            if not self.handle_command(command.strip()):
                                break
                    else:
                        # Unix/Linux/Mac - use select
                        import select
                        if sys.stdin in select.select([sys.stdin], [], [], self.refresh_interval)[0]:
                            command = input().strip()
                            if command:
                                if not self.handle_command(command):
                                    break
                        else:
                            # Timeout reached - check for data changes
                            old_data_hash = self.last_data_hash
                            self.load_all_data()
                            
                            if self.has_data_changed() or old_data_hash != self.last_data_hash:
                                # Only refresh screen if data changed
                                print(f"\n{Colors.GREEN}📊 Data updated!{Colors.END}")
                                time.sleep(0.5)
                                self.print_header()
                                self.print_performance_summary()
                                self.print_active_positions()
                                self.print_recent_trades()
                                self.print_market_overview()
                                self.print_alerts()
                                self.print_commands()
                else:
                    command = input(f"{Colors.CYAN}Command: {Colors.END}").strip()
                    if command:
                        if not self.handle_command(command):
                            break
        
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Exiting trading CLI...{Colors.END}")
        except Exception as e:
            print(f"\n{Colors.RED}Error in dashboard: {e}{Colors.END}")
            print(f"{Colors.YELLOW}Restarting dashboard...{Colors.END}")
            time.sleep(2)
            # Restart the dashboard
            self.run_dashboard()


def main():
    """Main CLI entry point"""
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print(f"\n{Colors.YELLOW}Goodbye! Keep those gains coming! 💰{Colors.END}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and run CLI
    cli = TradingBotCLI()
    cli.run_dashboard()


if __name__ == "__main__":
    main() 