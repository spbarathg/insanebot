#!/usr/bin/env python3
"""
Enhanced Ant Bot - CLI Monitor
Real-time monitoring of AI model development and bot status
"""

import os
import json
import time
import asyncio
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import subprocess
import threading
from collections import deque

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

class AntBotMonitor:
    """Real-time CLI monitor for Enhanced Ant Bot"""
    
    def __init__(self):
        self.current_view = "dashboard"
        self.running = True
        self.refresh_rate = 2  # seconds
        self.ai_history = deque(maxlen=50)  # Store last 50 AI metrics
        self.trade_history = deque(maxlen=20)  # Store last 20 trades
        self.log_buffer = deque(maxlen=100)  # Store recent log entries
        
        # Monitoring data
        self.ai_metrics = {}
        self.system_status = {}
        self.portfolio_data = {}
        self.recent_activity = []
        
        # Views available
        self.views = {
            "1": ("dashboard", "📊 Dashboard - Overall Status"),
            "2": ("ai_learning", "🧠 AI Learning Progress"),
            "3": ("trading", "💰 Trading Performance"),
            "4": ("system", "⚙️ System Metrics"),
            "5": ("logs", "📋 Live Logs"),
            "6": ("ai_development", "🔬 AI Development Details")
        }
        
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Print monitor header"""
        print(f"{Colors.BOLD}{Colors.CYAN}")
        print("=" * 80)
        print("🚀 ENHANCED ANT BOT - AI LEARNING MONITOR")
        print("=" * 80)
        print(f"{Colors.END}")
        print(f"{Colors.YELLOW}Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}")
        print()
    
    def print_navigation(self):
        """Print navigation menu"""
        print(f"{Colors.BOLD}{Colors.WHITE}NAVIGATION:{Colors.END}")
        for key, (view_name, description) in self.views.items():
            indicator = "👉" if self.current_view == view_name else "  "
            color = Colors.GREEN if self.current_view == view_name else Colors.WHITE
            print(f"{indicator} {color}[{key}] {description}{Colors.END}")
        print(f"   {Colors.RED}[q] Quit Monitor{Colors.END}")
        print()
    
    def load_portfolio_data(self):
        """Load current portfolio data"""
        try:
            if os.path.exists('portfolio.json'):
                with open('portfolio.json', 'r') as f:
                    self.portfolio_data = json.load(f)
            else:
                self.portfolio_data = {"current_value": 0, "total_pnl": 0, "total_positions": 0}
        except Exception as e:
            self.portfolio_data = {"error": str(e)}
    
    def load_ai_metrics(self):
        """Extract AI learning metrics from logs"""
        try:
            ai_data = {
                "intelligence_score": 0.0,
                "learning_trend": "UNKNOWN",
                "model_weights": {"grok_weight": 0.5, "local_llm_weight": 0.5},
                "prediction_accuracy": 0.0,
                "total_learning_cycles": 0,
                "improvement_rate": 0.0
            }
            
            # Read main log file for AI metrics
            if os.path.exists('logs/ant_bot_main.log'):
                with open('logs/ant_bot_main.log', 'r') as f:
                    lines = f.readlines()
                    
                # Parse recent AI metrics
                for line in reversed(lines[-100:]):  # Check last 100 lines
                    if "Intelligence Score:" in line:
                        try:
                            score = float(line.split("Intelligence Score:")[1].split()[0])
                            ai_data["intelligence_score"] = score
                            break
                        except:
                            pass
                
                # Extract learning trend
                for line in reversed(lines[-50:]):
                    if "Learning Trend:" in line:
                        try:
                            trend = line.split("Learning Trend:")[1].strip().split()[0]
                            ai_data["learning_trend"] = trend
                            break
                        except:
                            pass
                
                # Extract model weights
                for line in reversed(lines[-50:]):
                    if "Grok Weight:" in line and "Local Weight:" in line:
                        try:
                            parts = line.split()
                            grok_idx = parts.index("Weight:") + 1
                            local_idx = parts.index("Weight:", grok_idx + 1) + 1
                            ai_data["model_weights"]["grok_weight"] = float(parts[grok_idx])
                            ai_data["model_weights"]["local_llm_weight"] = float(parts[local_idx])
                            break
                        except:
                            pass
                
                # Count learning cycles
                learning_cycles = sum(1 for line in lines if "Learning Cycle" in line)
                ai_data["total_learning_cycles"] = learning_cycles
            
            self.ai_metrics = ai_data
            
            # Store in history for trend analysis
            current_time = time.time()
            self.ai_history.append({
                "timestamp": current_time,
                "intelligence_score": ai_data["intelligence_score"],
                "learning_trend": ai_data["learning_trend"]
            })
            
        except Exception as e:
            self.ai_metrics = {"error": str(e)}
    
    def load_system_status(self):
        """Load system status and health"""
        try:
            # Check if bot process is running
            try:
                result = subprocess.run(['pgrep', '-f', 'python main.py'], 
                                      capture_output=True, text=True)
                bot_running = len(result.stdout.strip()) > 0
            except:
                # Windows fallback
                try:
                    result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                                          capture_output=True, text=True)
                    bot_running = 'python.exe' in result.stdout
                except:
                    bot_running = False
            
            self.system_status = {
                "bot_running": bot_running,
                "timestamp": datetime.now().isoformat(),
                "log_files_exist": {
                    "main_log": os.path.exists('logs/ant_bot_main.log'),
                    "trading_log": os.path.exists('logs/trading/trades.log'),
                    "system_log": os.path.exists('logs/monitoring/system_metrics.log')
                }
            }
            
            # Check recent activity
            if os.path.exists('logs/ant_bot_main.log'):
                with open('logs/ant_bot_main.log', 'r') as f:
                    lines = f.readlines()
                    self.recent_activity = lines[-10:] if lines else []
            
        except Exception as e:
            self.system_status = {"error": str(e)}
    
    def load_recent_trades(self):
        """Load recent trading activity"""
        try:
            trades = []
            if os.path.exists('logs/trading/trades.log'):
                with open('logs/trading/trades.log', 'r') as f:
                    lines = f.readlines()
                    for line in reversed(lines[-20:]):  # Last 20 trades
                        if any(keyword in line for keyword in ['BUY', 'SELL', 'P&L', 'Trade']):
                            trades.append(line.strip())
            
            self.trade_history.clear()
            self.trade_history.extend(trades)
            
        except Exception as e:
            self.trade_history.clear()
            self.trade_history.append(f"Error loading trades: {e}")
    
    def calculate_ai_development_stats(self):
        """Calculate AI development statistics"""
        if len(self.ai_history) < 2:
            return {
                "development_rate": 0.0,
                "learning_velocity": 0.0,
                "stability_score": 0.0,
                "trend_direction": "INSUFFICIENT_DATA"
            }
        
        # Calculate improvement rate
        recent_scores = [entry["intelligence_score"] for entry in list(self.ai_history)[-10:]]
        if len(recent_scores) >= 2:
            improvement = recent_scores[-1] - recent_scores[0]
            development_rate = improvement / len(recent_scores)
        else:
            development_rate = 0.0
        
        # Calculate learning velocity (rate of change)
        if len(recent_scores) >= 3:
            deltas = [recent_scores[i] - recent_scores[i-1] for i in range(1, len(recent_scores))]
            learning_velocity = sum(deltas) / len(deltas)
        else:
            learning_velocity = 0.0
        
        # Calculate stability (consistency of learning)
        if len(recent_scores) >= 3:
            variance = sum((x - sum(recent_scores)/len(recent_scores))**2 for x in recent_scores) / len(recent_scores)
            stability_score = max(0, 1.0 - variance)
        else:
            stability_score = 0.0
        
        # Determine trend direction
        if development_rate > 0.01:
            trend_direction = "ACCELERATING"
        elif development_rate > 0:
            trend_direction = "IMPROVING"
        elif development_rate < -0.01:
            trend_direction = "DECLINING"
        else:
            trend_direction = "STABLE"
        
        return {
            "development_rate": development_rate,
            "learning_velocity": learning_velocity,
            "stability_score": stability_score,
            "trend_direction": trend_direction
        }
    
    def show_dashboard(self):
        """Show main dashboard view"""
        print(f"{Colors.BOLD}{Colors.BLUE}📊 SYSTEM DASHBOARD{Colors.END}")
        print("=" * 60)
        
        # Bot Status
        status_color = Colors.GREEN if self.system_status.get("bot_running", False) else Colors.RED
        status_text = "RUNNING" if self.system_status.get("bot_running", False) else "STOPPED"
        print(f"🤖 Bot Status: {status_color}{status_text}{Colors.END}")
        
        # Portfolio Overview
        portfolio = self.portfolio_data
        current_value = portfolio.get("current_value", 0)
        total_pnl = portfolio.get("total_pnl", 0)
        pnl_color = Colors.GREEN if total_pnl >= 0 else Colors.RED
        print(f"💰 Portfolio Value: {Colors.YELLOW}{current_value:.4f} SOL{Colors.END}")
        print(f"📈 Total P&L: {pnl_color}{total_pnl:+.4f} SOL{Colors.END}")
        
        # AI Status
        ai_score = self.ai_metrics.get("intelligence_score", 0)
        ai_trend = self.ai_metrics.get("learning_trend", "UNKNOWN")
        trend_color = Colors.GREEN if ai_trend == "IMPROVING" else Colors.YELLOW if ai_trend == "STABLE" else Colors.RED
        print(f"🧠 AI Intelligence: {Colors.CYAN}{ai_score:.3f}{Colors.END}")
        print(f"📊 Learning Trend: {trend_color}{ai_trend}{Colors.END}")
        
        # Recent Activity Summary
        print(f"\n{Colors.BOLD}📋 Recent Activity:{Colors.END}")
        for activity in self.recent_activity[-5:]:
            timestamp = activity[:19] if len(activity) > 19 else ""
            message = activity[20:] if len(activity) > 20 else activity
            print(f"  {Colors.WHITE}{timestamp}{Colors.END} {message[:50]}...")
    
    def show_ai_learning(self):
        """Show AI learning progress view"""
        print(f"{Colors.BOLD}{Colors.PURPLE}🧠 AI LEARNING PROGRESS{Colors.END}")
        print("=" * 60)
        
        # Current AI Metrics
        ai = self.ai_metrics
        print(f"Intelligence Score: {Colors.CYAN}{ai.get('intelligence_score', 0):.4f}{Colors.END}")
        print(f"Learning Trend: {Colors.YELLOW}{ai.get('learning_trend', 'UNKNOWN')}{Colors.END}")
        print(f"Total Learning Cycles: {Colors.WHITE}{ai.get('total_learning_cycles', 0)}{Colors.END}")
        
        # Model Weights
        weights = ai.get("model_weights", {})
        grok_weight = weights.get("grok_weight", 0.5)
        local_weight = weights.get("local_llm_weight", 0.5)
        print(f"\n{Colors.BOLD}Model Balance:{Colors.END}")
        print(f"  Grok AI Weight: {Colors.GREEN}{grok_weight:.3f}{Colors.END}")
        print(f"  Local LLM Weight: {Colors.BLUE}{local_weight:.3f}{Colors.END}")
        
        # Learning History
        if len(self.ai_history) > 1:
            print(f"\n{Colors.BOLD}Learning History (Last 10):{Colors.END}")
            for entry in list(self.ai_history)[-10:]:
                timestamp = datetime.fromtimestamp(entry["timestamp"]).strftime("%H:%M:%S")
                score = entry["intelligence_score"]
                trend = entry["learning_trend"]
                print(f"  {timestamp}: {score:.4f} ({trend})")
        
        # Development Statistics
        dev_stats = self.calculate_ai_development_stats()
        print(f"\n{Colors.BOLD}Development Statistics:{Colors.END}")
        print(f"  Development Rate: {Colors.CYAN}{dev_stats['development_rate']:+.6f}{Colors.END}")
        print(f"  Learning Velocity: {Colors.YELLOW}{dev_stats['learning_velocity']:+.6f}{Colors.END}")
        print(f"  Stability Score: {Colors.GREEN}{dev_stats['stability_score']:.3f}{Colors.END}")
        print(f"  Trend Direction: {Colors.WHITE}{dev_stats['trend_direction']}{Colors.END}")
    
    def show_ai_development(self):
        """Show detailed AI development view"""
        print(f"{Colors.BOLD}{Colors.PURPLE}🔬 AI DEVELOPMENT DETAILS{Colors.END}")
        print("=" * 60)
        
        dev_stats = self.calculate_ai_development_stats()
        
        # Visual representation of learning progress
        if len(self.ai_history) >= 5:
            print(f"{Colors.BOLD}Intelligence Score Progression:{Colors.END}")
            recent_scores = [entry["intelligence_score"] for entry in list(self.ai_history)[-10:]]
            
            for i, score in enumerate(recent_scores):
                bar_length = int(score * 50)  # Scale to 50 chars
                bar = "█" * bar_length + "░" * (50 - bar_length)
                print(f"  {i+1:2d}: {Colors.CYAN}{bar}{Colors.END} {score:.4f}")
        
        # Learning Pattern Analysis
        print(f"\n{Colors.BOLD}Learning Pattern Analysis:{Colors.END}")
        if dev_stats["trend_direction"] == "ACCELERATING":
            print(f"  🚀 {Colors.GREEN}AI is learning rapidly and accelerating{Colors.END}")
        elif dev_stats["trend_direction"] == "IMPROVING":
            print(f"  📈 {Colors.YELLOW}AI is steadily improving{Colors.END}")
        elif dev_stats["trend_direction"] == "STABLE":
            print(f"  ⚖️ {Colors.BLUE}AI performance is stable{Colors.END}")
        elif dev_stats["trend_direction"] == "DECLINING":
            print(f"  📉 {Colors.RED}AI performance is declining - may need adjustment{Colors.END}")
        else:
            print(f"  ❓ {Colors.WHITE}Insufficient data for analysis{Colors.END}")
        
        # Model Evolution Insights
        if len(self.ai_history) >= 3:
            print(f"\n{Colors.BOLD}Model Evolution:{Colors.END}")
            first_score = self.ai_history[0]["intelligence_score"]
            latest_score = self.ai_history[-1]["intelligence_score"]
            total_improvement = latest_score - first_score
            improvement_pct = (total_improvement / first_score * 100) if first_score > 0 else 0
            
            print(f"  Initial Intelligence: {Colors.WHITE}{first_score:.4f}{Colors.END}")
            print(f"  Current Intelligence: {Colors.CYAN}{latest_score:.4f}{Colors.END}")
            print(f"  Total Improvement: {Colors.GREEN}{total_improvement:+.4f} ({improvement_pct:+.2f}%){Colors.END}")
    
    def show_trading(self):
        """Show trading performance view"""
        print(f"{Colors.BOLD}{Colors.GREEN}💰 TRADING PERFORMANCE{Colors.END}")
        print("=" * 60)
        
        # Portfolio Summary
        portfolio = self.portfolio_data
        print(f"Current Portfolio Value: {Colors.YELLOW}{portfolio.get('current_value', 0):.4f} SOL{Colors.END}")
        print(f"Total P&L: {Colors.GREEN}{portfolio.get('total_pnl', 0):+.4f} SOL{Colors.END}")
        print(f"Total Positions: {Colors.WHITE}{portfolio.get('total_positions', 0)}{Colors.END}")
        
        # Recent Trades
        print(f"\n{Colors.BOLD}Recent Trading Activity:{Colors.END}")
        if self.trade_history:
            for i, trade in enumerate(list(self.trade_history)[-10:], 1):
                print(f"  {i:2d}: {trade[:70]}...")
        else:
            print(f"  {Colors.YELLOW}No recent trading activity found{Colors.END}")
    
    def show_system(self):
        """Show system metrics view"""
        print(f"{Colors.BOLD}{Colors.BLUE}⚙️ SYSTEM METRICS{Colors.END}")
        print("=" * 60)
        
        # System Status
        system = self.system_status
        bot_status = "RUNNING" if system.get("bot_running", False) else "STOPPED"
        status_color = Colors.GREEN if system.get("bot_running", False) else Colors.RED
        print(f"Bot Process: {status_color}{bot_status}{Colors.END}")
        
        # Log Files Status
        logs = system.get("log_files_exist", {})
        print(f"\n{Colors.BOLD}Log Files:{Colors.END}")
        for log_name, exists in logs.items():
            status = "EXISTS" if exists else "MISSING"
            color = Colors.GREEN if exists else Colors.RED
            print(f"  {log_name}: {color}{status}{Colors.END}")
        
        # System Resources (if available)
        try:
            import psutil
            print(f"\n{Colors.BOLD}System Resources:{Colors.END}")
            print(f"  CPU Usage: {Colors.YELLOW}{psutil.cpu_percent():.1f}%{Colors.END}")
            print(f"  Memory Usage: {Colors.CYAN}{psutil.virtual_memory().percent:.1f}%{Colors.END}")
            print(f"  Disk Usage: {Colors.WHITE}{psutil.disk_usage('/').percent:.1f}%{Colors.END}")
        except ImportError:
            print(f"\n{Colors.YELLOW}Install psutil for system resource monitoring{Colors.END}")
    
    def show_logs(self):
        """Show live logs view"""
        print(f"{Colors.BOLD}{Colors.WHITE}📋 LIVE LOGS{Colors.END}")
        print("=" * 60)
        
        # Show recent log entries
        if self.recent_activity:
            print(f"{Colors.BOLD}Recent Activity (Last 15 lines):{Colors.END}")
            for line in self.recent_activity[-15:]:
                # Color code by log level
                if "ERROR" in line:
                    color = Colors.RED
                elif "WARNING" in line:
                    color = Colors.YELLOW
                elif "INFO" in line:
                    color = Colors.WHITE
                else:
                    color = Colors.WHITE
                
                print(f"{color}{line.strip()}{Colors.END}")
        else:
            print(f"{Colors.YELLOW}No recent log activity found{Colors.END}")
    
    def update_data(self):
        """Update all monitoring data"""
        self.load_portfolio_data()
        self.load_ai_metrics()
        self.load_system_status()
        self.load_recent_trades()
    
    def display_current_view(self):
        """Display the current view"""
        if self.current_view == "dashboard":
            self.show_dashboard()
        elif self.current_view == "ai_learning":
            self.show_ai_learning()
        elif self.current_view == "ai_development":
            self.show_ai_development()
        elif self.current_view == "trading":
            self.show_trading()
        elif self.current_view == "system":
            self.show_system()
        elif self.current_view == "logs":
            self.show_logs()
    
    def handle_input(self):
        """Handle user input in a separate thread"""
        import select
        import sys
        
        while self.running:
            try:
                # Non-blocking input check (Unix/Linux)
                if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                    user_input = sys.stdin.readline().strip()
                    if user_input.lower() == 'q':
                        self.running = False
                    elif user_input in self.views:
                        self.current_view = self.views[user_input][0]
            except:
                # Windows fallback - blocking input
                pass
            time.sleep(0.1)
    
    def run(self):
        """Run the monitoring interface"""
        print(f"{Colors.BOLD}{Colors.GREEN}Starting Enhanced Ant Bot Monitor...{Colors.END}")
        time.sleep(1)
        
        # Start input handler thread
        try:
            input_thread = threading.Thread(target=self.handle_input, daemon=True)
            input_thread.start()
        except:
            pass
        
        try:
            while self.running:
                self.clear_screen()
                self.print_header()
                self.print_navigation()
                
                # Update data
                self.update_data()
                
                # Display current view
                self.display_current_view()
                
                # Footer
                print(f"\n{Colors.BOLD}{Colors.CYAN}Press [1-6] to switch views, [q] to quit{Colors.END}")
                print(f"{Colors.YELLOW}Auto-refresh every {self.refresh_rate} seconds{Colors.END}")
                
                # Wait for refresh
                time.sleep(self.refresh_rate)
                
        except KeyboardInterrupt:
            pass
        finally:
            self.clear_screen()
            print(f"{Colors.BOLD}{Colors.GREEN}Enhanced Ant Bot Monitor stopped.{Colors.END}")

def main():
    """Main entry point"""
    try:
        monitor = AntBotMonitor()
        monitor.run()
    except Exception as e:
        print(f"Error starting monitor: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 