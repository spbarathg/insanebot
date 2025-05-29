#!/usr/bin/env python3
"""
Enhanced Ant Bot - Modern CLI Control Center
Beautiful terminal-based control and monitoring for the trading bot
"""

import asyncio
import logging
import sys
import argparse
import os
import json
import time
import shutil
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Enhanced Ant Bot components
from src.core.portfolio_manager import PortfolioManager
from src.core.ai.enhanced_ai_coordinator import AICoordinator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModernColors:
    """Modern color palette for terminal output"""
    # Base colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Backgrounds
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    STRIKETHROUGH = '\033[9m'
    
    # Reset
    RESET = '\033[0m'
    END = '\033[0m'
    
    # Custom theme colors
    PRIMARY = '\033[38;5;39m'      # Bright Blue
    SECONDARY = '\033[38;5;141m'   # Light Purple
    SUCCESS = '\033[38;5;46m'      # Bright Green
    WARNING = '\033[38;5;226m'     # Bright Yellow
    ERROR = '\033[38;5;196m'       # Bright Red
    INFO = '\033[38;5;87m'         # Light Cyan
    ACCENT = '\033[38;5;213m'      # Pink
    MUTED = '\033[38;5;247m'       # Gray

class ModernUI:
    """Modern UI components and utilities"""
    
    @staticmethod
    def get_terminal_size():
        """Get terminal dimensions"""
        return shutil.get_terminal_size((80, 24))
    
    @staticmethod
    def center_text(text: str, width: int = None) -> str:
        """Center text in terminal"""
        if width is None:
            width = ModernUI.get_terminal_size().columns
        return text.center(width)
    
    @staticmethod
    def create_box(content: str, title: str = "", width: int = None, style: str = "single") -> str:
        """Create a bordered box around content"""
        if width is None:
            width = min(ModernUI.get_terminal_size().columns - 4, 80)
        
        # Box drawing characters
        if style == "single":
            corners = "┌┐└┘"
            horizontal = "─"
            vertical = "│"
        elif style == "double":
            corners = "╔╗╚╝"
            horizontal = "═"
            vertical = "║"
        elif style == "rounded":
            corners = "╭╮╰╯"
            horizontal = "─"
            vertical = "│"
        else:
            corners = "++++"
            horizontal = "-"
            vertical = "|"
        
        lines = content.split('\n')
        max_content_width = width - 4
        
        # Process content lines
        wrapped_lines = []
        for line in lines:
            if len(line) <= max_content_width:
                wrapped_lines.append(line)
            else:
                # Simple word wrapping
                words = line.split()
                current_line = ""
                for word in words:
                    if len(current_line + " " + word) <= max_content_width:
                        current_line = current_line + " " + word if current_line else word
                    else:
                        if current_line:
                            wrapped_lines.append(current_line)
                        current_line = word
                if current_line:
                    wrapped_lines.append(current_line)
        
        # Build box
        result = []
        
        # Top border with title
        if title:
            title_text = f" {title} "
            remaining_width = width - len(title_text) - 2
            left_border = horizontal * (remaining_width // 2)
            right_border = horizontal * (remaining_width - len(left_border))
            result.append(f"{corners[0]}{left_border}{title_text}{right_border}{corners[1]}")
        else:
            result.append(f"{corners[0]}{horizontal * (width - 2)}{corners[1]}")
        
        # Content lines
        for line in wrapped_lines:
            padded_line = line.ljust(max_content_width)
            result.append(f"{vertical} {padded_line} {vertical}")
        
        # Bottom border
        result.append(f"{corners[2]}{horizontal * (width - 2)}{corners[3]}")
        
        return '\n'.join(result)
    
    @staticmethod
    def create_progress_bar(progress: float, width: int = 30, filled_char: str = "█", empty_char: str = "░") -> str:
        """Create a progress bar"""
        filled_width = int(width * progress)
        empty_width = width - filled_width
        
        bar = filled_char * filled_width + empty_char * empty_width
        percentage = f"{progress * 100:.1f}%"
        
        return f"[{bar}] {percentage}"
    
    @staticmethod
    def create_status_indicator(status: str, value: any = None) -> str:
        """Create a colored status indicator"""
        indicators = {
            "online": f"{ModernColors.SUCCESS}● ONLINE{ModernColors.RESET}",
            "offline": f"{ModernColors.ERROR}● OFFLINE{ModernColors.RESET}",
            "warning": f"{ModernColors.WARNING}● WARNING{ModernColors.RESET}",
            "info": f"{ModernColors.INFO}● INFO{ModernColors.RESET}",
            "success": f"{ModernColors.SUCCESS}✓ SUCCESS{ModernColors.RESET}",
            "error": f"{ModernColors.ERROR}✗ ERROR{ModernColors.RESET}",
            "loading": f"{ModernColors.INFO}⟳ LOADING{ModernColors.RESET}",
            "running": f"{ModernColors.SUCCESS}▶ RUNNING{ModernColors.RESET}",
            "stopped": f"{ModernColors.MUTED}⏸ STOPPED{ModernColors.RESET}",
        }
        
        base_indicator = indicators.get(status, f"{ModernColors.MUTED}◦ {status.upper()}{ModernColors.RESET}")
        
        if value is not None:
            return f"{base_indicator} {value}"
        return base_indicator

class EnhancedAntBotCLI:
    """Modern CLI interface for Enhanced Ant Bot"""
    
    def __init__(self):
        self.portfolio_manager = None
        self.ai_coordinator = None
        self.running = True
        self.terminal_width = ModernUI.get_terminal_size().columns
        
    async def initialize(self):
        """Initialize bot components with progress indication"""
        try:
            self.clear_screen()
            print(f"\n{ModernColors.PRIMARY}{ModernColors.BOLD}")
            print(ModernUI.center_text("🚀 ENHANCED ANT BOT INITIALIZATION"))
            print(f"{ModernColors.RESET}\n")
            
            # Step 1: Portfolio Manager
            print(f"{ModernColors.INFO}📊 Initializing Portfolio Manager...{ModernColors.RESET}")
            progress_bar = ModernUI.create_progress_bar(0.3, 40)
            print(f"   {progress_bar}")
            time.sleep(0.5)
            
            self.portfolio_manager = PortfolioManager()
            self.portfolio_manager.initialize(0.1)
            
            # Step 2: AI Coordinator
            print(f"\n{ModernColors.INFO}🧠 Initializing AI Coordinator...{ModernColors.RESET}")
            progress_bar = ModernUI.create_progress_bar(0.7, 40)
            print(f"   {progress_bar}")
            time.sleep(0.5)
            
            self.ai_coordinator = AICoordinator()
            await self.ai_coordinator.initialize()
            
            # Completion
            print(f"\n{ModernColors.SUCCESS}✅ System Ready{ModernColors.RESET}")
            progress_bar = ModernUI.create_progress_bar(1.0, 40)
            print(f"   {progress_bar}\n")
            
            time.sleep(1)
            return True
            
        except Exception as e:
            print(f"\n{ModernColors.ERROR}❌ Initialization failed: {str(e)}{ModernColors.RESET}")
            return False
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Print modern CLI header"""
        self.clear_screen()
        
        # Header with gradient-like effect
        header_lines = [
            f"{ModernColors.PRIMARY}{'█' * self.terminal_width}",
            f"{ModernColors.BRIGHT_BLUE}{'▓' * self.terminal_width}",
            f"{ModernColors.CYAN}{'▒' * self.terminal_width}",
            f"{ModernColors.BRIGHT_CYAN}{'░' * self.terminal_width}",
        ]
        
        for line in header_lines:
            print(f"{line}{ModernColors.RESET}")
        
        # Title
        title = "🐜 ENHANCED ANT BOT - CONTROL CENTER"
        print(f"\n{ModernColors.BOLD}{ModernColors.WHITE}")
        print(ModernUI.center_text(title))
        print(f"{ModernColors.RESET}")
        
        # Subtitle with time
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        subtitle = f"AI-Driven Solana Trading System │ {current_time}"
        print(f"{ModernColors.MUTED}")
        print(ModernUI.center_text(subtitle))
        print(f"{ModernColors.RESET}\n")
    
    def print_main_menu(self):
        """Print modern main menu"""
        menu_items = [
            ("1", "📊", "Portfolio Status", "View holdings, P&L, and trading statistics"),
            ("2", "🧠", "AI System Status", "Monitor AI performance and learning progress"),
            ("3", "📈", "Trading Controls", "Manage trades, risk settings, and analysis"),
            ("4", "🔧", "System Settings", "Configure bot parameters and preferences"),
            ("5", "📋", "View Logs", "Access real-time logging and activity"),
            ("6", "🎯", "Market Analysis", "Perform market research and token analysis"),
            ("7", "🚀", "Start Trading Bot", "Launch automated trading operations"),
            ("8", "⏹️", "Stop Trading Bot", "Halt trading operations safely"),
        ]
        
        # Create menu box
        menu_content = []
        for key, icon, title, description in menu_items:
            menu_line = f"{ModernColors.PRIMARY}[{key}]{ModernColors.RESET} {icon} {ModernColors.BOLD}{title}{ModernColors.RESET}"
            menu_content.append(menu_line)
            menu_content.append(f"    {ModernColors.MUTED}{description}{ModernColors.RESET}")
            menu_content.append("")  # Empty line for spacing
        
        # Remove last empty line
        if menu_content and menu_content[-1] == "":
            menu_content.pop()
        
        # Add quit option
        menu_content.append("")
        menu_content.append(f"{ModernColors.ERROR}[q]{ModernColors.RESET} 🚪 {ModernColors.BOLD}Quit Application{ModernColors.RESET}")
        menu_content.append(f"    {ModernColors.MUTED}Exit the Enhanced Ant Bot CLI{ModernColors.RESET}")
        
        menu_box = ModernUI.create_box(
            "\n".join(menu_content),
            title="MAIN MENU",
            style="rounded"
        )
        
        print(f"{ModernColors.BRIGHT_BLUE}{menu_box}{ModernColors.RESET}")
        print()
    
    async def show_portfolio_status(self):
        """Display modern portfolio status"""
        self.print_header()
        
        try:
            # Get portfolio data
            summary = self.portfolio_manager.get_portfolio_summary()
            holdings = self.portfolio_manager.get_holdings()
            
            # Portfolio overview
            overview_content = []
            balance = summary.get('current_balance', 0)
            pnl = summary.get('total_pnl', 0)
            trades = summary.get('total_trades', 0)
            positions = len(holdings)
            
            overview_content.append(f"{ModernColors.SUCCESS}💰 Current Balance:{ModernColors.RESET} {balance:.4f} SOL")
            overview_content.append(f"{ModernColors.WARNING if pnl < 0 else ModernColors.SUCCESS}📈 Total P&L:{ModernColors.RESET} {pnl:+.4f} SOL")
            overview_content.append(f"{ModernColors.INFO}🎯 Active Positions:{ModernColors.RESET} {positions}")
            overview_content.append(f"{ModernColors.PRIMARY}📊 Total Trades:{ModernColors.RESET} {trades}")
            
            overview_box = ModernUI.create_box(
                "\n".join(overview_content),
                title="PORTFOLIO OVERVIEW",
                style="double"
            )
            
            print(f"{ModernColors.BRIGHT_GREEN}{overview_box}{ModernColors.RESET}\n")
            
            # Holdings details
            if holdings:
                holdings_content = []
                for i, holding in enumerate(holdings, 1):
                    symbol = holding.get('symbol', 'UNKNOWN')
                    amount = holding.get('amount', 0)
                    entry_price = holding.get('entry_price', 0)
                    current_price = holding.get('current_price', 0)
                    pnl_amount = holding.get('pnl', 0)
                    pnl_pct = holding.get('pnl_percentage', 0) * 100
                    
                    pnl_color = ModernColors.SUCCESS if pnl_amount >= 0 else ModernColors.ERROR
                    
                    holdings_content.append(f"{ModernColors.BOLD}{i}. {symbol}{ModernColors.RESET}")
                    holdings_content.append(f"   Amount: {amount:.2f} tokens")
                    holdings_content.append(f"   Entry: ${entry_price:.6f} │ Current: ${current_price:.6f}")
                    holdings_content.append(f"   {pnl_color}P&L: {pnl_amount:+.4f} SOL ({pnl_pct:+.2f}%){ModernColors.RESET}")
                    
                    # Mini progress bar for P&L
                    if pnl_pct != 0:
                        bar_progress = min(abs(pnl_pct) / 100, 1.0)
                        bar_char = "█" if pnl_amount >= 0 else "▓"
                        progress_bar = ModernUI.create_progress_bar(bar_progress, 20, bar_char, "░")
                        holdings_content.append(f"   {pnl_color}{progress_bar}{ModernColors.RESET}")
                    
                    holdings_content.append("")
                
                holdings_box = ModernUI.create_box(
                    "\n".join(holdings_content[:-1]),  # Remove last empty line
                    title="CURRENT HOLDINGS",
                    style="single"
                )
                
                print(f"{ModernColors.BRIGHT_YELLOW}{holdings_box}{ModernColors.RESET}")
            else:
                no_holdings_box = ModernUI.create_box(
                    f"{ModernColors.MUTED}No active positions{ModernColors.RESET}",
                    title="CURRENT HOLDINGS",
                    style="single"
                )
                print(f"{ModernColors.BRIGHT_YELLOW}{no_holdings_box}{ModernColors.RESET}")
                
        except Exception as e:
            error_box = ModernUI.create_box(
                f"{ModernColors.ERROR}Error loading portfolio: {str(e)}{ModernColors.RESET}",
                title="ERROR",
                style="single"
            )
            print(f"{ModernColors.BRIGHT_RED}{error_box}{ModernColors.RESET}")
        
        print(f"\n{ModernColors.MUTED}Press Enter to continue...{ModernColors.RESET}", end="")
        input()
    
    async def show_ai_status(self):
        """Display modern AI system status"""
        self.print_header()
        
        try:
            # Get AI performance data
            performance = self.ai_coordinator.get_performance_summary()
            
            # AI Overview
            overview_content = []
            intelligence_score = performance.get('overall_accuracy', 0)
            total_decisions = performance.get('total_decisions', 0)
            
            # Intelligence score with progress bar
            score_bar = ModernUI.create_progress_bar(intelligence_score, 30)
            overview_content.append(f"{ModernColors.SUCCESS}🎯 Intelligence Score:{ModernColors.RESET}")
            overview_content.append(f"   {score_bar}")
            overview_content.append("")
            overview_content.append(f"{ModernColors.INFO}📊 Total Decisions:{ModernColors.RESET} {total_decisions}")
            overview_content.append(f"{ModernColors.PRIMARY}🔄 Learning Status:{ModernColors.RESET} {ModernUI.create_status_indicator('running')}")
            
            overview_box = ModernUI.create_box(
                "\n".join(overview_content),
                title="AI SYSTEM OVERVIEW",
                style="double"
            )
            
            print(f"{ModernColors.BRIGHT_MAGENTA}{overview_box}{ModernColors.RESET}\n")
            
            # Model Weights
            weights_content = []
            weights = performance.get('model_weights', {})
            
            for role, weight in weights.items():
                weight_bar = ModernUI.create_progress_bar(weight, 25)
                weights_content.append(f"{ModernColors.ACCENT}{role}:{ModernColors.RESET}")
                weights_content.append(f"   {weight_bar}")
                weights_content.append("")
            
            if weights_content:
                weights_content.pop()  # Remove last empty line
                
                weights_box = ModernUI.create_box(
                    "\n".join(weights_content),
                    title="MODEL WEIGHTS",
                    style="single"
                )
                
                print(f"{ModernColors.BRIGHT_CYAN}{weights_box}{ModernColors.RESET}\n")
            
            # Performance Metrics
            perf_content = []
            model_performance = performance.get('model_performance', {})
            
            for role, metrics in model_performance.items():
                accuracy = metrics.get('accuracy', 0)
                recent_acc = metrics.get('recent_accuracy_rate', 0)
                
                # Determine status color
                if recent_acc > 0.7:
                    status_color = ModernColors.SUCCESS
                    status_icon = "🔥"
                elif recent_acc > 0.5:
                    status_color = ModernColors.WARNING
                    status_icon = "⚡"
                else:
                    status_color = ModernColors.ERROR
                    status_icon = "🔧"
                
                perf_content.append(f"{status_icon} {ModernColors.BOLD}{role}{ModernColors.RESET}")
                perf_content.append(f"   Overall: {ModernUI.create_progress_bar(accuracy, 20)}")
                perf_content.append(f"   Recent:  {status_color}{ModernUI.create_progress_bar(recent_acc, 20)}{ModernColors.RESET}")
                perf_content.append("")
            
            if perf_content:
                perf_content.pop()  # Remove last empty line
                
                perf_box = ModernUI.create_box(
                    "\n".join(perf_content),
                    title="PERFORMANCE METRICS",
                    style="rounded"
                )
                
                print(f"{ModernColors.BRIGHT_BLUE}{perf_box}{ModernColors.RESET}")
                
        except Exception as e:
            error_box = ModernUI.create_box(
                f"{ModernColors.ERROR}Error loading AI status: {str(e)}{ModernColors.RESET}",
                title="ERROR",
                style="single"
            )
            print(f"{ModernColors.BRIGHT_RED}{error_box}{ModernColors.RESET}")
        
        print(f"\n{ModernColors.MUTED}Press Enter to continue...{ModernColors.RESET}", end="")
        input()
    
    async def main_loop(self):
        """Modern main CLI loop"""
        while self.running:
            self.print_header()
            self.print_main_menu()
            
            # Styled input prompt
            prompt = f"{ModernColors.PRIMARY}❯{ModernColors.RESET} {ModernColors.BOLD}Select option:{ModernColors.RESET} "
            choice = input(prompt).strip().lower()
            
            if choice == '1':
                await self.show_portfolio_status()
            elif choice == '2':
                await self.show_ai_status()
            elif choice == '3':
                await self.show_placeholder("TRADING CONTROLS")
            elif choice == '4':
                await self.show_placeholder("SYSTEM SETTINGS")
            elif choice == '5':
                await self.show_placeholder("VIEW LOGS")
            elif choice == '6':
                await self.show_placeholder("MARKET ANALYSIS")
            elif choice == '7':
                await self.show_placeholder("START TRADING BOT")
            elif choice == '8':
                await self.show_placeholder("STOP TRADING BOT")
            elif choice == 'q':
                await self.show_goodbye()
                self.running = False
            else:
                await self.show_invalid_option()
    
    async def show_placeholder(self, feature_name: str):
        """Show a modern placeholder for features"""
        self.print_header()
        
        placeholder_content = [
            f"{ModernColors.INFO}🚧 Feature Under Development{ModernColors.RESET}",
            "",
            f"The {feature_name} feature is being enhanced",
            "with modern UI components and will be available soon.",
            "",
            f"{ModernColors.ACCENT}Coming Soon:{ModernColors.RESET}",
            "• Advanced interactive controls",
            "• Real-time data visualization",
            "• Enhanced user experience",
            "",
            f"{ModernColors.MUTED}Thank you for your patience!{ModernColors.RESET}"
        ]
        
        placeholder_box = ModernUI.create_box(
            "\n".join(placeholder_content),
            title=feature_name,
            style="rounded"
        )
        
        print(f"{ModernColors.BRIGHT_YELLOW}{placeholder_box}{ModernColors.RESET}")
        print(f"\n{ModernColors.MUTED}Press Enter to continue...{ModernColors.RESET}", end="")
        input()
    
    async def show_invalid_option(self):
        """Show invalid option message"""
        self.clear_screen()
        
        error_content = [
            f"{ModernColors.ERROR}❌ Invalid Option{ModernColors.RESET}",
            "",
            "Please select a valid option from the menu.",
            "",
            f"{ModernColors.MUTED}Valid options: 1-8, q{ModernColors.RESET}"
        ]
        
        error_box = ModernUI.create_box(
            "\n".join(error_content),
            title="ERROR",
            style="single"
        )
        
        print(f"\n{ModernColors.BRIGHT_RED}{error_box}{ModernColors.RESET}")
        time.sleep(2)
    
    async def show_goodbye(self):
        """Show modern goodbye message"""
        self.clear_screen()
        
        goodbye_content = [
            f"{ModernColors.SUCCESS}✨ Thank you for using Enhanced Ant Bot!{ModernColors.RESET}",
            "",
            "Your AI-driven trading companion is always ready",
            "to help you navigate the Solana ecosystem.",
            "",
            f"{ModernColors.PRIMARY}🚀 Happy Trading!{ModernColors.RESET}",
            "",
            f"{ModernColors.MUTED}May your profits be high and your risks be low.{ModernColors.RESET}"
        ]
        
        goodbye_box = ModernUI.create_box(
            "\n".join(goodbye_content),
            title="GOODBYE",
            style="double"
        )
        
        print(f"\n{ModernColors.BRIGHT_GREEN}{goodbye_box}{ModernColors.RESET}\n")
        time.sleep(2)

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Enhanced Ant Bot - Modern CLI Control Center")
    parser.add_argument("--test", action="store_true", help="Test mode - just initialize and exit")
    args = parser.parse_args()
    
    cli = EnhancedAntBotCLI()
    
    if not await cli.initialize():
        print(f"{ModernColors.ERROR}Failed to initialize. Exiting.{ModernColors.RESET}")
        return
    
    if args.test:
        print(f"{ModernColors.SUCCESS}✅ Test mode - CLI initialized successfully!{ModernColors.RESET}")
        return
    
    try:
        await cli.main_loop()
    except KeyboardInterrupt:
        print(f"\n{ModernColors.WARNING}👋 Interrupted by user{ModernColors.RESET}")
    except Exception as e:
        print(f"{ModernColors.ERROR}❌ Error: {str(e)}{ModernColors.RESET}")

if __name__ == "__main__":
    asyncio.run(main()) 