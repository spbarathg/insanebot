#!/usr/bin/env python3
"""
Simple launcher for Enhanced Ant Bot monitoring
"""

import sys
import subprocess
import os

def main():
    """Launch the appropriate monitor"""
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # Launch full CLI monitor
        subprocess.run([sys.executable, "monitor_cli.py"])
    else:
        # Quick status check
        print("🤖 Enhanced Ant Bot - Quick Status")
        print("=" * 40)
        
        # Check if bot is running
        try:
            if os.name == 'nt':  # Windows
                result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                                      capture_output=True, text=True)
                bot_running = 'python.exe' in result.stdout
            else:  # Unix/Linux
                result = subprocess.run(['pgrep', '-f', 'python main.py'], 
                                      capture_output=True, text=True)
                bot_running = len(result.stdout.strip()) > 0
            
            status = "🟢 RUNNING" if bot_running else "🔴 STOPPED"
            print(f"Bot Status: {status}")
        except:
            print("Bot Status: ❓ UNKNOWN")
        
        # Check portfolio
        try:
            import json
            if os.path.exists('portfolio.json'):
                with open('portfolio.json', 'r') as f:
                    portfolio = json.load(f)
                print(f"Portfolio Value: {portfolio.get('current_value', 0):.4f} SOL")
                print(f"Total P&L: {portfolio.get('total_pnl', 0):+.4f} SOL")
            else:
                print("Portfolio: No data available")
        except:
            print("Portfolio: Error reading data")
        
        # Check AI intelligence
        try:
            if os.path.exists('logs/ant_bot_main.log'):
                with open('logs/ant_bot_main.log', 'r') as f:
                    lines = f.readlines()
                    
                # Find latest intelligence score
                for line in reversed(lines[-50:]):
                    if "Intelligence Score:" in line:
                        try:
                            score = float(line.split("Intelligence Score:")[1].split()[0])
                            print(f"AI Intelligence: {score:.4f}")
                            break
                        except:
                            pass
                else:
                    print("AI Intelligence: No data found")
            else:
                print("AI Intelligence: Log file not found")
        except:
            print("AI Intelligence: Error reading logs")
        
        print("\n📊 For detailed monitoring, run:")
        print("   python monitor.py --cli")
        print("\n📋 For specific views:")
        print("   tail -f logs/ant_bot_main.log     # Live activity")
        print("   grep 'Intelligence Score' logs/ant_bot_main.log | tail -10  # AI progress")

if __name__ == "__main__":
    main() 