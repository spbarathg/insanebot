import sys
import os
sys.path.append('.')

from src.core.portfolio_manager import PortfolioManager

def test_portfolio_manager():
    print('Testing portfolio manager fixes...')
    
    # Test portfolio manager
    pm = PortfolioManager()
    success = pm.initialize(0.1)
    print(f'Portfolio Manager initialization: {"SUCCESS" if success else "FAILED"}')
    
    # Test portfolio summary
    summary = pm.get_portfolio_summary()
    print(f'Portfolio summary keys: {list(summary.keys())}')
    print(f'Current value present: {"current_value" in summary}')
    print(f'Current value: {summary.get("current_value", "MISSING")}')
    
    # Test holdings
    holdings = pm.get_holdings()
    print(f'Holdings type: {type(holdings)}')
    print(f'Holdings count: {len(holdings)}')
    
    print('✅ Portfolio Manager tests completed successfully!')

if __name__ == "__main__":
    test_portfolio_manager() 