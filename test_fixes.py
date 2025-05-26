import asyncio
import sys
import os
sys.path.append('.')

from src.core.portfolio_manager import PortfolioManager
from src.core.portfolio_risk_manager import PortfolioRiskManager

async def test_fixes():
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
    
    # Test risk manager initialization
    try:
        risk_manager = PortfolioRiskManager(pm)
        await risk_manager.initialize()
        print('Risk Manager initialization: SUCCESS')
    except Exception as e:
        print(f'Risk Manager initialization: FAILED - {str(e)}')

if __name__ == "__main__":
    asyncio.run(test_fixes()) 