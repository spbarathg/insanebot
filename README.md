# Solana Trading Bot

A minimalistic and robust Solana trading bot with comprehensive error handling and testing.

## Features

- Real-time token monitoring
- Automated trading with configurable parameters
- Simulation mode for testing
- Comprehensive error handling
- Extensive test coverage
- Minimalistic design

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with the following variables:
```env
# Core Trading Settings
MIN_LIQUIDITY=10000
MAX_SLIPPAGE=0.02
MIN_PROFIT=0.05
MAX_POSITION=0.1
MIN_POSITION=0.01
COOLDOWN=300

# Monitoring Settings
CHECK_INTERVAL=60
MAX_RETRIES=3
RETRY_DELAY=5

# Error Handling
MAX_ERRORS=5
ERROR_COOLDOWN=300

# Market Settings
VOLATILITY=0.1
MIN_HOLDERS=100
MAX_TOKEN_AGE=30

# Trading Settings
STOP_LOSS=0.05
TAKE_PROFIT=0.1
MAX_TRADES=5

# RPC Settings
RPC_URL=https://api.mainnet-beta.solana.com
RPC_COMMITMENT=confirmed
```

3. Configure your wallet:
   - Set your wallet address in `main.py`
   - Ensure sufficient SOL balance
   - Test in simulation mode first

## Running the Bot

1. Simulation mode (recommended for testing):
```bash
python src/main.py
```

2. Live trading mode:
   - Set `simulation_mode=False` in `main.py`
   - Ensure proper wallet configuration
   - Run with sufficient SOL balance

## Testing

Run the test suite:
```bash
pytest
```

View test coverage:
```bash
pytest --cov=src --cov-report=html
```

## Configuration

The bot uses three main configuration sections:

1. Core Configuration (`CORE_CONFIG`):
   - Trading parameters
   - Monitoring settings
   - Error handling thresholds

2. Market Configuration (`MARKET_CONFIG`):
   - Liquidity requirements
   - Volatility thresholds
   - Token criteria

3. Trading Configuration (`TRADING_CONFIG`):
   - Position sizes
   - Stop loss and take profit
   - Trade limits

## Error Handling

The bot includes comprehensive error handling:
- Automatic retry for transient errors
- Critical error detection
- Trading suspension on repeated failures
- Detailed error logging

## Safety Features

1. Position Limits:
   - Maximum position size
   - Minimum liquidity requirements
   - Cooldown periods between trades

2. Risk Management:
   - Stop loss orders
   - Take profit targets
   - Maximum concurrent trades

3. Error Prevention:
   - Input validation
   - Balance checks
   - Transaction verification

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License
