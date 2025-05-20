# Solana Trading Bot

A comprehensive trading bot for Solana utilizing Jupiter and Helius APIs.

## Features

- Automated trading on Jupiter DEX
- Token analysis via Helius API
- Risk management and position sizing
- Transaction monitoring and history tracking
- Optional local LLM integration for market analysis
- Simulation mode for testing strategies

## Prerequisites

- Python 3.10+ (3.13 may have compatibility issues with solana-py)
- Solana wallet with SOL
- Helius API key (https://helius.xyz/)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/solana-trading-bot.git
cd solana-trading-bot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file from the example:
```bash
cp env.example .env
```

5. Edit the `.env` file with your API keys and settings.

## Directory Structure

```
├── data/               # Data storage (trade history, watchlist)
├── logs/               # Application logs
├── models/             # LLM model storage (if using local LLM)
├── src/                # Source code
│   ├── core/           # Core components
│   │   ├── helius_service.py    # Helius API integration
│   │   ├── jupiter_service.py   # Jupiter API integration
│   │   ├── local_llm.py         # Local LLM for analysis
│   │   ├── main.py              # Main bot implementation
│   │   ├── market_data.py       # Market data collection
│   │   ├── trade_execution.py   # Trade execution
│   │   └── wallet_manager.py    # Wallet operations
│   └── utils/          # Utilities
│       ├── config.py           # Configuration
│       └── logging_config.py   # Logging setup
├── .env                # Environment variables (create from env.example)
├── env.example         # Example environment file
├── main.py             # Entry point
└── requirements.txt    # Dependencies
```

## Usage

1. Start the bot:
```bash
python main.py
```

2. For simulation mode (no real trading), set `SIMULATION_MODE=True` in your `.env` file.

## Trading Strategies

The bot supports both rule-based and ML-based (if using local LLM) trading strategies:

### Rule-based Strategy
- Buy tokens with high liquidity (above threshold in settings)
- Set stop-loss and take-profit levels based on settings
- Scale position size based on liquidity and risk parameters

### ML-based Strategy (Optional)
- Use local LLM to analyze market data and make predictions
- Learn from trade outcomes to improve future decisions
- Adjust confidence levels based on historical performance

## Configuration

All trading parameters can be adjusted in the `.env` file:

- `MIN_LIQUIDITY`: Minimum liquidity threshold for trading
- `MAX_PRICE_IMPACT`: Maximum allowed price impact
- `TARGET_PROFIT`: Target profit percentage
- `STOP_LOSS`: Stop loss percentage
- `DAILY_LOSS_LIMIT`: Maximum daily loss in SOL
- `POSITION_SIZE` parameters: Control trade sizes

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Use at your own risk. Trading cryptocurrencies involves significant risk of loss.
