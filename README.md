# Solana Memecoin Trading Bot

An advanced trading bot for Solana memecoins with technical analysis, risk management, and portfolio tracking.

## Features

- **Technical Analysis**: RSI, MACD, Bollinger Bands, momentum detection, and trend analysis
- **Risk Management**: Assess token risk based on liquidity, market cap, volume, and holders
- **Portfolio Tracking**: Track performance, profit/loss, and holdings
- **Market Scanning**: Automatically discover new trading opportunities
- **Simulation Mode**: Test strategies without risking real funds

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/solana-memecoin-bot.git
   cd solana-memecoin-bot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys (optional for simulation mode):
   ```
   HELIUS_API_KEY=your_helius_api_key
   SOLANA_PRIVATE_KEY=your_solana_private_key
   SIMULATION_MODE=true
   SIMULATION_CAPITAL=1.0
   ```

## Usage

Run the bot in simulation mode:
```
python run_bot.py --balance 1.0 --log-level info
```

### Command Line Options

- `--simulation`: Run in simulation mode (default: True)
- `--balance`: Initial balance for simulation in SOL (default: 1.0)
- `--log-level`: Logging level (choices: debug, info, warning, error, critical)
- `--log-file`: Log file path (default: logs/trading_bot.log)
- `--config`: Configuration file path (default: config.json)
- `--no-color`: Disable colored output

## Architecture

The bot consists of several key components:

- **MemeCoinBot**: Main bot implementation that orchestrates all components
- **HeliusService**: Interface to Solana blockchain data (simulated in test mode)
- **JupiterService**: Interface to Jupiter aggregator for token swaps (simulated in test mode)
- **LocalLLM**: Advanced trading algorithm using technical indicators
- **WalletManager**: Manages wallet operations (simulated in test mode)
- **PortfolioManager**: Tracks portfolio performance and holdings
- **MarketScanner**: Discovers and evaluates new trading opportunities

## Configuration

You can customize the bot's behavior by editing the `config.json` file:

```json
{
  "risk_limit": 0.1,
  "max_concurrent_trades": 5,
  "min_trade_interval": 600,
  "sentiment_threshold": 0.3
}
```

## Disclaimer

This bot is for educational purposes only. Trading cryptocurrencies involves significant risk. The bot does not guarantee profits and should not be used with funds you cannot afford to lose.

## License

MIT
