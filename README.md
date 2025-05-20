# Solana Trading Bot

A high-frequency trading bot for Solana utilizing Jupiter DEX, Helius API, and AI-driven analytics.

## Features

- **Automated Trading**
  - Real-time trading on Jupiter DEX with sub-second execution
  - Risk management with position sizing and stop-losses
  - Configurable trading strategies

- **Market Analysis**
  - Token discovery via Helius API
  - Liquidity and volume analysis
  - Price impact prediction

- **AI Components**
  - Local LLM for market prediction
  - Continuous learning from trade outcomes
  - Ant-Queen architecture for parallel prediction
  - Sentiment analysis integration
  - Conversational AI with millennial-style personality
  
- **Comprehensive Monitoring**
  - Complete LGTM stack (Loki, Grafana, Tempo/Prometheus, Mimir)
  - Real-time performance dashboard
  - AI model accuracy tracking
  - API latency monitoring

## Quick Start

### Prerequisites
- Python 3.10+ (3.13 may have compatibility issues)
- Docker and Docker Compose
- Solana wallet with SOL
- Helius API key from [helius.xyz](https://helius.xyz/)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/solana-trading-bot.git
cd solana-trading-bot

# Setup environment
cp env.example .env
# Edit .env with your API keys and settings

# Create config directories
mkdir -p config/prometheus config/grafana/provisioning/dashboards config/loki config/promtail

# Deploy with Docker
docker-compose up -d
```

### Access Dashboards

- **Trading Dashboard**: http://localhost:3000 
  - Default credentials: admin/admin

## Configuration

### Trading Parameters (.env)
- `SIMULATION_MODE=True` - Test without real trading
- `MIN_LIQUIDITY=1.0` - Minimum token liquidity
- `MAX_PRICE_IMPACT=0.05` - Maximum acceptable price impact
- `TARGET_PROFIT=10.0` - Take profit percentage
- `STOP_LOSS=5.0` - Stop loss percentage
- `DAILY_LOSS_LIMIT=1.0` - Maximum daily loss in SOL

### AI Parameters
- `USE_LOCAL_LLM=True` - Enable local LLM predictions
- `MIN_CONFIDENCE=0.7` - Minimum confidence for execution

## Monitoring

The integrated LGTM stack provides:

- **Trading Performance**
  - Win rate, profit/loss tracking
  - Transaction execution speed monitoring
  
- **AI Performance**
  - Model accuracy visualization
  - Learning curve tracking
  - Prediction confidence analysis
  
- **System Health**
  - CPU/Memory/Disk usage
  - API performance metrics
  - Error rate monitoring

## Chat with Your Bot

The bot includes a conversational interface with a millennial personality:

```bash
# Start the chat interface
python src/cli.py
```

- **Casual Language**: Speaks with modern expressions and emojis
- **Market Insights**: Automatically shares relevant market updates
- **Trading Assistance**: Discuss strategies and performance
- **Adaptive Personality**: Mood and tone adapt based on market conditions and trading success
- **Humor**: Occasional memes and trading jokes

The local LLM continuously improves its conversational abilities and trading insights as it learns from interactions and trading outcomes.

## Deployment

### Local Development
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### Production Deployment
```bash
# Run setup verification
python test_config.py

# Start services
docker-compose up -d

# Monitor logs
docker-compose logs -f trading-bot
```

### Digital Ocean Hosting
Recommended configuration:
- Basic Droplet (Standard Plan): 4GB RAM / 2 vCPUs

## Architecture

```
├── data/               # Historical data, trade records
├── logs/               # Application logs
├── models/             # LLM model storage
├── src/
│   ├── core/           # Core trading components
│   │   ├── ai/         # AI prediction models
│   │   ├── chat/       # Conversational interface
│   │   ├── market/     # Market data analysis
│   │   └── trading/    # Trading execution
│   ├── monitoring/     # Performance monitoring
│   └── utils/          # Helper utilities
├── config/             # Configuration files
│   ├── prometheus/     # Metrics configuration
│   ├── grafana/        # Dashboard configuration
│   ├── loki/           # Log aggregation
│   └── promtail/       # Log shipping
└── docker-compose.yml  # Container orchestration
```

## Disclaimer

This software is for educational purposes only. Use at your own risk. Trading cryptocurrencies involves significant risk of loss.

## Production Deployment Guide

This bot has been optimized for production use with the following features:

- Real-time Solana trading via Jupiter DEX
- Market scanning via Helius API
- Local LLM for predictions with continuous learning
- Complete monitoring stack (Grafana, Loki, Prometheus)
- Resource-optimized Docker containers

### System Requirements

**Recommended Configuration (Option B):**
- 8GB+ RAM
- 4+ vCPUs
- 50GB SSD storage
- Ubuntu 20.04 LTS or later

### Quick Start

1. Clone the repository
2. Copy `env.example` to `.env` and add your API keys
3. Download the quantized Mistral-7B model to the models directory:
   ```
   wget -O models/mistral-7b-v0.1.Q4_K_M.gguf https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf
   ```
4. Run the deployment:
   ```
   docker-compose up -d
   ```
5. Access Grafana at http://localhost:3000 (default creds: admin/admin)

### Resource Optimization

- All containers have appropriate resource limits
- Prometheus data retention set to 15 days
- Loki log retention set to 7 days
- LLM batch processing to minimize RAM usage
- Model quantization (4-bit) for efficient inference

### Monitoring

All critical metrics are available in Grafana:
- Trading performance
- System resources
- LLM accuracy and confidence
- API response times

### AI Implementation

The bot uses a quantized Mistral-7B model for market analysis:
- Reduced memory footprint (3-4GB)
- Batched inference for prediction
- Scheduled training during low-activity periods
- Fallback to rule-based analysis if LLM unavailable

### Maintenance

Regular maintenance tasks:
- Check logs directory for growing log files
- Monitor Prometheus and Loki data volume growth
- Review model performance metrics
- Test system with `python test_config.py`

### Troubleshooting

Common issues:
- If the bot fails to start, check the `.env` file
- If Prometheus metrics are missing, check container health
- For LLM-related issues, check the model file exists
- Use `docker-compose logs` to diagnose container issues

## License

See LICENSE file for details.
