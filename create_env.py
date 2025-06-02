#!/usr/bin/env python3
"""
Create clean production .env file
"""

env_content = '''# Production Environment Configuration - Solana Trading Bot
ENVIRONMENT=production
SIMULATION_MODE=false
NETWORK=mainnet-beta

# Trading Configuration  
INITIAL_CAPITAL=0.5
MAX_POSITION_SIZE=0.015
MAX_DAILY_LOSS=0.03
EMERGENCY_STOP_LOSS=0.10
MAX_CONCURRENT_TRADES=2

# Security Configuration
WALLET_ENCRYPTION=true
MASTER_ENCRYPTION_KEY=dGhpc19pc19hX3NlY3VyZV9lbmNyeXB0aW9uX2tleV8zMl9ieXRlcw==
SECURITY_AUDIT_MODE=true
API_AUTH_TOKEN=secure_api_token_32_chars_long_abcd1234
MONITORING_AUTH_TOKEN=secure_monitor_token_32_chars_efgh5678

# Safety & Compliance
EMERGENCY_STOP_ENABLED=true
PANIC_SELL_ENABLED=true
TRADING_CIRCUIT_BREAKERS=true
RAPID_LOSS_PROTECTION=true
AUDIT_LOGGING=true
TRADE_LOGGING=true
COMPLIANCE_MODE=true

# Infrastructure
CONTAINER_MODE=true
API_HOST=0.0.0.0
API_PORT=8080

# Logging
LOG_LEVEL=INFO
STRUCTURED_LOGGING=true
LOG_TO_FILE=true

# Features
TITAN_SHIELD_ENABLED=true
ANT_COLONY_ENABLED=true
MEV_PROTECTION=true
AI_LEARNING_ENABLED=true
SMART_MONEY_TRACKING=true

# Disabled Features
EXPERIMENTAL_FEATURES=false
DEBUG_MODE=false

# Health Checks
HEALTH_CHECK_INTERVAL=30
METRICS_ENABLED=true
UPTIME_MONITORING=true

# Version Info
VERSION=2.0.0-production
BUILD_NUMBER=build_production
DEPLOYMENT_TIMESTAMP=1733263648

# User Configuration (Replace with actual values)
PRIVATE_KEY=your-wallet-private-key-here
WALLET_ADDRESS=your-wallet-address-here
HELIUS_API_KEY=your-helius-api-key-here
QUICKNODE_ENDPOINT=your-quicknode-endpoint-here
DISCORD_WEBHOOK_URL=your-discord-webhook-url-here
'''

# Write the .env file
with open('.env', 'w', encoding='utf-8') as f:
    f.write(env_content)

print("✅ Clean production .env file created successfully!")
print("📝 Edit .env file to add your actual API keys and wallet information.") 