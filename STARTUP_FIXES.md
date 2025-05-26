# Enhanced Ant Bot - Startup Issue Fixes

## Issues Identified

Your Enhanced Ant Bot was failing to start due to two main issues:

### 1. Missing Risk Configuration
**Error:** `AttributeError: 'Settings' object has no attribute 'RISK_LIMITS'`
- **Root Cause:** The `portfolio_manager.py` expected `RISK_LIMITS` and `POSITION_LIMITS` settings that weren't defined in the configuration
- **Status:** ✅ **FIXED** - Added missing risk limits to `src/utils/config.py`

### 2. Missing Wallet Credentials
**Error:** `Missing or invalid environment variables: ['WALLET_PASSWORD', 'WALLET_SALT']`
- **Root Cause:** Wallet manager requires encryption credentials that weren't in the environment template
- **Status:** ✅ **FIXED** - Updated `env.template` with required credentials

## Fixes Applied

### Configuration Updates

#### 1. Updated `src/utils/config.py`
Added missing risk management configurations:
```python
# Risk limits configuration
RISK_LIMITS = {
    "max_token_exposure": float(os.getenv("MAX_TOKEN_EXPOSURE", "0.2")),
    "max_portfolio_exposure": float(os.getenv("MAX_PORTFOLIO_EXPOSURE", "0.8")),
    "max_exposure": float(os.getenv("MAX_EXPOSURE", "0.7")),
    "daily_loss_limit": DAILY_LOSS_LIMIT,
    "max_drawdown": float(os.getenv("MAX_DRAWDOWN", "0.1")),
}

# Position limits configuration
POSITION_LIMITS = {
    "max_position_size": MAX_POSITION_SIZE,
    "min_position_size": MIN_POSITION_SIZE,
    "max_positions": int(os.getenv("MAX_POSITIONS", "10")),
    "position_concentration": float(os.getenv("POSITION_CONCENTRATION", "0.3")),
}
```

#### 2. Updated `env.template`
Added missing wallet encryption credentials:
```bash
# Wallet encryption credentials (REQUIRED for secure wallet management)
WALLET_PASSWORD=your_secure_password_here
WALLET_SALT=your_32_byte_hex_salt_here
```

### Helper Scripts Created

#### 1. `generate_wallet_credentials.py`
- Generates secure `WALLET_PASSWORD` and `WALLET_SALT` values
- Can automatically append to your `.env` file
- Usage: `python generate_wallet_credentials.py`

#### 2. `fix_bot_startup.py`
- Comprehensive fix script that addresses all startup issues
- Checks and updates `config.json` with missing risk settings
- Generates and adds wallet credentials to `.env` file
- Usage: `python fix_bot_startup.py`

## Quick Fix Steps

### Option 1: Automatic Fix (Recommended)
```bash
python fix_bot_startup.py
```

### Option 2: Manual Fix
1. **Generate wallet credentials:**
   ```bash
   python generate_wallet_credentials.py
   ```

2. **Create `.env` file from template:**
   ```bash
   cp env.template .env
   ```

3. **Update `.env` with the generated credentials**

4. **Restart the bot:**
   ```bash
   docker-compose up enhanced-ant-bot
   ```

## What Each Fix Does

### Risk Limits Configuration
- **max_token_exposure:** Maximum exposure per individual token (20% default)
- **max_portfolio_exposure:** Maximum total portfolio exposure (80% default)
- **max_exposure:** Overall risk exposure limit (70% default)
- **daily_loss_limit:** Maximum daily loss allowed
- **max_drawdown:** Maximum portfolio drawdown (10% default)

### Position Limits Configuration
- **max_position_size:** Maximum size for individual positions
- **min_position_size:** Minimum size for individual positions
- **max_positions:** Maximum number of concurrent positions (10 default)
- **position_concentration:** Maximum concentration in single position (30% default)

### Wallet Security
- **WALLET_PASSWORD:** Secure password for wallet encryption
- **WALLET_SALT:** 32-byte hex salt for cryptographic operations
- Both are required for secure wallet management and trading operations

## Next Steps

1. **Run the fix script:** `python fix_bot_startup.py`
2. **Update your `.env` file with real API keys:**
   - Add your actual QuickNode endpoint URL
   - Add your Helius API key
   - Add your wallet private key
3. **Test the bot:** `docker-compose up enhanced-ant-bot`

## Security Notes

- **Keep your wallet credentials secure** - losing the salt means losing access to encrypted wallets
- **Never share your private keys** or wallet credentials
- **Backup your `.env` file** in a secure location
- **Use environment variables in production** rather than storing sensitive data in files

## Troubleshooting

If you still encounter issues after applying these fixes:

1. Check that all required API keys are properly set in your `.env` file
2. Verify that your wallet has sufficient SOL balance
3. Ensure your network connection allows access to Solana RPC endpoints
4. Check the logs for any new error messages

The bot should now start successfully with these fixes applied! 