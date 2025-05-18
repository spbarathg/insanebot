from typing import Dict, Any

WALLET_TRACKER_CONFIG: Dict[str, Any] = {
    "wallet_file": "config/wallets.json",
    "solana_rpc_url": "https://api.mainnet-beta.solana.com",
    "update_intervals": {
        "high": 30,    # seconds
        "medium": 60,  # seconds
        "low": 300     # seconds
    },
    "batch_size": 20,  # Number of wallets to process in parallel
    "cache_ttl": 300,  # Cache TTL in seconds
    "activity_thresholds": {
        "high": 0.8,
        "medium": 0.5,
        "low": 0.0
    },
    "activity_factors": {
        "transaction_count": 0.3,
        "balance_change": 0.3,
        "token_holdings": 0.2,
        "program_interactions": 0.2
    },
    "rpc_limits": {
        "max_requests_per_second": 10,
        "max_concurrent_requests": 20,
        "retry_attempts": 3,
        "retry_delay": 1
    },
    "monitoring": {
        "log_level": "INFO",
        "metrics_interval": 60,  # seconds
        "alert_thresholds": {
            "error_rate": 0.1,
            "response_time": 2.0,  # seconds
            "cache_hit_ratio": 0.7
        }
    }
} 