from typing import Dict, Any

# Base configuration for Ant Princess
ANT_PRINCESS_CONFIG: Dict[str, Any] = {
    "initial_worker_count": 10,
    "ai_models": {
        "grok": {
            "purpose": "hype_analysis",
            "capabilities": ["tweet_analysis", "sentiment_analysis", "trend_detection"],
            "parameters": {
                "tweet_volume_weight": 0.4,
                "content_analysis_weight": 0.6,
                "cronbach_alpha_threshold": 0.7
            }
        },
        "local": {
            "purpose": "decision_making",
            "capabilities": ["market_analysis", "risk_assessment", "trade_execution"],
            "parameters": {
                "risk_tolerance": 0.5,
                "confidence_threshold": 0.8,
                "max_position_size": 0.1
            }
        }
    },
    "worker_ant_config": {
        "cost": "low",
        "capability": "basic",
        "max_count": 100,
        "creation_threshold": 0.8
    },
    "multiplication_thresholds": {
        "performance_score": 0.85,
        "profit_threshold": 1000,
        "experience_threshold": 100
    },
    "experience_sharing": {
        "update_frequency": 3600,  # seconds
        "min_experience_quality": 0.7,
        "max_pool_size": 1000
    }
}

# Queen configuration
QUEEN_CONFIG: Dict[str, Any] = {
    "optimization_frequency": 86400,  # seconds
    "performance_metrics": {
        "score_weights": {
            "profit": 0.4,
            "risk_management": 0.3,
            "execution_speed": 0.2,
            "adaptability": 0.1
        },
        "adaptation_threshold": 0.7
    },
    "colony_management": {
        "max_princesses": 10,
        "pruning_threshold": 0.5,
        "experience_retention_period": 2592000  # 30 days in seconds
    }
}

# System-wide constants
SYSTEM_CONSTANTS: Dict[str, Any] = {
    "max_concurrent_trades": 5,
    "min_liquidity_requirement": 10000,
    "default_timeout": 30,
    "retry_attempts": 3,
    "logging_level": "INFO"
} 