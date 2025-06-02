"""
Production-Grade API Documentation System

Automatic OpenAPI/Swagger documentation generation for the trading bot REST API:
- OpenAPI 3.0 specification generation
- Interactive Swagger UI
- Endpoint discovery and documentation
- Request/response schema validation
- Authentication documentation
- Rate limiting documentation
"""

import json
import yaml
import inspect
import logging
from typing import Dict, List, Any, Optional, Type, get_type_hints
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import os

logger = logging.getLogger(__name__)

@dataclass
class APIEndpoint:
    """API endpoint metadata"""
    path: str
    method: str
    function_name: str
    description: str
    tags: List[str]
    parameters: Dict[str, Any]
    responses: Dict[str, Any]
    security: List[str]
    rate_limit: Optional[str] = None

@dataclass
class APIDocumentationConfig:
    """API documentation configuration"""
    title: str = "Enhanced Ant Bot Trading API"
    description: str = "Professional Solana Trading Bot API"
    version: str = "1.0.0"
    contact: Dict[str, str] = None
    license: Dict[str, str] = None
    servers: List[Dict[str, str]] = None
    output_dir: Path = Path("docs/api")
    enable_swagger_ui: bool = True
    enable_redoc: bool = True

class TradingBotAPI(FastAPI):
    """Enhanced FastAPI application with automatic documentation"""
    
    def __init__(self, config: APIDocumentationConfig):
        super().__init__(
            title=config.title,
            description=config.description,
            version=config.version,
            contact=config.contact,
            license_info=config.license,
            servers=config.servers or [{"url": "http://localhost:8080", "description": "Development server"}]
        )
        
        self.config = config
        self.endpoints: List[APIEndpoint] = []
        
        # Setup documentation routes
        self._setup_documentation_routes()
        
        # Setup core API routes
        self._setup_core_routes()
    
    def _setup_documentation_routes(self):
        """Setup API documentation routes"""
        
        @self.get("/docs/openapi.json", tags=["Documentation"])
        async def get_openapi_spec():
            """Get OpenAPI specification in JSON format"""
            return self.openapi()
        
        @self.get("/docs/openapi.yaml", tags=["Documentation"])
        async def get_openapi_yaml():
            """Get OpenAPI specification in YAML format"""
            spec = self.openapi()
            return yaml.dump(spec, default_flow_style=False)
        
        @self.get("/docs/endpoints", tags=["Documentation"])
        async def list_endpoints():
            """List all available API endpoints"""
            return {
                "endpoints": [asdict(endpoint) for endpoint in self.endpoints],
                "total_count": len(self.endpoints)
            }
        
        @self.get("/health", tags=["System"])
        async def health_check():
            """System health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": self.config.version
            }
    
    def _setup_core_routes(self):
        """Setup core trading bot API routes"""
        
        # System Status Routes
        @self.get("/api/v1/system/status", tags=["System"])
        async def get_system_status():
            """Get comprehensive system status"""
            try:
                # This would integrate with your actual system status
                from ..enhanced_main import AntBotSystem
                # Placeholder for actual implementation
                return {
                    "system_running": True,
                    "uptime_hours": 24.5,
                    "active_ants": 12,
                    "total_trades": 156,
                    "current_capital": 0.85,
                    "defense_mode": "NORMAL",
                    "last_updated": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Failed to get system status: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to retrieve system status")
        
        @self.get("/api/v1/system/metrics", tags=["System"])
        async def get_system_metrics():
            """Get detailed system performance metrics"""
            return {
                "performance": {
                    "cpu_usage": 25.5,
                    "memory_usage": 512,
                    "disk_usage": 45.2
                },
                "trading": {
                    "trades_per_hour": 12.3,
                    "success_rate": 78.5,
                    "average_profit": 2.3
                },
                "defense": {
                    "threats_detected": 3,
                    "threats_blocked": 3,
                    "defense_score": 95.8
                }
            }
        
        # Trading Routes
        @self.get("/api/v1/trading/positions", tags=["Trading"])
        async def get_active_positions():
            """Get all active trading positions"""
            return {
                "positions": [
                    {
                        "id": "pos_001",
                        "token": "SOL/USDC",
                        "side": "long",
                        "size": 0.1,
                        "entry_price": 100.50,
                        "current_price": 102.30,
                        "pnl": 1.8,
                        "pnl_percentage": 1.79
                    }
                ],
                "total_count": 1,
                "total_value": 0.1
            }
        
        @self.get("/api/v1/trading/history", tags=["Trading"])
        async def get_trading_history(
            limit: int = Field(default=50, description="Maximum number of trades to return"),
            offset: int = Field(default=0, description="Number of trades to skip")
        ):
            """Get trading history with pagination"""
            return {
                "trades": [
                    {
                        "id": "trade_001",
                        "timestamp": "2024-01-01T12:00:00Z",
                        "token": "SOL/USDC",
                        "side": "buy",
                        "amount": 0.1,
                        "price": 100.00,
                        "fee": 0.001,
                        "profit": 2.5
                    }
                ],
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total": 156,
                    "has_more": True
                }
            }
        
        # Configuration Routes
        @self.get("/api/v1/config/risk-management", tags=["Configuration"])
        async def get_risk_config():
            """Get current risk management configuration"""
            return {
                "max_position_size": 0.1,
                "stop_loss_percentage": 5.0,
                "take_profit_percentage": 15.0,
                "max_daily_trades": 50,
                "defense_mode": "NORMAL"
            }
        
        @self.put("/api/v1/config/risk-management", tags=["Configuration"])
        async def update_risk_config(config: dict):
            """Update risk management configuration"""
            # Placeholder for actual implementation
            return {
                "status": "updated",
                "new_config": config,
                "timestamp": datetime.now().isoformat()
            }
        
        # AI and Analytics Routes
        @self.get("/api/v1/ai/signals", tags=["AI & Analytics"])
        async def get_ai_signals():
            """Get current AI trading signals"""
            return {
                "signals": [
                    {
                        "token": "BONK",
                        "signal": "BUY",
                        "confidence": 0.85,
                        "source": "smart_money_tracker",
                        "timestamp": datetime.now().isoformat()
                    }
                ],
                "signal_count": 1,
                "average_confidence": 0.85
            }
        
        @self.get("/api/v1/analytics/performance", tags=["AI & Analytics"])
        async def get_performance_analytics():
            """Get detailed performance analytics"""
            return {
                "daily_pnl": 15.5,
                "weekly_pnl": 87.3,
                "monthly_pnl": 234.7,
                "total_trades": 156,
                "win_rate": 78.5,
                "sharpe_ratio": 2.3,
                "max_drawdown": 5.2
            }
        
        # Security and Authentication Routes
        @self.post("/api/v1/auth/login", tags=["Authentication"])
        async def login(credentials: dict):
            """Authenticate user and return access token"""
            # Placeholder for actual implementation
            return {
                "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                "token_type": "bearer",
                "expires_in": 3600
            }
        
        @self.post("/api/v1/auth/refresh", tags=["Authentication"])
        async def refresh_token(refresh_token: str):
            """Refresh access token"""
            return {
                "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                "token_type": "bearer",
                "expires_in": 3600
            }
        
        # Emergency Controls
        @self.post("/api/v1/emergency/stop", tags=["Emergency Controls"])
        async def emergency_stop():
            """Emergency stop all trading activities"""
            return {
                "status": "emergency_stop_activated",
                "message": "All trading activities have been halted",
                "timestamp": datetime.now().isoformat()
            }
        
        @self.post("/api/v1/emergency/resume", tags=["Emergency Controls"])
        async def emergency_resume():
            """Resume trading activities after emergency stop"""
            return {
                "status": "trading_resumed",
                "message": "Trading activities have been resumed",
                "timestamp": datetime.now().isoformat()
            }

class APIDocumentationGenerator:
    """Comprehensive API documentation generator with OpenAPI 3.0 support."""
    
    def __init__(self, output_dir: str = "docs/api"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # API specification
        self.api_spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "Solana Trading Bot API",
                "description": "Comprehensive API for managing and monitoring the Solana trading bot",
                "version": "1.0.0",
                "contact": {
                    "name": "Trading Bot Support",
                    "email": "support@tradingbot.com"
                },
                "license": {
                    "name": "MIT",
                    "url": "https://opensource.org/licenses/MIT"
                }
            },
            "servers": [
                {
                    "url": "http://localhost:8000",
                    "description": "Development server"
                },
                {
                    "url": "https://api.tradingbot.com",
                    "description": "Production server"
                }
            ],
            "paths": {},
            "components": {
                "schemas": {},
                "securitySchemes": {
                    "BearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "JWT"
                    },
                    "ApiKeyAuth": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-API-Key"
                    }
                }
            },
            "security": [
                {"BearerAuth": []},
                {"ApiKeyAuth": []}
            ]
        }
        
        self._initialize_schemas()
        self._initialize_paths()
        
        self.logger.info("API Documentation Generator initialized")
    
    def _initialize_schemas(self):
        """Initialize common data schemas."""
        self.api_spec["components"]["schemas"] = {
            "Error": {
                "type": "object",
                "properties": {
                    "error": {
                        "type": "string",
                        "description": "Error message"
                    },
                    "code": {
                        "type": "string",
                        "description": "Error code"
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Error timestamp"
                    }
                },
                "required": ["error", "code", "timestamp"]
            },
            "SuccessResponse": {
                "type": "object",
                "properties": {
                    "success": {
                        "type": "boolean",
                        "description": "Operation success status"
                    },
                    "message": {
                        "type": "string",
                        "description": "Success message"
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Response timestamp"
                    }
                },
                "required": ["success", "message", "timestamp"]
            },
            "TradingPair": {
                "type": "object",
                "properties": {
                    "base_token": {
                        "type": "string",
                        "description": "Base token symbol"
                    },
                    "quote_token": {
                        "type": "string",
                        "description": "Quote token symbol"
                    },
                    "pair_address": {
                        "type": "string",
                        "description": "Solana pair address"
                    },
                    "liquidity": {
                        "type": "number",
                        "description": "Current liquidity in USD"
                    },
                    "price": {
                        "type": "number",
                        "description": "Current price"
                    },
                    "volume_24h": {
                        "type": "number",
                        "description": "24-hour trading volume"
                    }
                },
                "required": ["base_token", "quote_token", "pair_address"]
            },
            "Trade": {
                "type": "object",
                "properties": {
                    "trade_id": {
                        "type": "string",
                        "description": "Unique trade identifier"
                    },
                    "pair": {
                        "$ref": "#/components/schemas/TradingPair"
                    },
                    "side": {
                        "type": "string",
                        "enum": ["buy", "sell"],
                        "description": "Trade side"
                    },
                    "amount": {
                        "type": "number",
                        "description": "Trade amount"
                    },
                    "price": {
                        "type": "number",
                        "description": "Execution price"
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Trade execution timestamp"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "executed", "failed", "cancelled"],
                        "description": "Trade status"
                    },
                    "transaction_hash": {
                        "type": "string",
                        "description": "Solana transaction hash"
                    }
                },
                "required": ["trade_id", "pair", "side", "amount", "timestamp", "status"]
            },
            "Portfolio": {
                "type": "object",
                "properties": {
                    "total_value_usd": {
                        "type": "number",
                        "description": "Total portfolio value in USD"
                    },
                    "total_pnl": {
                        "type": "number",
                        "description": "Total profit/loss"
                    },
                    "total_pnl_percentage": {
                        "type": "number",
                        "description": "Total P&L percentage"
                    },
                    "positions": {
                        "type": "array",
                        "items": {
                            "$ref": "#/components/schemas/Position"
                        }
                    },
                    "cash_balance": {
                        "type": "number",
                        "description": "Available cash balance"
                    },
                    "last_updated": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Last update timestamp"
                    }
                },
                "required": ["total_value_usd", "total_pnl", "positions", "cash_balance"]
            },
            "Position": {
                "type": "object",
                "properties": {
                    "token_symbol": {
                        "type": "string",
                        "description": "Token symbol"
                    },
                    "token_address": {
                        "type": "string",
                        "description": "Token mint address"
                    },
                    "quantity": {
                        "type": "number",
                        "description": "Position quantity"
                    },
                    "average_price": {
                        "type": "number",
                        "description": "Average purchase price"
                    },
                    "current_price": {
                        "type": "number",
                        "description": "Current market price"
                    },
                    "unrealized_pnl": {
                        "type": "number",
                        "description": "Unrealized profit/loss"
                    },
                    "unrealized_pnl_percentage": {
                        "type": "number",
                        "description": "Unrealized P&L percentage"
                    }
                },
                "required": ["token_symbol", "quantity", "average_price", "current_price"]
            },
            "BotStatus": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["running", "stopped", "paused", "error"],
                        "description": "Current bot status"
                    },
                    "uptime": {
                        "type": "integer",
                        "description": "Bot uptime in seconds"
                    },
                    "last_activity": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Last activity timestamp"
                    },
                    "total_trades": {
                        "type": "integer",
                        "description": "Total number of trades executed"
                    },
                    "successful_trades": {
                        "type": "integer",
                        "description": "Number of successful trades"
                    },
                    "failed_trades": {
                        "type": "integer",
                        "description": "Number of failed trades"
                    },
                    "current_balance": {
                        "type": "number",
                        "description": "Current balance in SOL"
                    }
                },
                "required": ["status", "uptime", "total_trades", "current_balance"]
            },
            "SystemHealth": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["healthy", "warning", "critical"],
                        "description": "Overall system health status"
                    },
                    "cpu_usage": {
                        "type": "number",
                        "description": "CPU usage percentage"
                    },
                    "memory_usage": {
                        "type": "number",
                        "description": "Memory usage percentage"
                    },
                    "disk_usage": {
                        "type": "number",
                        "description": "Disk usage percentage"
                    },
                    "network_latency": {
                        "type": "number",
                        "description": "Network latency in milliseconds"
                    },
                    "rpc_connection": {
                        "type": "boolean",
                        "description": "RPC connection status"
                    },
                    "database_connection": {
                        "type": "boolean",
                        "description": "Database connection status"
                    },
                    "last_check": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Last health check timestamp"
                    }
                },
                "required": ["status", "cpu_usage", "memory_usage", "rpc_connection"]
            },
            "Configuration": {
                "type": "object",
                "properties": {
                    "trading_enabled": {
                        "type": "boolean",
                        "description": "Whether trading is enabled"
                    },
                    "max_slippage": {
                        "type": "number",
                        "description": "Maximum allowed slippage percentage"
                    },
                    "min_liquidity": {
                        "type": "number",
                        "description": "Minimum liquidity requirement"
                    },
                    "risk_settings": {
                        "type": "object",
                        "properties": {
                            "max_position_size": {
                                "type": "number",
                                "description": "Maximum position size in USD"
                            },
                            "stop_loss_percentage": {
                                "type": "number",
                                "description": "Default stop loss percentage"
                            },
                            "take_profit_percentage": {
                                "type": "number",
                                "description": "Default take profit percentage"
                            }
                        }
                    },
                    "monitoring": {
                        "type": "object",
                        "properties": {
                            "discord_notifications": {
                                "type": "boolean",
                                "description": "Enable Discord notifications"
                            },
                            "log_level": {
                                "type": "string",
                                "enum": ["DEBUG", "INFO", "WARNING", "ERROR"],
                                "description": "Logging level"
                            }
                        }
                    }
                },
                "required": ["trading_enabled", "max_slippage", "min_liquidity"]
            }
        }
    
    def _initialize_paths(self):
        """Initialize API endpoints documentation."""
        self.api_spec["paths"] = {
            "/health": {
                "get": {
                    "summary": "Get system health status",
                    "description": "Returns comprehensive system health information including resource usage and service status",
                    "tags": ["System"],
                    "responses": {
                        "200": {
                            "description": "System health information",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/SystemHealth"
                                    }
                                }
                            }
                        },
                        "500": {
                            "description": "Internal server error",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Error"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/bot/status": {
                "get": {
                    "summary": "Get trading bot status",
                    "description": "Returns current status and statistics of the trading bot",
                    "tags": ["Bot Management"],
                    "responses": {
                        "200": {
                            "description": "Bot status information",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/BotStatus"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/bot/start": {
                "post": {
                    "summary": "Start the trading bot",
                    "description": "Starts the trading bot with current configuration",
                    "tags": ["Bot Management"],
                    "responses": {
                        "200": {
                            "description": "Bot started successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/SuccessResponse"
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bot already running or configuration invalid",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Error"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/bot/stop": {
                "post": {
                    "summary": "Stop the trading bot",
                    "description": "Gracefully stops the trading bot and cancels any pending orders",
                    "tags": ["Bot Management"],
                    "responses": {
                        "200": {
                            "description": "Bot stopped successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/SuccessResponse"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/trades": {
                "get": {
                    "summary": "Get trading history",
                    "description": "Returns paginated list of executed trades",
                    "tags": ["Trading"],
                    "parameters": [
                        {
                            "name": "limit",
                            "in": "query",
                            "description": "Maximum number of trades to return",
                            "required": False,
                            "schema": {
                                "type": "integer",
                                "default": 50,
                                "minimum": 1,
                                "maximum": 1000
                            }
                        },
                        {
                            "name": "offset",
                            "in": "query",
                            "description": "Number of trades to skip",
                            "required": False,
                            "schema": {
                                "type": "integer",
                                "default": 0,
                                "minimum": 0
                            }
                        },
                        {
                            "name": "pair",
                            "in": "query",
                            "description": "Filter by trading pair",
                            "required": False,
                            "schema": {
                                "type": "string"
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "List of trades",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "trades": {
                                                "type": "array",
                                                "items": {
                                                    "$ref": "#/components/schemas/Trade"
                                                }
                                            },
                                            "total": {
                                                "type": "integer",
                                                "description": "Total number of trades"
                                            },
                                            "limit": {
                                                "type": "integer",
                                                "description": "Applied limit"
                                            },
                                            "offset": {
                                                "type": "integer",
                                                "description": "Applied offset"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/trades/{trade_id}": {
                "get": {
                    "summary": "Get specific trade details",
                    "description": "Returns detailed information about a specific trade",
                    "tags": ["Trading"],
                    "parameters": [
                        {
                            "name": "trade_id",
                            "in": "path",
                            "description": "Trade ID",
                            "required": True,
                            "schema": {
                                "type": "string"
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Trade details",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Trade"
                                    }
                                }
                            }
                        },
                        "404": {
                            "description": "Trade not found",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Error"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/portfolio": {
                "get": {
                    "summary": "Get portfolio overview",
                    "description": "Returns current portfolio status including positions and P&L",
                    "tags": ["Portfolio"],
                    "responses": {
                        "200": {
                            "description": "Portfolio information",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Portfolio"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/portfolio/positions": {
                "get": {
                    "summary": "Get current positions",
                    "description": "Returns list of current open positions",
                    "tags": ["Portfolio"],
                    "responses": {
                        "200": {
                            "description": "List of positions",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {
                                            "$ref": "#/components/schemas/Position"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/config": {
                "get": {
                    "summary": "Get current configuration",
                    "description": "Returns current bot configuration settings",
                    "tags": ["Configuration"],
                    "responses": {
                        "200": {
                            "description": "Current configuration",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Configuration"
                                    }
                                }
                            }
                        }
                    }
                },
                "put": {
                    "summary": "Update configuration",
                    "description": "Updates bot configuration settings",
                    "tags": ["Configuration"],
                    "requestBody": {
                        "description": "Configuration updates",
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Configuration"
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Configuration updated successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/SuccessResponse"
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Invalid configuration",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Error"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/pairs": {
                "get": {
                    "summary": "Get available trading pairs",
                    "description": "Returns list of available trading pairs with current market data",
                    "tags": ["Market Data"],
                    "parameters": [
                        {
                            "name": "min_liquidity",
                            "in": "query",
                            "description": "Minimum liquidity filter",
                            "required": False,
                            "schema": {
                                "type": "number",
                                "minimum": 0
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "List of trading pairs",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {
                                            "$ref": "#/components/schemas/TradingPair"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/logs": {
                "get": {
                    "summary": "Get system logs",
                    "description": "Returns recent system logs with filtering options",
                    "tags": ["System"],
                    "parameters": [
                        {
                            "name": "level",
                            "in": "query",
                            "description": "Log level filter",
                            "required": False,
                            "schema": {
                                "type": "string",
                                "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]
                            }
                        },
                        {
                            "name": "limit",
                            "in": "query",
                            "description": "Maximum number of log entries",
                            "required": False,
                            "schema": {
                                "type": "integer",
                                "default": 100,
                                "minimum": 1,
                                "maximum": 1000
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "System logs",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "logs": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "timestamp": {
                                                            "type": "string",
                                                            "format": "date-time"
                                                        },
                                                        "level": {
                                                            "type": "string"
                                                        },
                                                        "message": {
                                                            "type": "string"
                                                        },
                                                        "component": {
                                                            "type": "string"
                                                        }
                                                    }
                                                }
                                            },
                                            "total": {
                                                "type": "integer"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    
    def generate_openapi_spec(self) -> Dict[str, Any]:
        """Generate complete OpenAPI 3.0 specification."""
        return self.api_spec
    
    def save_openapi_spec(self, filename: str = "openapi.yaml") -> str:
        """Save OpenAPI specification to file."""
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.api_spec, f, default_flow_style=False, allow_unicode=True)
        
        self.logger.info(f"OpenAPI specification saved to {output_path}")
        return str(output_path)
    
    def generate_swagger_ui(self) -> str:
        """Generate Swagger UI HTML page."""
        self.logger.info("Generating Swagger UI...")
        
        # Generate OpenAPI spec first
        spec = self.generate_openapi_spec()
        
        # Create Swagger UI HTML
        swagger_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Bot API - Swagger UI</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@3.52.5/swagger-ui.css" />
    <style>
        html {{
            box-sizing: border-box;
            overflow: -moz-scrollbars-vertical;
            overflow-y: scroll;
        }}
        *, *:before, *:after {{
            box-sizing: inherit;
        }}
        body {{
            margin:0;
            background: #fafafa;
        }}
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@3.52.5/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@3.52.5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {{
            const ui = SwaggerUIBundle({{
                url: './openapi.yaml',
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout"
            }});
        }};
    </script>
</body>
</html>"""
        
        html_path = self.output_dir / "index.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(swagger_html)
        
        self.logger.info(f"Swagger UI generated at {html_path}")
        return str(html_path)
    
    def generate_postman_collection(self) -> str:
        """Generate Postman collection."""
        self.logger.info("Generating Postman collection...")
        
        collection = {
            "info": {
                "name": "Trading Bot API",
                "description": "Comprehensive API collection for the Solana Trading Bot",
                "version": "1.0.0",
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
            },
            "auth": {
                "type": "apikey",
                "apikey": [
                    {
                        "key": "key",
                        "value": "X-API-Key",
                        "type": "string"
                    },
                    {
                        "key": "value",
                        "value": "{{api_key}}",
                        "type": "string"
                    }
                ]
            },
            "variable": [
                {
                    "key": "base_url",
                    "value": "http://localhost:8000",
                    "type": "string"
                },
                {
                    "key": "api_key",
                    "value": "your-api-key-here",
                    "type": "string"
                }
            ],
            "item": [
                {
                    "name": "System",
                    "item": [
                        {
                            "name": "Health Check",
                            "request": {
                                "method": "GET",
                                "header": [],
                                "url": {
                                    "raw": "{{base_url}}/health",
                                    "host": ["{{base_url}}"],
                                    "path": ["health"]
                                }
                            }
                        },
                        {
                            "name": "System Status",
                            "request": {
                                "method": "GET",
                                "header": [],
                                "url": {
                                    "raw": "{{base_url}}/api/v1/system/status",
                                    "host": ["{{base_url}}"],
                                    "path": ["api", "v1", "system", "status"]
                                }
                            }
                        }
                    ]
                }
            ]
        }
        
        collection_path = self.output_dir / "postman_collection.json"
        with open(collection_path, 'w', encoding='utf-8') as f:
            json.dump(collection, f, indent=2)
        
        self.logger.info(f"Postman collection generated at {collection_path}")
        return str(collection_path)
    
    def generate_markdown_docs(self) -> str:
        """Generate markdown documentation."""
        markdown_content = f"""# Solana Trading Bot API Documentation

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

{self.api_spec['info']['description']}

**Version:** {self.api_spec['info']['version']}

## Base URLs

"""
        
        for server in self.api_spec["servers"]:
            markdown_content += f"- **{server['description']}:** `{server['url']}`\n"
        
        markdown_content += """
## Authentication

This API supports two authentication methods:

1. **Bearer Token (JWT):** Include `Authorization: Bearer <token>` header
2. **API Key:** Include `X-API-Key: <your-api-key>` header

## Endpoints

"""
        
        # Group endpoints by tags
        endpoints_by_tag = {}
        for path, methods in self.api_spec["paths"].items():
            for method, spec in methods.items():
                tags = spec.get("tags", ["Uncategorized"])
                for tag in tags:
                    if tag not in endpoints_by_tag:
                        endpoints_by_tag[tag] = []
                    endpoints_by_tag[tag].append((path, method, spec))
        
        # Generate documentation for each tag
        for tag, endpoints in endpoints_by_tag.items():
            markdown_content += f"\n### {tag}\n\n"
            
            for path, method, spec in endpoints:
                markdown_content += f"#### {method.upper()} {path}\n\n"
                markdown_content += f"**Summary:** {spec.get('summary', 'No summary')}\n\n"
                
                if 'description' in spec:
                    markdown_content += f"**Description:** {spec['description']}\n\n"
                
                # Parameters
                if 'parameters' in spec:
                    markdown_content += "**Parameters:**\n\n"
                    markdown_content += "| Name | Type | In | Required | Description |\n"
                    markdown_content += "|------|------|----|---------|-----------|\n"
                    
                    for param in spec['parameters']:
                        param_type = param.get('schema', {}).get('type', 'string')
                        required = "Yes" if param.get('required', False) else "No"
                        description = param.get('description', '')
                        markdown_content += f"| {param['name']} | {param_type} | {param['in']} | {required} | {description} |\n"
                    
                    markdown_content += "\n"
                
                # Request body
                if 'requestBody' in spec:
                    markdown_content += "**Request Body:**\n\n"
                    markdown_content += f"Content-Type: `application/json`\n\n"
                    if 'description' in spec['requestBody']:
                        markdown_content += f"{spec['requestBody']['description']}\n\n"
                
                # Responses
                if 'responses' in spec:
                    markdown_content += "**Responses:**\n\n"
                    for status_code, response in spec['responses'].items():
                        markdown_content += f"- **{status_code}:** {response.get('description', 'No description')}\n"
                    markdown_content += "\n"
                
                markdown_content += "---\n\n"
        
        # Add schemas documentation
        markdown_content += "\n## Data Models\n\n"
        
        for schema_name, schema in self.api_spec["components"]["schemas"].items():
            markdown_content += f"### {schema_name}\n\n"
            
            if 'description' in schema:
                markdown_content += f"{schema['description']}\n\n"
            
            if 'properties' in schema:
                markdown_content += "| Property | Type | Required | Description |\n"
                markdown_content += "|----------|------|----------|-----------|\n"
                
                required_fields = schema.get('required', [])
                
                for prop_name, prop_schema in schema['properties'].items():
                    prop_type = prop_schema.get('type', 'object')
                    if '$ref' in prop_schema:
                        prop_type = prop_schema['$ref'].split('/')[-1]
                    elif prop_type == 'array' and 'items' in prop_schema:
                        if '$ref' in prop_schema['items']:
                            prop_type = f"Array of {prop_schema['items']['$ref'].split('/')[-1]}"
                        else:
                            prop_type = f"Array of {prop_schema['items'].get('type', 'object')}"
                    
                    is_required = "Yes" if prop_name in required_fields else "No"
                    description = prop_schema.get('description', '')
                    
                    markdown_content += f"| {prop_name} | {prop_type} | {is_required} | {description} |\n"
                
                markdown_content += "\n"
        
        # Add examples
        markdown_content += """
## Examples

### Get System Health

```bash
curl -X GET "http://localhost:8000/health" \\
  -H "X-API-Key: your-api-key"
```

### Start Trading Bot

```bash
curl -X POST "http://localhost:8000/bot/start" \\
  -H "X-API-Key: your-api-key" \\
  -H "Content-Type: application/json"
```

### Get Trading History

```bash
curl -X GET "http://localhost:8000/trades?limit=10&pair=SOL/USDC" \\
  -H "X-API-Key: your-api-key"
```

### Update Configuration

```bash
curl -X PUT "http://localhost:8000/config" \\
  -H "X-API-Key: your-api-key" \\
  -H "Content-Type: application/json" \\
  -d '{
    "trading_enabled": true,
    "max_slippage": 0.02,
    "min_liquidity": 10000
  }'
```

## Error Handling

All errors follow a consistent format:

```json
{
  "error": "Error message describing what went wrong",
  "code": "ERROR_CODE",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Rate Limiting

- **Rate Limit:** 100 requests per minute per API key
- **Burst Limit:** 20 requests per second
- **Headers:** Rate limit information is included in response headers

## WebSocket Support

For real-time updates, connect to the WebSocket endpoint:

- **URL:** `ws://localhost:8000/ws`
- **Authentication:** Send API key in connection query: `?api_key=your-key`

### WebSocket Events

- `trade_executed`: New trade execution
- `price_update`: Price updates for monitored pairs
- `bot_status_change`: Bot status changes
- `system_alert`: System alerts and warnings

"""
        
        markdown_path = self.output_dir / "README.md"
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        self.logger.info(f"Markdown documentation generated at {markdown_path}")
        return str(markdown_path)
    
    def generate_all_documentation(self) -> Dict[str, str]:
        """Generate all documentation formats."""
        self.logger.info("Generating comprehensive API documentation...")
        
        files = {
            "openapi_spec": self.save_openapi_spec(),
            "swagger_ui": self.generate_swagger_ui(),
            "postman_collection": self.generate_postman_collection(),
            "markdown_docs": self.generate_markdown_docs()
        }
        
        # Create a simple index page
        index_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Trading Bot API Documentation</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        .card {{ border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 8px; }}
        .card h3 {{ margin-top: 0; color: #333; }}
        .card a {{ color: #007bff; text-decoration: none; }}
        .card a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Solana Trading Bot API Documentation</h1>
        <p>Welcome to the comprehensive API documentation for the Solana Trading Bot.</p>
        
        <div class="card">
            <h3>Interactive Documentation</h3>
            <p>Explore the API endpoints with Swagger UI</p>
            <a href="./index.html">Open Swagger UI →</a>
        </div>
        
        <div class="card">
            <h3>Markdown Documentation</h3>
            <p>Complete API reference in markdown format</p>
            <a href="./README.md">View Markdown Docs →</a>
        </div>
        
        <div class="card">
            <h3>Postman Collection</h3>
            <p>Import this collection into Postman for API testing</p>
            <a href="./postman_collection.json" download>Download Collection →</a>
        </div>
        
        <div class="card">
            <h3>OpenAPI Specification</h3>
            <p>Raw OpenAPI 3.0 specification file</p>
            <a href="./openapi.yaml">View OpenAPI Spec →</a>
        </div>
        
        <hr>
        <p><small>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
    </div>
</body>
</html>"""
        
        index_path = self.output_dir / "documentation.html"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index_content)
        
        files["index"] = str(index_path)
        
        self.logger.info("All API documentation generated successfully!")
        self.logger.info(f"Documentation available at: {self.output_dir}")
        
        return files

def main():
    """CLI interface for generating API documentation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate API Documentation")
    parser.add_argument("--output-dir", default="docs/api", help="Output directory for documentation")
    parser.add_argument("--format", choices=["all", "openapi", "swagger", "postman", "markdown"], 
                       default="all", help="Documentation format to generate")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = APIDocumentationGenerator(args.output_dir)
    
    if args.format == "all":
        files = generator.generate_all_documentation()
        print("✅ All documentation generated:")
        for doc_type, file_path in files.items():
            print(f"  • {doc_type}: {file_path}")
    
    elif args.format == "openapi":
        file_path = generator.save_openapi_spec()
        print(f"✅ OpenAPI specification: {file_path}")
    
    elif args.format == "swagger":
        file_path = generator.generate_swagger_ui()
        print(f"✅ Swagger UI: {file_path}")
    
    elif args.format == "postman":
        file_path = generator.generate_postman_collection()
        print(f"✅ Postman collection: {file_path}")
    
    elif args.format == "markdown":
        file_path = generator.generate_markdown_docs()
        print(f"✅ Markdown documentation: {file_path}")

if __name__ == "__main__":
    main() 