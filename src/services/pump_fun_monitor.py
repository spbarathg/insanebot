"""
Pump.fun Real-Time Monitor

This service monitors pump.fun for new token launches and generates
immediate trading signals for maximum profit opportunities.

Features:
- Real-time WebSocket monitoring of pump.fun
- New token launch detection within seconds
- Developer activity analysis
- Initial liquidity assessment
- Immediate buy signal generation
- Viral potential scoring
"""

import asyncio
import websockets
import aiohttp
import json
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)

@dataclass
class PumpFunToken:
    """New token detected on pump.fun"""
    address: str
    symbol: str
    name: str
    description: str
    image_url: str
    creator: str
    created_at: float
    initial_liquidity: float
    market_cap: float
    price_usd: float
    volume_24h: float
    holder_count: int
    buy_count: int
    sell_count: int
    website: str
    twitter: str
    telegram: str
    
    # Analysis fields
    viral_score: float = 0.0
    risk_score: float = 0.0
    signal_strength: float = 0.0
    recommendation: str = "HOLD"
    
    @property
    def age_minutes(self) -> float:
        """Age of token in minutes"""
        return (time.time() - self.created_at) / 60
    
    @property
    def is_new(self) -> bool:
        """Is token newer than 10 minutes"""
        return self.age_minutes < 10

@dataclass 
class PumpFunSignal:
    """Trading signal generated from pump.fun analysis"""
    token_address: str
    token_symbol: str
    signal_type: str  # 'BUY', 'SELL', 'WATCH'
    confidence: float  # 0.0 to 1.0
    urgency: str  # 'low', 'medium', 'high', 'critical'
    entry_price: float
    target_price: float
    stop_loss: float
    position_size: float
    reasoning: str
    viral_indicators: List[str]
    risk_factors: List[str]
    timestamp: float
    expires_at: float

class PumpFunMonitor:
    """
    Real-time pump.fun monitoring system for new token launches
    
    This system provides the fastest possible detection of new memecoins
    with viral potential for maximum profit opportunities.
    """
    
    def __init__(self, callback_handler: Optional[Callable] = None):
        self.callback_handler = callback_handler
        
        # WebSocket connections
        self.ws_connection = None
        self.monitoring_active = False
        
        # Configuration
        self.config = {
            'min_liquidity': 1000,      # Min $1K liquidity to trade
            'max_token_age_minutes': 15, # Only trade tokens < 15 min old
            'min_viral_score': 0.6,     # Min viral score to generate signal
            'max_risk_score': 0.7,      # Max risk score to trade
            'position_size_pct': 0.02,  # 2% of portfolio per trade
            'stop_loss_pct': 0.15,      # 15% stop loss
            'target_profit_pct': 3.0,   # 300% target profit
            'monitoring_interval': 1    # Check every 1 second
        }
        
        # State tracking
        self.detected_tokens: Dict[str, PumpFunToken] = {}
        self.generated_signals: Dict[str, PumpFunSignal] = {}
        self.performance_metrics = {
            'tokens_detected': 0,
            'signals_generated': 0,
            'successful_trades': 0,
            'total_profit': 0.0,
            'win_rate': 0.0
        }
        
        # Viral indicators
        self.viral_keywords = [
            'moon', 'rocket', '🚀', '💎', 'diamond', 'hands', 'hodl',
            'pump', 'ape', 'degen', 'based', 'chad', 'gigachad',
            'pepe', 'wojak', 'apu', 'doge', 'shiba', 'inu',
            'meme', 'funny', 'lol', 'kek', 'stonks', 'tendies'
        ]
        
        # Risk keywords  
        self.risk_keywords = [
            'scam', 'rug', 'honeypot', 'fake', 'copy', 'duplicate',
            'exit', 'dump', 'sell', 'crash', 'dead', 'rip'
        ]
        
        # Social media patterns
        self.social_patterns = {
            'twitter': r'(?:twitter\.com|x\.com)/[a-zA-Z0-9_]+',
            'telegram': r't\.me/[a-zA-Z0-9_]+',
            'website': r'https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        }
        
        logger.info("🎯 Pump.fun Monitor initialized - Ready for token launches!")
    
    async def start_monitoring(self) -> bool:
        """Start real-time monitoring of pump.fun"""
        try:
            logger.info("🚀 Starting pump.fun real-time monitoring...")
            self.monitoring_active = True
            
            # Start monitoring tasks
            tasks = [
                asyncio.create_task(self._monitor_websocket()),
                asyncio.create_task(self._monitor_api()),
                asyncio.create_task(self._process_signals())
            ]
            
            # Wait for tasks to start
            await asyncio.sleep(1)
            
            logger.info("✅ Pump.fun monitoring started successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start pump.fun monitoring: {e}")
            return False
    
    async def _monitor_websocket(self):
        """Monitor pump.fun via WebSocket (primary method)"""
        while self.monitoring_active:
            try:
                async with aiohttp.ClientSession() as session:
                    try:
                        # Connect to pump.fun WebSocket
                        ws_url = "wss://pumpportal.fun/api/data"
                        async with session.ws_connect(ws_url) as ws:
                            logger.info("🔌 Connected to pump.fun WebSocket")
                            
                            # Subscribe to new tokens
                            subscribe_msg = {
                                "method": "subscribe",
                                "keys": ["pump"]
                            }
                            await ws.send_str(json.dumps(subscribe_msg))
                            
                            # Listen for messages
                            async for msg in ws:
                                if not self.monitoring_active:
                                    break
                                    
                                if msg.type == aiohttp.WSMsgType.TEXT:
                                    try:
                                        data = json.loads(msg.data)
                                        await self._process_websocket_message(data)
                                    except Exception as e:
                                        logger.error(f"Error processing WebSocket message: {e}")
                                elif msg.type == aiohttp.WSMsgType.ERROR:
                                    logger.warning(f"WebSocket error: {ws.exception()}")
                                    break
                                        
                    except Exception as e:
                        if self.monitoring_active:  # Only log if we're still supposed to be running
                            logger.warning(f"WebSocket connection failed, using API fallback: {e}")
                            # Fallback to API monitoring if websocket fails
                            await self._monitor_api()
                        break
                    
            except asyncio.CancelledError:
                logger.info("WebSocket monitoring cancelled")
                break
            except Exception as e:
                if self.monitoring_active:
                    logger.error(f"WebSocket monitoring error: {e}")
                    try:
                        await asyncio.sleep(5)  # Wait before retry
                    except asyncio.CancelledError:
                        break
                else:
                    break
    
    async def _monitor_api(self):
        """Monitor pump.fun via API polling (fallback method)"""
        while self.monitoring_active:
            try:
                # Get new tokens from pump.fun API
                new_tokens = await self._fetch_new_tokens()

                for token_data in new_tokens:
                    if not self.monitoring_active:
                        break
                    await self._process_new_token(token_data)

                # Wait before next poll
                try:
                    await asyncio.sleep(self.config['monitoring_interval'])
                except asyncio.CancelledError:
                    break

            except asyncio.CancelledError:
                logger.info("API monitoring cancelled")
                break
            except Exception as e:
                if self.monitoring_active:
                    logger.error(f"API monitoring error: {e}")
                    try:
                        await asyncio.sleep(5)
                    except asyncio.CancelledError:
                        break
                else:
                    break
    
    async def _fetch_new_tokens(self) -> List[Dict]:
        """Fetch new tokens from pump.fun API"""
        try:
            async with aiohttp.ClientSession() as session:
                # Pump.fun API endpoint for new tokens
                url = "https://pump.fun/api/tokens/new"
                params = {
                    'limit': 50,
                    'offset': 0,
                    'sort': 'created_at',
                    'order': 'desc'
                }
                
                async with session.get(url, params=params, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('tokens', [])
                    else:
                        logger.warning(f"API request failed: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error fetching new tokens: {e}")
            return []
    
    async def _process_new_token(self, token_data: Dict):
        """Process a newly detected token"""
        try:
            token_address = token_data.get('address', '')
            
            if not token_address or token_address in self.detected_tokens:
                return  # Skip if no address or already processed
            
            # Create token object
            token = PumpFunToken(
                address=token_address,
                symbol=token_data.get('symbol', ''),
                name=token_data.get('name', ''),
                description=token_data.get('description', ''),
                image_url=token_data.get('image_url', ''),
                creator=token_data.get('creator', ''),
                created_at=token_data.get('created_at', time.time()),
                initial_liquidity=token_data.get('liquidity', 0),
                market_cap=token_data.get('market_cap', 0),
                price_usd=token_data.get('price_usd', 0),
                volume_24h=token_data.get('volume_24h', 0),
                holder_count=token_data.get('holder_count', 0),
                buy_count=token_data.get('buy_count', 0),
                sell_count=token_data.get('sell_count', 0),
                website=token_data.get('website', ''),
                twitter=token_data.get('twitter', ''),
                telegram=token_data.get('telegram', '')
            )
            
            # Analyze token potential
            await self._analyze_token(token)
            
            # Store token
            self.detected_tokens[token_address] = token
            self.performance_metrics['tokens_detected'] += 1
            
            logger.info(f"🔍 New token detected: {token.symbol} ({token.address[:8]}...) "
                       f"Viral: {token.viral_score:.2f} Risk: {token.risk_score:.2f}")
            
            # Generate signal if criteria met
            if await self._should_generate_signal(token):
                signal = await self._generate_trading_signal(token)
                if signal:
                    self.generated_signals[token_address] = signal
                    self.performance_metrics['signals_generated'] += 1
                    
                    # Call callback handler
                    if self.callback_handler:
                        await self.callback_handler(signal)
                    
                    logger.info(f"🎯 SIGNAL GENERATED: {signal.signal_type} {token.symbol} "
                               f"Confidence: {signal.confidence:.2f} Urgency: {signal.urgency}")
                    
        except Exception as e:
            logger.error(f"Error processing new token: {e}")
    
    async def _analyze_token(self, token: PumpFunToken):
        """Analyze token for viral potential and risk"""
        try:
            # Calculate viral score
            token.viral_score = await self._calculate_viral_score(token)
            
            # Calculate risk score  
            token.risk_score = await self._calculate_risk_score(token)
            
            # Overall signal strength
            token.signal_strength = token.viral_score * (1 - token.risk_score)
            
            # Generate recommendation
            if token.signal_strength > 0.7 and token.risk_score < 0.3:
                token.recommendation = "STRONG_BUY"
            elif token.signal_strength > 0.5 and token.risk_score < 0.5:
                token.recommendation = "BUY"  
            elif token.signal_strength > 0.3:
                token.recommendation = "WATCH"
            else:
                token.recommendation = "AVOID"
                
        except Exception as e:
            logger.error(f"Error analyzing token {token.address}: {e}")
    
    async def _calculate_viral_score(self, token: PumpFunToken) -> float:
        """Calculate viral potential score (0-1)"""
        try:
            score = 0.0
            
            # Name/symbol viral indicators
            text_content = f"{token.name} {token.symbol} {token.description}".lower()
            
            viral_matches = sum(1 for keyword in self.viral_keywords if keyword in text_content)
            viral_density = viral_matches / max(len(text_content.split()), 1)
            score += min(0.3, viral_density * 10)  # Max 0.3 from keywords
            
            # Social presence
            if token.twitter:
                score += 0.2
            if token.telegram:
                score += 0.2  
            if token.website:
                score += 0.1
            
            # Activity indicators
            if token.holder_count > 50:
                score += 0.1
            if token.buy_count > token.sell_count:
                score += 0.1
            if token.volume_24h > token.market_cap * 0.1:  # High turnover
                score += 0.1
                
            # Early stage bonus
            if token.age_minutes < 5:
                score += 0.1  # Early bird bonus
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating viral score: {e}")
            return 0.0
    
    async def _calculate_risk_score(self, token: PumpFunToken) -> float:
        """Calculate risk score (0-1, higher is riskier)"""
        try:
            risk = 0.0
            
            # Risk keywords in description
            text_content = f"{token.name} {token.symbol} {token.description}".lower()
            risk_matches = sum(1 for keyword in self.risk_keywords if keyword in text_content)
            risk += min(0.4, risk_matches * 0.2)
            
            # Low liquidity risk
            if token.initial_liquidity < self.config['min_liquidity']:
                risk += 0.3
            
            # Creator risk factors
            if not token.website and not token.twitter and not token.telegram:
                risk += 0.2  # No social presence
            
            # Market structure risks
            if token.holder_count < 10:
                risk += 0.2  # Too few holders
            
            if token.sell_count > token.buy_count * 2:
                risk += 0.3  # More sells than buys
            
            # Age risk (too old might miss pump)
            if token.age_minutes > self.config['max_token_age_minutes']:
                risk += 0.4
            
            return min(1.0, risk)
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 1.0  # Assume high risk on error
    
    async def _should_generate_signal(self, token: PumpFunToken) -> bool:
        """Determine if we should generate a trading signal"""
        try:
            # Basic filters
            if token.viral_score < self.config['min_viral_score']:
                return False
            
            if token.risk_score > self.config['max_risk_score']:
                return False
                
            if token.initial_liquidity < self.config['min_liquidity']:
                return False
            
            if token.age_minutes > self.config['max_token_age_minutes']:
                return False
                
            # Must have some social presence
            if not (token.twitter or token.telegram or token.website):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in signal criteria check: {e}")
            return False
    
    async def _generate_trading_signal(self, token: PumpFunToken) -> Optional[PumpFunSignal]:
        """Generate a trading signal for the token"""
        try:
            # Determine signal type based on analysis
            if token.signal_strength > 0.8:
                signal_type = "BUY"
                urgency = "critical"
                confidence = min(0.95, token.signal_strength)
            elif token.signal_strength > 0.6:
                signal_type = "BUY"
                urgency = "high"
                confidence = token.signal_strength
            elif token.signal_strength > 0.4:
                signal_type = "WATCH"
                urgency = "medium"
                confidence = token.signal_strength
            else:
                return None  # No signal
            
            # Calculate entry, target, and stop loss
            entry_price = token.price_usd
            target_price = entry_price * (1 + self.config['target_profit_pct'])
            stop_loss = entry_price * (1 - self.config['stop_loss_pct'])
            
            # Position sizing
            position_size = self.config['position_size_pct']  # 2% of portfolio
            
            # Generate reasoning
            viral_indicators = [
                f"Viral score: {token.viral_score:.2f}",
                f"Age: {token.age_minutes:.1f} minutes",
                f"Holders: {token.holder_count}",
                f"Social presence: {bool(token.twitter or token.telegram)}"
            ]
            
            risk_factors = [
                f"Risk score: {token.risk_score:.2f}",
                f"Liquidity: ${token.initial_liquidity:,.0f}",
                f"Buy/Sell ratio: {token.buy_count}/{token.sell_count}"
            ]
            
            reasoning = f"Pump.fun new launch with {token.viral_score:.2f} viral score"
            
            signal = PumpFunSignal(
                token_address=token.address,
                token_symbol=token.symbol,
                signal_type=signal_type,
                confidence=confidence,
                urgency=urgency,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                position_size=position_size,
                reasoning=reasoning,
                viral_indicators=viral_indicators,
                risk_factors=risk_factors,
                timestamp=time.time(),
                expires_at=time.time() + 300  # 5 minute expiry
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return None
    
    async def _process_websocket_message(self, data: Dict):
        """Process incoming WebSocket message"""
        try:
            message_type = data.get('type')
            
            if message_type == 'new_token':
                await self._process_new_token(data.get('token', {}))
            elif message_type == 'token_update':
                await self._update_token(data.get('token', {}))
            elif message_type == 'trade':
                await self._process_trade(data.get('trade', {}))
                
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    async def _process_signals(self):
        """Background task to process and clean up signals"""
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                # Remove expired signals
                expired_signals = [
                    addr for addr, signal in self.generated_signals.items()
                    if current_time > signal.expires_at
                ]
                
                for addr in expired_signals:
                    del self.generated_signals[addr]
                
                try:
                    await asyncio.sleep(10)  # Check every 10 seconds
                except asyncio.CancelledError:
                    break
                
            except asyncio.CancelledError:
                logger.info("Signal processing cancelled")
                break
            except Exception as e:
                if self.monitoring_active:
                    logger.error(f"Signal processing error: {e}")
                    try:
                        await asyncio.sleep(10)
                    except asyncio.CancelledError:
                        break
                else:
                    break
    
    async def stop_monitoring(self):
        """Stop monitoring pump.fun"""
        try:
            logger.info("🛑 Stopping pump.fun monitoring...")
            self.monitoring_active = False
            
            if self.ws_connection:
                await self.ws_connection.close()
            
            logger.info("✅ Pump.fun monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
    
    def get_active_signals(self) -> List[PumpFunSignal]:
        """Get currently active trading signals"""
        current_time = time.time()
        return [
            signal for signal in self.generated_signals.values()
            if current_time <= signal.expires_at
        ]
    
    def get_detected_tokens(self, limit: int = 20) -> List[PumpFunToken]:
        """Get recently detected tokens"""
        tokens = list(self.detected_tokens.values())
        tokens.sort(key=lambda t: t.created_at, reverse=True)
        return tokens[:limit]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.performance_metrics,
            'active_signals': len(self.generated_signals),
            'detected_tokens_24h': len([
                t for t in self.detected_tokens.values()
                if time.time() - t.created_at < 86400
            ])
        }

# Integration helper functions
async def setup_pump_fun_monitor(callback_handler: Optional[Callable] = None) -> PumpFunMonitor:
    """Set up and start pump.fun monitoring"""
    monitor = PumpFunMonitor(callback_handler)
    success = await monitor.start_monitoring()
    
    if success:
        logger.info("✅ Pump.fun monitor setup complete")
    else:
        logger.error("❌ Failed to setup pump.fun monitor")
    
    return monitor 