"""
Jupiter service for Solana DEX aggregation with real trading capabilities.
"""
import asyncio
import aiohttp
import json
import logging
import time
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from loguru import logger

@dataclass
class SwapQuote:
    """Represents a swap quote from Jupiter."""
    input_mint: str
    output_mint: str
    input_amount: int
    output_amount: int
    price_impact_pct: float
    platform_fee_pct: float
    route_plan: List[Dict]
    time_taken_ms: int
    quote_id: str = ""
    
    @property
    def price(self) -> float:
        """Calculate effective price per input token."""
        return self.output_amount / self.input_amount if self.input_amount > 0 else 0
    
    @property
    def slippage_bps(self) -> int:
        """Convert price impact to basis points."""
        return int(self.price_impact_pct * 100)

@dataclass
class SwapResult:
    """Result of a swap execution."""
    success: bool
    transaction_id: Optional[str] = None
    input_amount: int = 0
    output_amount: int = 0
    actual_price: float = 0
    slippage: float = 0
    gas_used: int = 0
    execution_time: float = 0
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

class JupiterAPIError(Exception):
    """Raised when Jupiter API calls fail"""
    pass

class JupiterService:
    """
    Real Jupiter service for DEX aggregation and swap execution.
    Supports both live trading and simulation mode.
    """
    
    def __init__(self):
        """Initialize Jupiter service with real API configuration."""
        self.simulation_mode = os.getenv("SIMULATION_MODE", "true").lower() == "true"
        self.api_key = os.getenv("JUPITER_API_KEY", "")
        self.base_url = "https://quote-api.jup.ag/v6"
        
        # Rate limiting
        self.max_requests_per_second = 5
        self.request_interval = 1.0 / self.max_requests_per_second
        self.last_request_time = 0
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self.timeout = aiohttp.ClientTimeout(total=30)
        
        # Trading settings
        self.default_slippage_bps = 300  # 3%
        self.max_slippage_bps = 1000     # 10%
        self.min_output_amount_threshold = 0.95  # 95% of quoted amount
        
        # Performance tracking
        self.successful_swaps = 0
        self.failed_swaps = 0
        self.total_volume_traded = 0
        
        # Validation
        self._validate_configuration()
        
        logger.info(f"Jupiter service initialized in {'simulation' if self.simulation_mode else 'live'} mode")
    
    def _validate_configuration(self) -> None:
        """Validate Jupiter service configuration."""
        if not self.simulation_mode:
            # For live mode, we don't strictly require API key as Jupiter quote API is public
            # But if provided, it should be valid
            if self.api_key and self.api_key in ["", "demo_key_for_testing"]:
                logger.warning("Using demo Jupiter API key - consider getting real key for better rate limits")
            
            logger.info("Jupiter service configured for live trading")
        else:
            logger.info("Running in simulation mode - using mock data")
    
    async def _ensure_session(self) -> None:
        """Ensure HTTP session is created."""
        if not self.session or self.session.closed:
            headers = {"User-Agent": "Solana-Trading-Bot/1.0"}
            if self.api_key and self.api_key not in ["", "demo_key_for_testing"]:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            self.session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers=headers
            )
    
    async def _rate_limit(self) -> None:
        """Implement rate limiting for API calls."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.request_interval:
            sleep_time = self.request_interval - time_since_last_request
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    async def _make_api_request(self, endpoint: str, method: str = "GET", params: Dict = None, data: Dict = None) -> Dict:
        """Make API request to Jupiter."""
        if self.simulation_mode:
            return self._get_simulation_data(endpoint, params)
        
        await self._ensure_session()
        await self._rate_limit()
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                params=params,
                json=data
            ) as response:
                
                if response.status == 429:
                    # Rate limited, wait and retry
                    await asyncio.sleep(2)
                    return await self._make_api_request(endpoint, method, params, data)
                
                if response.status >= 400:
                    error_text = await response.text()
                    raise JupiterAPIError(f"API request failed: {response.status} - {error_text}")
                
                return await response.json()
                
        except aiohttp.ClientError as e:
            raise JupiterAPIError(f"Network error: {str(e)}")
        except Exception as e:
            raise JupiterAPIError(f"Unexpected error: {str(e)}")
    
    def _get_simulation_data(self, endpoint: str, params: Dict = None) -> Dict:
        """Generate realistic simulation data for Jupiter API."""
        if "quote" in endpoint:
            input_amount = int(params.get("amount", 1000000)) if params else 1000000
            # Simulate realistic price impact and routing
            price_impact = min(0.05, input_amount / 10000000)  # Higher amount = more impact
            output_amount = int(input_amount * 0.99 * (1 - price_impact))  # 1% base fee + impact
            
            return {
                "inputMint": params.get("inputMint", "So11111111111111111111111111111111111111112"),
                "inAmount": str(input_amount),
                "outputMint": params.get("outputMint", "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"),
                "outAmount": str(output_amount),
                "priceImpactPct": price_impact,
                "platformFee": {"amount": str(input_amount // 1000), "feeBps": 100},
                "routePlan": [
                    {
                        "swapInfo": {
                            "ammKey": "SIM_AMM_KEY",
                            "label": "Raydium",
                            "inputMint": params.get("inputMint"),
                            "outputMint": params.get("outputMint"),
                            "inAmount": str(input_amount),
                            "outAmount": str(output_amount),
                            "feeAmount": str(input_amount // 1000),
                            "feeMint": params.get("inputMint")
                        }
                    }
                ],
                "timeTaken": 150
            }
        elif "swap" in endpoint:
            return {
                "txid": f"SIM_SWAP_{int(time.time() * 1000000)}",
                "success": True
            }
        elif "tokens" in endpoint:
            return [
                {
                    "address": f"SIM{i:010d}TokenAddress",
                    "symbol": f"SIM{i}",
                    "name": f"Simulation Token {i}",
                    "decimals": 9,
                    "logoURI": f"https://example.com/sim{i}.png",
                    "tags": ["simulation"]
                }
                for i in range(1, 11)
            ]
        else:
            return {"simulated": True, "endpoint": endpoint, "timestamp": time.time()}
    
    async def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: Optional[int] = None
    ) -> Optional[SwapQuote]:
        """Get a swap quote from Jupiter."""
        try:
            slippage_bps = slippage_bps or self.default_slippage_bps
            
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(amount),
                "slippageBps": str(slippage_bps),
                "swapMode": "ExactIn",
                "onlyDirectRoutes": "false",
                "asLegacyTransaction": "false"
            }
            
            response = await self._make_api_request("quote", params=params)
            
            if not response:
                return None
            
            return SwapQuote(
                input_mint=response["inputMint"],
                output_mint=response["outputMint"],
                input_amount=int(response["inAmount"]),
                output_amount=int(response["outAmount"]),
                price_impact_pct=float(response.get("priceImpactPct", 0)),
                platform_fee_pct=float(response.get("platformFee", {}).get("feeBps", 0)) / 10000,
                route_plan=response.get("routePlan", []),
                time_taken_ms=response.get("timeTaken", 0),
                quote_id=response.get("quoteResponse", "")
            )
            
        except Exception as e:
            logger.error(f"Failed to get quote: {str(e)}")
            return None

    async def execute_swap(
        self,
        quote: SwapQuote,
        wallet_manager,
        max_slippage_bps: Optional[int] = None
    ) -> SwapResult:
        """Execute a swap based on a quote."""
        start_time = time.time()
        
        try:
            if self.simulation_mode:
                # Simulate swap execution
                await asyncio.sleep(0.2)  # Simulate network delay
                
                # Simulate occasional failures (5% chance)
                if time.time() % 20 < 1:
                    return SwapResult(
                        success=False,
                        error="Simulated network timeout",
                        execution_time=time.time() - start_time
                    )
                
                # Simulate successful swap with slight slippage
                actual_slippage = quote.price_impact_pct * (0.8 + 0.4 * (time.time() % 1))
                actual_output = int(quote.output_amount * (1 - actual_slippage / 100))
                
                self.successful_swaps += 1
                self.total_volume_traded += quote.input_amount
                
                return SwapResult(
                    success=True,
                    transaction_id=f"SIM_SWAP_{int(time.time() * 1000000)}",
                    input_amount=quote.input_amount,
                    output_amount=actual_output,
                    actual_price=actual_output / quote.input_amount,
                    slippage=actual_slippage,
                    gas_used=50000,  # Simulated gas
                    execution_time=time.time() - start_time
                )
            
            # Real swap execution
            max_slippage_bps = max_slippage_bps or self.max_slippage_bps
            
            # Validate slippage is within acceptable limits
            if quote.slippage_bps > max_slippage_bps:
                return SwapResult(
                    success=False,
                    error=f"Price impact too high: {quote.price_impact_pct:.2f}% > {max_slippage_bps/100:.2f}%",
                    execution_time=time.time() - start_time
                )
            
            # Validate wallet has sufficient balance
            balance = await wallet_manager.check_balance()
            # Convert balance to lamports (assuming SOL input for simplicity)
            balance_lamports = int(balance * 1_000_000_000)
            
            if quote.input_mint == "So11111111111111111111111111111111111111112":  # SOL
                if balance_lamports < quote.input_amount:
                    return SwapResult(
                        success=False,
                        error=f"Insufficient balance: {balance} SOL < {quote.input_amount / 1_000_000_000} SOL",
                        execution_time=time.time() - start_time
                    )
            
            # Build swap transaction
            swap_data = {
                "quoteResponse": quote.__dict__,
                "userPublicKey": wallet_manager.get_public_key(),
                "wrapAndUnwrapSol": True,
                "useSharedAccounts": True,
                "feeAccount": None,
                "trackingAccount": None,
                "computeUnitPriceMicroLamports": "auto",
                "prioritizationFeeLamports": "auto"
            }
            
            # Get swap transaction from Jupiter
            swap_response = await self._make_api_request("swap", method="POST", data=swap_data)
            
            if not swap_response.get("swapTransaction"):
                return SwapResult(
                    success=False,
                    error="Failed to get swap transaction from Jupiter",
                    execution_time=time.time() - start_time
                )
            
            # Decode and send transaction
            # Note: This is a simplified version - full implementation would need
            # proper transaction deserialization and signing
            tx_id = await wallet_manager.send_raw_transaction(swap_response["swapTransaction"])
            
            # Wait for confirmation
            await asyncio.sleep(2)
            tx_status = await wallet_manager.get_transaction_status(tx_id)
            
            if tx_status.get("status") == "confirmed":
                self.successful_swaps += 1
                self.total_volume_traded += quote.input_amount
                
                return SwapResult(
                    success=True,
                    transaction_id=tx_id,
                    input_amount=quote.input_amount,
                    output_amount=quote.output_amount,  # Would need to parse from transaction
                    actual_price=quote.price,
                    slippage=quote.price_impact_pct,
                    execution_time=time.time() - start_time
                )
            else:
                self.failed_swaps += 1
                return SwapResult(
                    success=False,
                    error=f"Transaction failed: {tx_status.get('error', 'Unknown error')}",
                    execution_time=time.time() - start_time
                )
                
        except Exception as e:
            self.failed_swaps += 1
            logger.error(f"Swap execution failed: {str(e)}")
            return SwapResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    async def get_supported_tokens(self) -> List[Dict]:
        """Get list of tokens supported by Jupiter."""
        try:
            response = await self._make_api_request("tokens")
            return response if isinstance(response, list) else []
            
        except Exception as e:
            logger.error(f"Failed to get supported tokens: {str(e)}")
            return []
    
    async def get_token_price(self, token_mint: str, vs_token_mint: str = "So11111111111111111111111111111111111111112") -> Optional[float]:
        """Get token price in terms of another token (default: SOL)."""
        try:
            # Get quote for 1 token
            quote = await self.get_quote(
                input_mint=token_mint,
                output_mint=vs_token_mint,
                amount=1_000_000_000  # 1 token (assuming 9 decimals)
            )
            
            if quote:
                return quote.output_amount / 1_000_000_000  # Convert back to tokens
            
            return None
                    
        except Exception as e:
            logger.error(f"Failed to get token price: {str(e)}")
            return None
    
    async def find_arbitrage_opportunities(self, token_mints: List[str], min_profit_pct: float = 1.0) -> List[Dict]:
        """Find arbitrage opportunities between different DEXes."""
        try:
            if self.simulation_mode:
                # Return simulated arbitrage opportunities
                return [
                    {
                        "token_in": token_mints[0] if token_mints else "So11111111111111111111111111111111111111112",
                        "token_out": token_mints[1] if len(token_mints) > 1 else "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                        "profit_pct": 2.5,
                        "amount": 1000000000,
                        "dex_route": ["Raydium", "Orca"],
                        "confidence": 0.8
                    }
                ] if len(token_mints) >= 2 else []
            
            arbitrage_opportunities = []
            
            # Check all token pairs for arbitrage
            for i, token_a in enumerate(token_mints):
                for token_b in token_mints[i+1:]:
                    # Get quotes in both directions
                    quote_a_to_b = await self.get_quote(token_a, token_b, 1_000_000_000)
                    quote_b_to_a = await self.get_quote(token_b, token_a, 1_000_000_000)
                    
                    if quote_a_to_b and quote_b_to_a:
                        # Calculate potential profit
                        forward_rate = quote_a_to_b.price
                        reverse_rate = 1 / quote_b_to_a.price
                        
                        profit_pct = ((forward_rate - reverse_rate) / reverse_rate) * 100
                        
                        if profit_pct > min_profit_pct:
                            arbitrage_opportunities.append({
                                "token_in": token_a,
                                "token_out": token_b,
                                "profit_pct": profit_pct,
                                "amount": 1_000_000_000,
                                "forward_quote": quote_a_to_b,
                                "reverse_quote": quote_b_to_a
                            })
            
            return arbitrage_opportunities
                
        except Exception as e:
            logger.error(f"Failed to find arbitrage opportunities: {str(e)}")
            return []
            
    async def get_random_tokens(self, count: int = 5) -> List[Dict]:
        """Get random tokens for discovery."""
        try:
            all_tokens = await self.get_supported_tokens()
            
            if not all_tokens:
                return []
                
            # Filter out stable coins and major tokens for meme coin discovery
            filtered_tokens = [
                token for token in all_tokens
                if token.get("symbol", "").upper() not in ["USDC", "USDT", "SOL", "BTC", "ETH"]
                and "stable" not in token.get("tags", [])
            ]
            
            # Return random selection
            import random
            return random.sample(filtered_tokens, min(count, len(filtered_tokens)))
            
        except Exception as e:
            logger.error(f"Failed to get random tokens: {str(e)}")
            return []
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_swaps = self.successful_swaps + self.failed_swaps
        success_rate = (self.successful_swaps / total_swaps * 100) if total_swaps > 0 else 0
        
        return {
            "total_swaps": total_swaps,
            "successful_swaps": self.successful_swaps,
            "failed_swaps": self.failed_swaps,
            "success_rate": success_rate,
            "total_volume_traded": self.total_volume_traded,
            "avg_volume_per_swap": self.total_volume_traded / max(self.successful_swaps, 1)
        }
    
    async def close(self) -> None:
        """Close HTTP session and cleanup."""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
            
            logger.info("Jupiter service closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing Jupiter service: {str(e)}")
    
    def __del__(self):
        """Ensure session is closed on object destruction."""
        if hasattr(self, 'session') and self.session and not self.session.closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.session.close())
                else:
                    loop.run_until_complete(self.session.close())
            except Exception:
                pass  # Ignore cleanup errors 