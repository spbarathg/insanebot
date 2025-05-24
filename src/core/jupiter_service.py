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
        """Initialize Jupiter Service with real API integration only."""
        self.base_url = "https://quote-api.jup.ag/v6"
        self.session = None
        self.default_slippage_bps = 100  # 1% default slippage
        self.max_retries = 3
        self.timeout = 30
        
        # Real-time cache for performance optimization
        self._quote_cache = {}
        self._cache_ttl = 5  # 5 seconds cache for quotes (very short for real-time data)
        
        logger.info("Jupiter Service initialized - REAL API MODE ONLY")
    
    async def _make_api_request(self, endpoint: str, params: Dict = None, method: str = "GET") -> Optional[Dict]:
        """Make API request to Jupiter with real data only."""
        try:
            if not self.session:
                import aiohttp
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                self.session = aiohttp.ClientSession(timeout=timeout)
            
            url = f"{self.base_url}/{endpoint}"
            
            for attempt in range(self.max_retries):
                try:
                    if method == "GET":
                        async with self.session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                logger.debug(f"Jupiter API success: {endpoint}")
                                return data
                            elif response.status == 429:  # Rate limited
                                wait_time = 2 ** attempt
                                logger.warning(f"Jupiter API rate limited, waiting {wait_time}s")
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                logger.warning(f"Jupiter API error {response.status} for {endpoint}")
                                break
                    elif method == "POST":
                        async with self.session.post(url, json=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                logger.debug(f"Jupiter API POST success: {endpoint}")
                                return data
                            elif response.status == 429:  # Rate limited
                                wait_time = 2 ** attempt
                                logger.warning(f"Jupiter API rate limited, waiting {wait_time}s")
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                logger.warning(f"Jupiter API POST error {response.status} for {endpoint}")
                                break
                except asyncio.TimeoutError:
                    logger.warning(f"Jupiter API timeout on attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                except Exception as e:
                    logger.error(f"Jupiter API request error: {str(e)}")
                    break
            
            logger.error(f"Jupiter API request failed after {self.max_retries} attempts: {endpoint}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to make Jupiter API request: {str(e)}")
            return None
    
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
                return SwapResult(
                    success=False,
                    error=f"Transaction failed: {tx_status.get('error', 'Unknown error')}",
                    execution_time=time.time() - start_time
                )
                
        except Exception as e:
            logger.error(f"Swap execution failed: {str(e)}")
            return SwapResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    async def get_supported_tokens(self) -> List[Dict]:
        """Get list of all supported tokens from Jupiter."""
        try:
            response = await self._make_api_request("tokens")
            return response if isinstance(response, list) else response.get("tokens", [])
            
        except Exception as e:
            logger.error(f"Failed to get supported tokens: {str(e)}")
            return []
    
    async def get_tokens(self) -> List[Dict]:
        """Alias for get_supported_tokens - used by cross-dex scanner."""
        return await self.get_supported_tokens()
    
    async def get_swap_quote(self, input_mint: str, output_mint: str, amount: int, slippage_bps: Optional[int] = None) -> Optional[SwapQuote]:
        """Alias for get_quote - used by cross-dex scanner."""
        return await self.get_quote(input_mint, output_mint, amount, slippage_bps)
    
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
        """Find REAL arbitrage opportunities using actual market data."""
        try:
            arbitrage_opportunities = []
            
            # Get real market data for arbitrage analysis
            for i, token_mint in enumerate(token_mints):
                for j, other_token in enumerate(token_mints[i+1:], start=i+1):
                    try:
                        # Get real quotes in both directions
                        quote_a_to_b = await self.get_quote(token_mint, other_token, 1_000_000_000)  # 1 token
                        quote_b_to_a = await self.get_quote(other_token, token_mint, 1_000_000_000)  # 1 token
                        
                        if quote_a_to_b and quote_b_to_a:
                            # Calculate real arbitrage potential
                            price_a_to_b = float(quote_a_to_b.output_amount) / float(quote_a_to_b.input_amount)
                            price_b_to_a = float(quote_b_to_a.output_amount) / float(quote_b_to_a.input_amount)
                            
                            # Check for arbitrage opportunity
                            round_trip_rate = price_a_to_b * price_b_to_a
                            profit_pct = (round_trip_rate - 1) * 100
                            
                            # Only include real profitable opportunities
                            if profit_pct >= min_profit_pct:
                                arbitrage_opportunities.append({
                                    "token_in": token_mint,
                                    "token_out": other_token,
                                    "profit_pct": profit_pct,
                                    "amount": quote_a_to_b.input_amount,
                                    "route_plan": quote_a_to_b.route_plan + quote_b_to_a.route_plan,
                                    "confidence": min(0.9, profit_pct / 10),  # Real confidence based on profit margin
                                    "price_impact_a_to_b": quote_a_to_b.price_impact_pct,
                                    "price_impact_b_to_a": quote_b_to_a.price_impact_pct,
                                    "timestamp": time.time()
                                })
                                
                    except Exception as e:
                        logger.debug(f"Error checking arbitrage between {token_mint[:8]}... and {other_token[:8]}...: {str(e)}")
                        continue
            
            # Sort by profitability (real opportunities first)
            arbitrage_opportunities.sort(key=lambda x: x["profit_pct"], reverse=True)
            
            logger.info(f"Found {len(arbitrage_opportunities)} real arbitrage opportunities")
            return arbitrage_opportunities[:10]  # Return top 10 real opportunities
            
        except Exception as e:
            logger.error(f"Failed to find arbitrage opportunities: {str(e)}")
            return []
            
    async def get_random_tokens(self, count: int = 10) -> List[Dict]:
        """Get a list of real popular tokens from Jupiter (not random simulation)."""
        try:
            # Get all token addresses from Jupiter
            token_addresses = await self.get_supported_tokens()
            
            if not token_addresses:
                logger.warning("No token addresses available from Jupiter API")
                return []
            
            # Jupiter /tokens endpoint returns array of strings (addresses), not objects
            if isinstance(token_addresses, list) and len(token_addresses) > 0:
                # If it's a list of strings (addresses), we need to create token objects
                if isinstance(token_addresses[0], str):
                    # Common well-known Solana tokens with metadata
                    known_tokens = [
                        {
                            "address": "So11111111111111111111111111111111111111112",
                            "symbol": "SOL",
                            "name": "Solana",
                            "decimals": 9
                        },
                        {
                            "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                            "symbol": "USDC",
                            "name": "USD Coin",
                            "decimals": 6
                        },
                        {
                            "address": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
                            "symbol": "USDT",
                            "name": "Tether USD",
                            "decimals": 6
                        },
                        {
                            "address": "DezXAZ8zDXzK82sYdDbGNQYJuUFzJPCL7yRNmEHYYAjK",
                            "symbol": "BONK",
                            "name": "Bonk",
                            "decimals": 5
                        },
                        {
                            "address": "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs",
                            "symbol": "ETH",
                            "name": "Ether",
                            "decimals": 8
                        },
                        {
                            "address": "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj",
                            "symbol": "STEP",
                            "name": "Step Finance",
                            "decimals": 9
                        },
                        {
                            "address": "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",
                            "symbol": "MSOL",
                            "name": "Marinade Staked SOL",
                            "decimals": 9
                        },
                        {
                            "address": "SRMuApVNdxXokk5GT7XD5cUUgXMBCoAz2LHeuAoKWRt",
                            "symbol": "SRM",
                            "name": "Serum",
                            "decimals": 6
                        }
                    ]
                    
                    # Filter known tokens that are available in Jupiter
                    available_tokens = []
                    for known_token in known_tokens:
                        if known_token["address"] in token_addresses:
                            available_tokens.append(known_token)
                    
                    # Add additional tokens from Jupiter list (as minimal objects)
                    additional_count = max(0, count - len(available_tokens))
                    for i, address in enumerate(token_addresses[:additional_count]):
                        if address not in [t["address"] for t in available_tokens]:
                            available_tokens.append({
                                "address": address,
                                "symbol": f"TOKEN_{address[:8]}",
                                "name": f"Token {address[:8]}",
                                "decimals": 9
                            })
                    
                    return available_tokens[:count]
                
                # If it's already a list of objects, filter them properly
                else:
                    valid_tokens = []
                    for token in token_addresses:
                        # Skip test/simulation tokens
                        symbol = token.get("symbol", "").upper()
                        if any(test_word in symbol for test_word in ["TEST", "SIM", "DEMO", "MOCK", "FAKE"]):
                            continue
                        
                        # Skip tokens without proper metadata
                        if not token.get("name") or not token.get("symbol") or not token.get("address"):
                            continue
                        
                        # Only include tokens with valid addresses (not simulation addresses)
                        address = token.get("address", "")
                        if address.startswith("SIM") or len(address) < 32:
                            continue
                        
                        valid_tokens.append(token)
                    
                    # Sort by popularity/liquidity if available, otherwise randomize
                    import random
                    if len(valid_tokens) > count:
                        # Prefer well-known tokens first, then randomly select from the rest
                        popular_tokens = []
                        other_tokens = []
                        
                        known_symbols = {"SOL", "USDC", "USDT", "BONK", "WIF", "JTO", "PYTH", "RAY", "ORCA", "MNGO"}
                        
                        for token in valid_tokens:
                            if token.get("symbol", "").upper() in known_symbols:
                                popular_tokens.append(token)
                            else:
                                other_tokens.append(token)
                        
                        # Take popular tokens first, then fill with random others
                        selected_tokens = popular_tokens[:count]
                        if len(selected_tokens) < count:
                            remaining_count = count - len(selected_tokens)
                            random.shuffle(other_tokens)
                            selected_tokens.extend(other_tokens[:remaining_count])
                        
                        return selected_tokens[:count]
                    else:
                        return valid_tokens[:count]
            
            return []
                
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