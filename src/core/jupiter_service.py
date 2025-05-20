"""
Jupiter API service for swap execution on Solana.
"""
import aiohttp
import asyncio
import logging
from typing import Dict, List, Optional, Any
import base58
from solana.transaction import Transaction
from solana.keypair import Keypair
from solana.publickey import PublicKey
from solana.instruction import Instruction
from ..utils.config import settings

logger = logging.getLogger(__name__)

class JupiterService:
    def __init__(self):
        self.base_url = settings.JUPITER_API_URL
        self.session: Optional[aiohttp.ClientSession] = None
        self._cache = {}

    async def initialize(self) -> bool:
        """Initialize the Jupiter service."""
        try:
            self.session = aiohttp.ClientSession()
            logger.info("Jupiter service initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Jupiter service: {str(e)}")
            return False

    async def close(self) -> None:
        """Close the Jupiter service."""
        if self.session:
            await self.session.close()

    async def get_swap_quote(
        self, 
        input_token: str, 
        output_token: str, 
        amount: float,
        slippage_bps: int = None
    ) -> Optional[Dict]:
        """Get swap quote from Jupiter."""
        try:
            if not self.session:
                await self.initialize()
                
            if slippage_bps is None:
                slippage_bps = int(settings.MIN_SLIPPAGE * 10000)  # Convert percentage to basis points
                
            url = f"{self.base_url}/quote"
            params = {
                "inputMint": input_token,
                "outputMint": output_token,
                "amount": str(int(amount * 1e9)),  # Convert to lamports
                "slippageBps": slippage_bps
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    quote_data = await response.json()
                    logger.info(f"Got swap quote for {input_token} -> {output_token}")
                    return quote_data
                else:
                    logger.error(f"Failed to get swap quote: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting swap quote: {str(e)}")
            return None

    async def get_swap_instructions(self, quote: Dict, wallet_public_key: str) -> Optional[Dict]:
        """Get swap instructions from Jupiter."""
        try:
            if not self.session:
                await self.initialize()
                
            url = f"{self.base_url}/swap"
            data = {
                "quoteResponse": quote,
                "userPublicKey": wallet_public_key,
                "wrapUnwrapSOL": True  # Handle wrapped SOL conversion
            }
            
            async with self.session.post(url, json=data) as response:
                if response.status == 200:
                    swap_data = await response.json()
                    logger.info(f"Got swap instructions")
                    return swap_data
                else:
                    logger.error(f"Failed to get swap instructions: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting swap instructions: {str(e)}")
            return None
            
    async def create_swap_transaction(self, quote: Dict, wallet_public_key: str) -> Optional[Dict]:
        """Create a swap transaction from quote."""
        try:
            swap_data = await self.get_swap_instructions(quote, wallet_public_key)
            if not swap_data:
                return None
                
            return {
                'swapTransaction': swap_data.get('swapTransaction'),
                'expectedOutputAmount': quote.get('outAmount'),
                'inputAmount': quote.get('inAmount'),
                'price': float(quote.get('outAmount')) / float(quote.get('inAmount'))
            }
        except Exception as e:
            logger.error(f"Error creating swap transaction: {str(e)}")
            return None
            
    async def get_tokens(self) -> Optional[List[Dict]]:
        """Get list of tokens supported by Jupiter."""
        try:
            if not self.session:
                await self.initialize()
                
            url = f"{self.base_url}/tokens"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    tokens_data = await response.json()
                    logger.info(f"Got {len(tokens_data)} tokens from Jupiter")
                    return tokens_data
                else:
                    logger.error(f"Failed to get tokens: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting tokens: {str(e)}")
            return None
            
    async def get_routes(self, input_token: str, output_token: str) -> Optional[List[Dict]]:
        """Get available routes between input and output tokens."""
        try:
            if not self.session:
                await self.initialize()
                
            url = f"{self.base_url}/routes"
            params = {
                "inputMint": input_token,
                "outputMint": output_token
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    routes_data = await response.json()
                    logger.info(f"Got routes for {input_token} -> {output_token}")
                    return routes_data.get('data', [])
                else:
                    logger.error(f"Failed to get routes: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting routes: {str(e)}")
            return None
            
    async def get_price(self, token_mint: str, vs_token: str = "USDC") -> Optional[float]:
        """Get token price in terms of vs_token."""
        try:
            # First get the vs_token mint if it's not a mint address already
            vs_token_mint = vs_token
            if not vs_token.startswith(""):  # Check if it's not already a mint address
                tokens = await self.get_tokens()
                if tokens:
                    for token in tokens:
                        if token.get('symbol') == vs_token:
                            vs_token_mint = token.get('address')
                            break
            
            # Then get a quote for a small amount to determine price
            quote = await self.get_swap_quote(
                token_mint,
                vs_token_mint,
                0.001  # Small amount for price check
            )
            
            if quote:
                in_amount = float(quote.get('inAmount', 0))
                out_amount = float(quote.get('outAmount', 0))
                if in_amount > 0:
                    return out_amount / in_amount
            
            return None
        except Exception as e:
            logger.error(f"Error getting price: {str(e)}")
            return None 