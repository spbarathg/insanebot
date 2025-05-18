import asyncio
import json
import time
from typing import Dict, List, Optional, Tuple
from loguru import logger
import aiohttp
from solana.rpc.async_api import AsyncClient
from solana.transaction import Transaction
from solana.keypair import Keypair
from solana.rpc.commitment import Confirmed
from spl.token.instructions import get_associated_token_address
from spl.token.constants import TOKEN_PROGRAM_ID
from src.core.config import CORE_CONFIG, TRADING_CONFIG

class TradeExecution:
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.solana_client: Optional[AsyncClient] = None
        self.active_trades: Dict[str, Dict] = {}
        self.pre_signed_txs: List[Transaction] = []
        self.current_rpc_index = 0
        self.last_rpc_switch = time.time()
        self.priority_fee = 0.000005  # Base priority fee in SOL
        self._load_wallet()

    def _load_wallet(self):
        """Load wallet from private key"""
        try:
            private_key = bytes.fromhex(CORE_CONFIG.SOLANA_PRIVATE_KEY)
            self.wallet = Keypair.from_secret_key(private_key)
        except Exception as e:
            logger.error(f"Load wallet error: {e}")
            raise

    async def initialize(self):
        """Initialize trade execution"""
        self.session = aiohttp.ClientSession()
        self.solana_client = AsyncClient(CORE_CONFIG.RPC_ENDPOINTS[0])
        await self._refresh_pre_signed_txs()

    async def close(self):
        """Close trade execution"""
        if self.session:
            await self.session.close()
        if self.solana_client:
            await self.solana_client.close()

    async def _refresh_pre_signed_txs(self):
        """Refresh pool of pre-signed transactions"""
        try:
            # Create new transactions
            new_txs = []
            for _ in range(5):  # Keep 5 pre-signed transactions
                tx = await self._create_swap_transaction()
                if tx:
                    new_txs.append(tx)

            self.pre_signed_txs = new_txs
            logger.info(f"Refreshed {len(new_txs)} pre-signed transactions")

        except Exception as e:
            logger.error(f"Refresh pre-signed transactions error: {e}")

    async def _get_liquidity_info(self, token: str) -> Optional[Dict]:
        """Get token liquidity information"""
        try:
            async with self.session.get(
                f"https://api.dexscreener.com/latest/dex/tokens/{token}"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('pairs'):
                        pair = data['pairs'][0]  # Get most liquid pair
                        return {
                            'amount': float(pair.get('liquidity', {}).get('usd', 0)),
                            'lock_time': int(pair.get('lockTime', 0)),
                            'market_cap': float(pair.get('fdv', 0))
                        }
                return None
        except Exception as e:
            logger.error(f"Get liquidity info error: {e}")
            return None

    async def _get_token_info(self, token: str) -> Optional[Dict]:
        """Get token information"""
        try:
            async with self.session.get(
                f"https://public-api.birdeye.so/public/token_info?address={token}",
                headers={'X-API-KEY': TRADING_CONFIG.BIRDEYE_API_KEY}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'holder_count': data.get('holderCount', 0),
                        'owner_concentration': data.get('ownerConcentration', 1.0),
                        'age': data.get('age', 0)
                    }
                return None
        except Exception as e:
            logger.error(f"Get token info error: {e}")
            return None

    async def _calculate_price_impact(self, token: str, amount: float) -> float:
        """Calculate price impact of trade"""
        try:
            async with self.session.get(
                f"https://quote-api.jup.ag/v6/quote?inputMint=So11111111111111111111111111111111111111112&outputMint={token}&amount={int(amount * 1e9)}"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data.get('priceImpactPct', 1.0))
                return 1.0
        except Exception as e:
            logger.error(f"Calculate price impact error: {e}")
            return 1.0

    async def _create_swap_transaction(self) -> Optional[Transaction]:
        """Create a pre-signed swap transaction"""
        try:
            # Get quote from Jupiter
            quote = await self._get_swap_quote()
            if not quote:
                return None

            # Create transaction
            tx = Transaction()
            
            # Add swap instruction
            swap_ix = await self._create_swap_instruction(quote)
            if not swap_ix:
                return None
                
            tx.add(swap_ix)
            
            # Sign transaction
            tx.sign(self.wallet)
            
            return tx
        except Exception as e:
            logger.error(f"Create swap transaction error: {e}")
            return None

    async def _get_swap_quote(self) -> Optional[Dict]:
        """Get swap quote from Jupiter"""
        try:
            async with self.session.get(
                "https://quote-api.jup.ag/v6/quote",
                params={
                    "inputMint": "So11111111111111111111111111111111111111112",
                    "outputMint": "YOUR_TOKEN_MINT",
                    "amount": "1000000000"  # 1 SOL
                }
            ) as response:
                if response.status == 200:
                    return await response.json()
                return None
        except Exception as e:
            logger.error(f"Get swap quote error: {e}")
            return None

    async def _create_swap_instruction(self, quote: Dict) -> Optional[Instruction]:
        """Create swap instruction from quote"""
        try:
            # Get swap instruction from Jupiter
            async with self.session.post(
                "https://quote-api.jup.ag/v6/swap",
                json={
                    "quoteResponse": quote,
                    "userPublicKey": str(self.wallet.public_key),
                    "wrapUnwrapSOL": True
                }
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('swapInstruction')
                return None
        except Exception as e:
            logger.error(f"Create swap instruction error: {e}")
            return None

    async def _execute_buy_order(self, token: str, amount: float, fee: float) -> Optional[Dict]:
        """Execute buy order"""
        try:
            # Get quote
            quote = await self._get_swap_quote()
            if not quote:
                return None

            # Create and sign transaction
            tx = await self._create_swap_transaction()
            if not tx:
                return None

            # Send transaction
            result = await self.solana_client.send_transaction(
                tx,
                self.wallet,
                opts={
                    'skip_preflight': True,
                    'max_retries': 3,
                    'preflight_commitment': Confirmed
                }
            )

            if result.get('result'):
                return {
                    'signature': result['result'],
                    'price': float(quote.get('outAmount', 0)) / float(quote.get('inAmount', 1)),
                    'amount': amount
                }
            return None

        except Exception as e:
            logger.error(f"Execute buy order error: {e}")
            return None

    async def _execute_sell_order(self, token: str, amount: float, fee: float) -> Optional[Dict]:
        """Execute sell order"""
        try:
            # Get quote (reverse of buy)
            quote = await self._get_swap_quote()
            if not quote:
                return None

            # Create and sign transaction
            tx = await self._create_swap_transaction()
            if not tx:
                return None

            # Send transaction
            result = await self.solana_client.send_transaction(
                tx,
                self.wallet,
                opts={
                    'skip_preflight': True,
                    'max_retries': 3,
                    'preflight_commitment': Confirmed
                }
            )

            if result.get('result'):
                return {
                    'signature': result['result'],
                    'price': float(quote.get('outAmount', 0)) / float(quote.get('inAmount', 1)),
                    'amount': amount
                }
            return None

        except Exception as e:
            logger.error(f"Execute sell order error: {e}")
            return None

    async def _get_price_data(self, token: str) -> Optional[Dict]:
        """Get token price data"""
        try:
            async with self.session.get(
                f"https://public-api.birdeye.so/public/price?address={token}",
                headers={'X-API-KEY': TRADING_CONFIG.BIRDEYE_API_KEY}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'current_price': float(data.get('value', 0)),
                        'prev_price': float(data.get('prevValue', 0))
                    }
                return None
        except Exception as e:
            logger.error(f"Get price data error: {e}")
            return None

    def _get_current_rpc(self) -> str:
        """Get current RPC endpoint with failover"""
        try:
            if time.time() - self.last_rpc_switch > TRADING_CONFIG.RPC_FAILOVER_TIMEOUT:
                self.current_rpc_index = (self.current_rpc_index + 1) % len(CORE_CONFIG.RPC_ENDPOINTS)
                self.last_rpc_switch = time.time()
            return CORE_CONFIG.RPC_ENDPOINTS[self.current_rpc_index]
        except Exception as e:
            logger.error(f"Get current RPC error: {e}")
            return CORE_CONFIG.RPC_ENDPOINTS[0]

    async def _execute_with_retry(self, func, *args, **kwargs) -> Optional[Dict]:
        """Execute function with retry logic"""
        for attempt in range(TRADING_CONFIG.MAX_RPC_RETRIES):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == TRADING_CONFIG.MAX_RPC_RETRIES - 1:
                    logger.error(f"Max retries reached: {e}")
                    return None
                await asyncio.sleep(TRADING_CONFIG.RETRY_INTERVAL * (attempt + 1))
                self.current_rpc_index = (self.current_rpc_index + 1) % len(CORE_CONFIG.RPC_ENDPOINTS)

    async def check_liquidity(self, token: str) -> bool:
        """Check token liquidity and lock status"""
        try:
            # Get liquidity info
            liquidity_info = await self._execute_with_retry(
                self._get_liquidity_info,
                token
            )
            if not liquidity_info:
                return False

            # Check minimum liquidity
            if liquidity_info['amount'] < TRADING_CONFIG.MIN_LIQUIDITY:
                return False

            # Check lock time
            if liquidity_info['lock_time'] < TRADING_CONFIG.MIN_LIQUIDITY_LOCK_TIME:
                return False

            # Check liquidity ratio
            liquidity_ratio = liquidity_info['amount'] / liquidity_info['market_cap']
            if liquidity_ratio < TRADING_CONFIG.MIN_LIQUIDITY_RATIO:
                return False

            return True

        except Exception as e:
            logger.error(f"Check liquidity error: {e}")
            return False

    async def check_rug_risk(self, token: str) -> bool:
        """Check for rug pull risk"""
        try:
            # Get token info
            token_info = await self._execute_with_retry(
                self._get_token_info,
                token
            )
            if not token_info:
                return True  # Assume risk if can't get info

            # Check holder count
            if token_info['holder_count'] < TRADING_CONFIG.MIN_HOLDER_COUNT:
                return True

            # Check owner concentration
            if token_info['owner_concentration'] > TRADING_CONFIG.MAX_OWNER_CONCENTRATION:
                return True

            # Check token age
            if token_info['age'] > TRADING_CONFIG.MAX_TOKEN_AGE:
                return True

            return False

        except Exception as e:
            logger.error(f"Check rug risk error: {e}")
            return True

    async def execute_buy(self, token: str, amount: float) -> Optional[Dict]:
        """Execute buy order with enhanced checks"""
        try:
            # Check liquidity
            if not await self.check_liquidity(token):
                logger.warning(f"Insufficient liquidity for {token}")
                return None

            # Check rug risk
            if await self.check_rug_risk(token):
                logger.warning(f"High rug risk for {token}")
                return None

            # Calculate price impact
            price_impact = await self._calculate_price_impact(token, amount)
            if price_impact > TRADING_CONFIG.MAX_PRICE_IMPACT:
                logger.warning(f"Price impact too high: {price_impact}")
                return None

            # Adjust priority fee based on network congestion
            current_fee = self.priority_fee * TRADING_CONFIG.PRIORITY_FEE_MULTIPLIER

            # Execute buy
            result = await self._execute_with_retry(
                self._execute_buy_order,
                token,
                amount,
                current_fee
            )

            if result:
                self.active_trades[token] = {
                    'amount': amount,
                    'price': result['price'],
                    'timestamp': time.time()
                }

            return result

        except Exception as e:
            logger.error(f"Execute buy error: {e}")
            return None

    async def execute_sell(self, token: str, amount: float) -> Optional[Dict]:
        """Execute sell order with enhanced checks"""
        try:
            # Check volatility
            if await self._check_volatility(token):
                logger.warning(f"High volatility for {token}")
                return None

            # Adjust priority fee based on network congestion
            current_fee = self.priority_fee * TRADING_CONFIG.PRIORITY_FEE_MULTIPLIER

            # Execute sell
            result = await self._execute_with_retry(
                self._execute_sell_order,
                token,
                amount,
                current_fee
            )

            if result:
                if token in self.active_trades:
                    del self.active_trades[token]

            return result

        except Exception as e:
            logger.error(f"Execute sell error: {e}")
            return None

    async def _check_volatility(self, token: str) -> bool:
        """Check if token price is too volatile"""
        try:
            price_data = await self._execute_with_retry(
                self._get_price_data,
                token
            )
            if not price_data:
                return True

            # Calculate price change
            price_change = abs(
                (price_data['current_price'] - price_data['prev_price'])
                / price_data['prev_price']
            )

            return price_change > TRADING_CONFIG.VOLATILITY_THRESHOLD

        except Exception as e:
            logger.error(f"Check volatility error: {e}")
            return True

    def get_available_balance(self) -> float:
        """Get available balance for trading"""
        try:
            # Calculate used capital
            used_capital = sum(
                trade['amount'] for trade in self.active_trades.values()
            )

            # Calculate available capital
            available = TRADING_CONFIG.STARTING_CAPITAL - used_capital

            # Apply capital efficiency threshold
            return available * TRADING_CONFIG.CAPITAL_EFFICIENCY_THRESHOLD

        except Exception as e:
            logger.error(f"Get available balance error: {e}")
            return 0.0

    def get_trade_pnl(self, token: str, current_price: float) -> float:
        """Calculate PnL for a trade"""
        try:
            trade = self.active_trades.get(token)
            if not trade:
                return 0.0

            return (current_price - trade['price']) / trade['price']

        except Exception as e:
            logger.error(f"Get trade PnL error: {e}")
            return 0.0 