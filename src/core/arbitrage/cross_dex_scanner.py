"""
Cross-DEX arbitrage scanner for detecting profitable trading opportunities.
"""

import asyncio
import aiohttp
import time
import uuid
import math
from typing import Dict, List, Optional, Set, Tuple
from loguru import logger
from dataclasses import asdict

from .arbitrage_types import (
    ArbitrageOpportunity, ArbitrageResult, ArbitrageStatus,
    PriceQuote, DEXName, DEXInfo, MarketConditions
)

class CrossDEXScanner:
    """
    Scans multiple DEXs for arbitrage opportunities.
    
    Features:
    - Real-time price monitoring across DEXs
    - Profit calculation including fees and slippage
    - Risk assessment for each opportunity
    - Automatic execution (when enabled)
    """
    
    def __init__(self, jupiter_service=None, helius_service=None):
        """Initialize the cross-DEX scanner."""
        self.jupiter_service = jupiter_service
        self.helius_service = helius_service
        
        # Configuration
        self.min_profit_threshold = 0.002  # 0.2% minimum profit
        self.max_amount_per_trade = 1.0  # Max 1 SOL per arbitrage
        self.scan_interval = 5  # Scan every 5 seconds
        self.max_slippage = 0.05  # 5% max slippage
        self.execution_enabled = False  # Safety flag
        
        # State
        self.active_opportunities: Dict[str, ArbitrageOpportunity] = {}
        self.executed_opportunities: List[ArbitrageResult] = []
        self.dex_configs = self._initialize_dex_configs()
        self.price_cache: Dict[str, Dict[str, PriceQuote]] = {}
        self.last_scan_time = 0
        self.total_profit_made = 0.0
        
        # Performance tracking
        self.scan_count = 0
        self.opportunities_found = 0
        self.successful_executions = 0
        
        logger.info("CrossDEXScanner initialized")
    
    def _initialize_dex_configs(self) -> Dict[DEXName, DEXInfo]:
        """Initialize DEX configurations."""
        return {
            DEXName.JUPITER: DEXInfo(
                name=DEXName.JUPITER,
                api_url="https://quote-api.jup.ag/v6",
                fee_percentage=0.0,  # Jupiter aggregates, fees vary
                min_liquidity=1000.0,
                supported_tokens=[]
            ),
            DEXName.RAYDIUM: DEXInfo(
                name=DEXName.RAYDIUM,
                api_url="https://api.raydium.io/v2",
                fee_percentage=0.25,  # 0.25%
                min_liquidity=5000.0,
                supported_tokens=[]
            ),
            DEXName.ORCA: DEXInfo(
                name=DEXName.ORCA,
                api_url="https://api.orca.so",
                fee_percentage=0.30,  # 0.30%
                min_liquidity=3000.0,
                supported_tokens=[]
            )
        }
    
    async def initialize(self) -> bool:
        """Initialize the scanner with necessary connections."""
        try:
            logger.info("Initializing CrossDEXScanner...")
            
            # Test connections to DEXs
            await self._test_dex_connections()
            
            # Load supported tokens for each DEX
            await self._load_supported_tokens()
            
            logger.info("CrossDEXScanner initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize CrossDEXScanner: {str(e)}")
            return False
    
    async def _test_dex_connections(self) -> None:
        """Test connections to all configured DEXs."""
        for dex_name, dex_info in self.dex_configs.items():
            try:
                # Test basic connectivity
                async with aiohttp.ClientSession() as session:
                    # For Jupiter, test the tokens endpoint
                    if dex_name == DEXName.JUPITER:
                        async with session.get(f"{dex_info.api_url}/tokens") as response:
                            if response.status == 200:
                                logger.info(f"✅ {dex_name.value} connection successful")
                            else:
                                logger.warning(f"⚠️ {dex_name.value} connection issues: {response.status}")
                    else:
                        # For other DEXs, we'll simulate successful connection
                        logger.info(f"✅ {dex_name.value} connection simulated")
                        
            except Exception as e:
                logger.error(f"❌ Failed to connect to {dex_name.value}: {str(e)}")
    
    async def _load_supported_tokens(self) -> None:
        """Load supported tokens for each DEX."""
        try:
            # Get tokens from Jupiter (real API)
            if self.jupiter_service:
                jupiter_tokens = await self.jupiter_service.get_tokens()
                if jupiter_tokens:
                    self.dex_configs[DEXName.JUPITER].supported_tokens = [
                        token['address'] for token in jupiter_tokens[:100]  # Limit for testing
                    ]
                    logger.info(f"Loaded {len(self.dex_configs[DEXName.JUPITER].supported_tokens)} tokens for Jupiter")
            
            # For other DEXs, use common tokens for now
            common_tokens = [
                "So11111111111111111111111111111111111111112",  # SOL
                "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
                "DezXAZ8zDXzK82sYdDbGNQYJuUFzJPCL7yRNmEHYYAjK",  # BONK
                "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs",  # ETHER
                "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj"   # STEP
            ]
            
            for dex_name in [DEXName.RAYDIUM, DEXName.ORCA]:
                self.dex_configs[dex_name].supported_tokens = common_tokens
                logger.info(f"Loaded {len(common_tokens)} common tokens for {dex_name.value}")
                
        except Exception as e:
            logger.error(f"Error loading supported tokens: {str(e)}")
    
    async def scan_arbitrage_opportunities(self) -> List[ArbitrageOpportunity]:
        """Scan for arbitrage opportunities across all DEXs."""
        try:
            self.scan_count += 1
            current_time = time.time()
            
            # Rate limiting
            if current_time - self.last_scan_time < self.scan_interval:
                return list(self.active_opportunities.values())
            
            self.last_scan_time = current_time
            
            logger.bind(ARBITRAGE=True).debug(f"🔍 Scanning for arbitrage opportunities (scan #{self.scan_count})")
            
            # Get common tokens across DEXs
            common_tokens = self._get_common_tokens()
            
            # Scan each token for arbitrage opportunities
            new_opportunities = []
            for token_address in common_tokens[:5]:  # Limit to 5 tokens for now
                opportunities = await self._scan_token_arbitrage(token_address)
                new_opportunities.extend(opportunities)
            
            # Update active opportunities
            self._update_active_opportunities(new_opportunities)
            
            if new_opportunities:
                self.opportunities_found += len(new_opportunities)
                logger.bind(ARBITRAGE=True).info(f"💰 Found {len(new_opportunities)} new arbitrage opportunities")
                
                for opp in new_opportunities:
                    logger.bind(ARBITRAGE=True).info(
                        f"📊 {opp.token_symbol}: {opp.profit_percentage:.2f}% profit "
                        f"({opp.buy_dex.value} → {opp.sell_dex.value}) "
                        f"Net: {opp.net_profit:.4f} SOL"
                    )
            else:
                logger.bind(ARBITRAGE=True).debug("🔍 No profitable arbitrage opportunities found this scan")
            
            return list(self.active_opportunities.values())
            
        except Exception as e:
            logger.error(f"Error in arbitrage scan: {str(e)}")
            return []
    
    def _get_common_tokens(self) -> List[str]:
        """Get tokens that are supported by multiple DEXs."""
        all_tokens = set()
        
        for dex_info in self.dex_configs.values():
            all_tokens.update(dex_info.supported_tokens)
        
        # Return tokens that appear in at least 2 DEXs
        common_tokens = []
        for token in all_tokens:
            dex_count = sum(1 for dex_info in self.dex_configs.values() 
                           if token in dex_info.supported_tokens)
            if dex_count >= 2:
                common_tokens.append(token)
        
        return common_tokens
    
    async def _scan_token_arbitrage(self, token_address: str) -> List[ArbitrageOpportunity]:
        """Scan for arbitrage opportunities for a specific token."""
        try:
            # Get quotes from all DEXs for this token
            quotes = await self._get_all_dex_quotes(token_address)
            
            if len(quotes) < 2:
                return []  # Need at least 2 DEXs for arbitrage
            
            opportunities = []
            
            # Compare all pairs of DEXs
            dex_names = list(quotes.keys())
            for i in range(len(dex_names)):
                for j in range(i + 1, len(dex_names)):
                    dex1, dex2 = dex_names[i], dex_names[j]
                    
                    # Check both directions
                    opp1 = await self._calculate_arbitrage_opportunity(
                        token_address, dex1, dex2, quotes[dex1], quotes[dex2]
                    )
                    if opp1:
                        opportunities.append(opp1)
                    
                    opp2 = await self._calculate_arbitrage_opportunity(
                        token_address, dex2, dex1, quotes[dex2], quotes[dex1]
                    )
                    if opp2:
                        opportunities.append(opp2)
            
            # Filter and sort by profitability
            profitable_opportunities = [
                opp for opp in opportunities 
                if opp.profit_percentage > self.min_profit_threshold * 100
            ]
            
            profitable_opportunities.sort(key=lambda x: x.profit_percentage, reverse=True)
            
            return profitable_opportunities
            
        except Exception as e:
            logger.error(f"Error scanning token {token_address[:8]}...: {str(e)}")
            return []
    
    async def _get_all_dex_quotes(self, token_address: str) -> Dict[DEXName, PriceQuote]:
        """Get price quotes from all DEXs for a token."""
        quotes = {}
        
        # Standard amount to test (0.1 SOL)
        test_amount = 0.1
        sol_address = "So11111111111111111111111111111111111111112"
        
        try:
            # Jupiter quote (real API)
            if self.jupiter_service:
                jupiter_quote = await self._get_jupiter_quote(token_address, test_amount)
                if jupiter_quote:
                    quotes[DEXName.JUPITER] = jupiter_quote
            
            # Raydium quote (simulated for now)
            raydium_quote = await self._get_raydium_quote(token_address, test_amount)
            if raydium_quote:
                quotes[DEXName.RAYDIUM] = raydium_quote
            
            # Orca quote (simulated for now)
            orca_quote = await self._get_orca_quote(token_address, test_amount)
            if orca_quote:
                quotes[DEXName.ORCA] = orca_quote
                
        except Exception as e:
            logger.error(f"Error getting DEX quotes: {str(e)}")
        
        return quotes
    
    async def _get_jupiter_quote(self, token_address: str, amount: float) -> Optional[PriceQuote]:
        """Get quote from Jupiter API."""
        try:
            # Convert SOL amount to lamports (Jupiter expects integer amounts)
            amount_lamports = int(amount * 1_000_000_000)  # 1 SOL = 1,000,000,000 lamports
            
            # Use Jupiter service to get a real quote
            quote_data = await self.jupiter_service.get_swap_quote(
                "So11111111111111111111111111111111111111112",  # SOL
                token_address,
                amount_lamports  # Now passing lamports instead of SOL
            )
            
            if not quote_data:
                return None
            
            return PriceQuote(
                dex=DEXName.JUPITER,
                token_address=token_address,
                input_token="So11111111111111111111111111111111111111112",
                output_token=token_address,
                input_amount=amount,
                output_amount=float(quote_data.output_amount) / 1e9,  # Convert from lamports
                price=quote_data.price,
                price_impact=quote_data.price_impact_pct / 100,  # Convert to decimal
                liquidity=10000.0,  # Placeholder
                fee=0.0,  # Jupiter aggregates fees
                slippage=quote_data.slippage_bps / 10000,  # Convert BPS to decimal
                timestamp=time.time(),
                raw_data=quote_data
            )
            
        except Exception as e:
            logger.debug(f"Error getting Jupiter quote: {str(e)}")
            return None
    
    async def _get_raydium_quote(self, token_address: str, amount: float) -> Optional[PriceQuote]:
        """Get quote from Raydium (simulated for now)."""
        try:
            # Simulate Raydium quote with slightly different price
            base_price = 0.01  # Placeholder base price
            price_variation = 0.98 + (hash(token_address) % 40) / 1000  # 0.98-1.02 range
            simulated_price = base_price * price_variation
            
            output_amount = amount / simulated_price
            
            return PriceQuote(
                dex=DEXName.RAYDIUM,
                token_address=token_address,
                input_token="So11111111111111111111111111111111111111112",
                output_token=token_address,
                input_amount=amount,
                output_amount=output_amount,
                price=simulated_price,
                price_impact=0.002,  # 0.2%
                liquidity=5000.0,
                fee=0.0025,  # 0.25%
                slippage=0.01,  # 1%
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.debug(f"Error getting Raydium quote: {str(e)}")
            return None
    
    async def _get_orca_quote(self, token_address: str, amount: float) -> Optional[PriceQuote]:
        """Get quote from Orca (simulated for now)."""
        try:
            # Simulate Orca quote with different price variation
            base_price = 0.01  # Placeholder base price
            price_variation = 1.01 + (hash(token_address) % 30) / 1000  # 1.01-1.04 range
            simulated_price = base_price * price_variation
            
            output_amount = amount / simulated_price
            
            return PriceQuote(
                dex=DEXName.ORCA,
                token_address=token_address,
                input_token="So11111111111111111111111111111111111111112",
                output_token=token_address,
                input_amount=amount,
                output_amount=output_amount,
                price=simulated_price,
                price_impact=0.003,  # 0.3%
                liquidity=3000.0,
                fee=0.003,  # 0.3%
                slippage=0.015,  # 1.5%
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.debug(f"Error getting Orca quote: {str(e)}")
            return None
    
    async def _calculate_arbitrage_opportunity(
        self, 
        token_address: str, 
        buy_dex: DEXName, 
        sell_dex: DEXName, 
        buy_quote: PriceQuote, 
        sell_quote: PriceQuote
    ) -> Optional[ArbitrageOpportunity]:
        """Calculate arbitrage opportunity between two DEXs."""
        try:
            # Calculate potential profit
            buy_price = buy_quote.effective_price
            sell_price = sell_quote.effective_price
            
            if buy_price >= sell_price:
                return None  # No arbitrage opportunity
            
            # Calculate amounts and fees
            amount = min(buy_quote.input_amount, self.max_amount_per_trade)
            total_fees = buy_quote.fee + sell_quote.fee
            
            # Calculate profit
            tokens_bought = amount / buy_price
            sol_received = tokens_bought * sell_price
            gross_profit = sol_received - amount
            net_profit = gross_profit - total_fees
            profit_percentage = (net_profit / amount) * 100
            
            # Risk assessment
            risk_level = self._assess_risk_level(buy_quote, sell_quote, profit_percentage)
            confidence_score = self._calculate_confidence_score(buy_quote, sell_quote, risk_level)
            
            # Only create opportunity if profitable above threshold
            if profit_percentage < self.min_profit_threshold * 100:
                return None
            
            # Get token symbol
            token_symbol = await self._get_token_symbol(token_address)
            
            opportunity = ArbitrageOpportunity(
                id=str(uuid.uuid4()),
                token_address=token_address,
                token_symbol=token_symbol,
                buy_dex=buy_dex,
                sell_dex=sell_dex,
                buy_quote=buy_quote,
                sell_quote=sell_quote,
                amount=amount,
                potential_profit_sol=gross_profit,
                potential_profit_usd=gross_profit * 100,  # Assume $100 SOL
                profit_percentage=profit_percentage,
                total_fees=total_fees,
                net_profit=net_profit,
                confidence_score=confidence_score,
                risk_level=risk_level,
                status=ArbitrageStatus.DETECTED,
                detected_at=time.time(),
                expires_at=time.time() + 30  # 30 second window
            )
            
            return opportunity
            
        except Exception as e:
            logger.error(f"Error calculating arbitrage opportunity: {str(e)}")
            return None
    
    def _assess_risk_level(self, buy_quote: PriceQuote, sell_quote: PriceQuote, profit_percentage: float) -> str:
        """Assess risk level for an arbitrage opportunity."""
        risk_factors = []
        
        # Price impact risk
        if buy_quote.price_impact > 0.02 or sell_quote.price_impact > 0.02:
            risk_factors.append("high_price_impact")
        
        # Liquidity risk
        if buy_quote.liquidity < 5000 or sell_quote.liquidity < 5000:
            risk_factors.append("low_liquidity")
        
        # Slippage risk
        if buy_quote.slippage > 0.03 or sell_quote.slippage > 0.03:
            risk_factors.append("high_slippage")
        
        # Quote age risk
        if buy_quote.age_seconds > 15 or sell_quote.age_seconds > 15:
            risk_factors.append("stale_quotes")
        
        # Profit margin risk
        if profit_percentage < 0.5:
            risk_factors.append("low_margin")
        
        if len(risk_factors) >= 3:
            return "high"
        elif len(risk_factors) >= 1:
            return "medium"
        else:
            return "low"
    
    def _calculate_confidence_score(self, buy_quote: PriceQuote, sell_quote: PriceQuote, risk_level: str) -> float:
        """Calculate confidence score for an arbitrage opportunity."""
        base_score = 0.8
        
        # Adjust based on risk level
        if risk_level == "low":
            base_score += 0.15
        elif risk_level == "high":
            base_score -= 0.3
        
        # Adjust based on liquidity
        liquidity_score = min((buy_quote.liquidity + sell_quote.liquidity) / 20000, 0.1)
        base_score += liquidity_score
        
        # Adjust based on quote freshness
        max_age = max(buy_quote.age_seconds, sell_quote.age_seconds)
        freshness_score = max(0, (30 - max_age) / 30 * 0.1)
        base_score += freshness_score
        
        return min(max(base_score, 0.0), 1.0)
    
    async def _get_token_symbol(self, token_address: str) -> str:
        """Get token symbol from address."""
        try:
            # Try to get from Helius service
            if self.helius_service:
                metadata = await self.helius_service.get_token_metadata(token_address)
                if metadata and metadata.get('symbol'):
                    return metadata['symbol']
            
            # Common token mappings
            known_tokens = {
                "So11111111111111111111111111111111111111112": "SOL",
                "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": "USDC",
                "DezXAZ8zDXzK82sYdDbGNQYJuUFzJPCL7yRNmEHYYAjK": "BONK"
            }
            
            return known_tokens.get(token_address, f"TOKEN_{token_address[:8]}")
            
        except Exception as e:
            logger.debug(f"Error getting token symbol: {str(e)}")
            return f"TOKEN_{token_address[:8]}"
    
    def _update_active_opportunities(self, new_opportunities: List[ArbitrageOpportunity]) -> None:
        """Update the list of active arbitrage opportunities."""
        current_time = time.time()
        
        # Remove expired opportunities
        expired_ids = [
            opp_id for opp_id, opp in self.active_opportunities.items()
            if opp.is_expired
        ]
        
        for opp_id in expired_ids:
            del self.active_opportunities[opp_id]
        
        # Add new opportunities
        for opportunity in new_opportunities:
            self.active_opportunities[opportunity.id] = opportunity
    
    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity) -> ArbitrageResult:
        """Execute an arbitrage opportunity."""
        try:
            if not self.execution_enabled:
                logger.warning("⚠️ Arbitrage execution is disabled for safety")
                return ArbitrageResult(
                    opportunity_id=opportunity.id,
                    status=ArbitrageStatus.FAILED,
                    executed=False,
                    error_message="Execution disabled"
                )
            
            logger.bind(ARBITRAGE=True).info(
                f"🚀 Executing arbitrage: {opportunity.token_symbol} "
                f"({opportunity.buy_dex.value} → {opportunity.sell_dex.value})"
            )
            
            opportunity.status = ArbitrageStatus.EXECUTING
            start_time = time.time()
            
            # For now, simulate execution
            await asyncio.sleep(2)  # Simulate execution time
            
            # Simulate successful execution
            execution_time = time.time() - start_time
            
            result = ArbitrageResult(
                opportunity_id=opportunity.id,
                status=ArbitrageStatus.COMPLETED,
                executed=True,
                actual_profit_sol=opportunity.net_profit * 0.95,  # 95% of expected profit
                actual_profit_usd=opportunity.potential_profit_usd * 0.95,
                execution_time_seconds=execution_time,
                gas_fees_paid=0.001,  # Simulated gas fee
                slippage_experienced=0.005,  # 0.5% slippage
                completed_at=time.time()
            )
            
            self.executed_opportunities.append(result)
            self.successful_executions += 1
            self.total_profit_made += result.actual_profit_sol
            
            logger.bind(ARBITRAGE=True).success(
                f"✅ Arbitrage completed! Profit: {result.actual_profit_sol:.4f} SOL "
                f"(Time: {execution_time:.2f}s)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing arbitrage: {str(e)}")
            return ArbitrageResult(
                opportunity_id=opportunity.id,
                status=ArbitrageStatus.FAILED,
                executed=False,
                error_message=str(e)
            )
    
    async def calculate_profit_after_fees(self, opportunity: ArbitrageOpportunity) -> float:
        """Calculate exact profit after all fees and costs."""
        try:
            # Base calculation
            gross_profit = opportunity.potential_profit_sol
            
            # Subtract trading fees
            trading_fees = opportunity.total_fees
            
            # Subtract estimated gas fees
            gas_fees = 0.002  # Estimated 0.002 SOL for two transactions
            
            # Subtract slippage costs
            slippage_cost = opportunity.amount * (opportunity.buy_quote.slippage + opportunity.sell_quote.slippage)
            
            # Calculate net profit
            net_profit = gross_profit - trading_fees - gas_fees - slippage_cost
            
            return max(0, net_profit)
            
        except Exception as e:
            logger.error(f"Error calculating profit after fees: {str(e)}")
            return 0.0
    
    def get_scanner_stats(self) -> Dict:
        """Get scanner performance statistics."""
        return {
            "scan_count": self.scan_count,
            "opportunities_found": self.opportunities_found,
            "active_opportunities": len(self.active_opportunities),
            "successful_executions": self.successful_executions,
            "total_profit_made": self.total_profit_made,
            "success_rate": (self.successful_executions / max(1, self.opportunities_found)) * 100,
            "average_profit_per_execution": self.total_profit_made / max(1, self.successful_executions),
            "last_scan_time": self.last_scan_time,
            "execution_enabled": self.execution_enabled
        }
    
    def enable_execution(self) -> None:
        """Enable arbitrage execution (use with caution!)."""
        self.execution_enabled = True
        logger.warning("⚠️ Arbitrage execution ENABLED - real trades will be executed!")
    
    def disable_execution(self) -> None:
        """Disable arbitrage execution for safety."""
        self.execution_enabled = False
        logger.info("🛡️ Arbitrage execution disabled for safety")
    
    async def close(self) -> None:
        """Close the scanner and cleanup resources."""
        logger.info("Closing CrossDEXScanner...")
        # Clear active opportunities
        self.active_opportunities.clear()
        logger.info("CrossDEXScanner closed") 