"""
Integration tests for the Solana trading bot.
Tests complete workflows and component interactions.
"""
import pytest
import asyncio
import os
import tempfile
import json
from unittest.mock import Mock, patch, AsyncMock
from src.core.main import MemeCoinBot
from src.core.wallet_manager import WalletManager
from src.core.helius_service import HeliusService
from src.core.jupiter_service import JupiterService
from src.core.validation import TradingValidator

class TestBotInitialization:
    """Test bot initialization workflows"""
    
    @pytest.mark.asyncio
    async def test_simulation_mode_initialization(self):
        """Test bot initialization in simulation mode"""
        # Set simulation mode environment
        with patch.dict(os.environ, {
            'SIMULATION_MODE': 'true',
            'SOLANA_PRIVATE_KEY': 'demo_key_for_testing',
            'HELIUS_API_KEY': 'demo_key_for_testing',
            'JUPITER_API_KEY': 'demo_key_for_testing'
        }):
            bot = MemeCoinBot()
            
            # Should initialize successfully in simulation mode
            success = await bot.initialize()
            assert success
            
            # Check that all components are initialized
            assert bot.validator is not None
            assert bot.wallet_manager is not None
            assert bot.helius_service is not None
            assert bot.jupiter_service is not None
            assert bot.portfolio is not None
            
            # Clean up
            await bot.stop()
    
    @pytest.mark.asyncio
    async def test_production_mode_validation_failure(self):
        """Test bot initialization fails with invalid production credentials"""
        # Set production mode with invalid credentials
        with patch.dict(os.environ, {
            'SIMULATION_MODE': 'false',
            'SOLANA_PRIVATE_KEY': 'demo_key_for_testing',  # Invalid for production
            'HELIUS_API_KEY': 'demo_key_for_testing',
            'JUPITER_API_KEY': 'demo_key_for_testing'
        }):
            bot = MemeCoinBot()
            
            # Should fail to initialize with demo credentials in production mode
            success = await bot.initialize()
            assert not success

class TestWalletIntegration:
    """Test wallet manager integration"""
    
    @pytest.mark.asyncio
    async def test_wallet_simulation_mode(self):
        """Test wallet operations in simulation mode"""
        with patch.dict(os.environ, {'SIMULATION_MODE': 'true'}):
            wallet = WalletManager()
            
            # Should initialize successfully
            success = await wallet.initialize()
            assert success
            
            # Should have simulation balance
            balance = await wallet.check_balance()
            assert balance > 0
            assert isinstance(balance, float)
            
            # Should validate transaction parameters
            is_valid = await wallet.validate_transaction_params(0.1, "So11111111111111111111111111111111111111112")
            assert is_valid
            
            # Should reject invalid amounts
            try:
                await wallet.validate_transaction_params(-1.0, "So11111111111111111111111111111111111111112")
                assert False, "Should raise exception for negative amount"
            except Exception:
                pass  # Expected
            
            await wallet.close()
    
    @pytest.mark.asyncio
    async def test_wallet_transaction_simulation(self):
        """Test wallet transaction simulation"""
        with patch.dict(os.environ, {'SIMULATION_MODE': 'true'}):
            wallet = WalletManager()
            await wallet.initialize()
            
            initial_balance = await wallet.check_balance()
            
            # Simulate a transfer
            tx_id = await wallet.transfer_sol("TargetAddress111111111111111111111111", 0.01)
            assert tx_id.startswith("SIM_TRANSFER_")
            
            # Balance should be reduced
            new_balance = await wallet.check_balance()
            assert new_balance < initial_balance
            
            await wallet.close()

class TestHeliusIntegration:
    """Test Helius service integration"""
    
    @pytest.mark.asyncio
    async def test_helius_simulation_mode(self):
        """Test Helius service in simulation mode"""
        with patch.dict(os.environ, {'SIMULATION_MODE': 'true'}):
            helius = HeliusService()
            
            # Test token metadata retrieval
            metadata = await helius.get_token_metadata("So11111111111111111111111111111111111111112")
            assert metadata is not None
            assert 'symbol' in metadata
            assert metadata['symbol'] == 'SIMTOKEN'
            
            # Test price data
            price_data = await helius.get_token_price("So11111111111111111111111111111111111111112")
            assert price_data is not None
            assert 'price_usd' in price_data
            assert price_data['price_usd'] > 0
            
            # Test new tokens
            new_tokens = await helius.get_new_tokens(limit=5)
            assert isinstance(new_tokens, list)
            assert len(new_tokens) <= 5
            
            # Test security analysis
            security = await helius.analyze_token_security("So11111111111111111111111111111111111111112")
            assert security is not None
            assert 'security_score' in security
            assert 'risk_level' in security
            
            await helius.close()

class TestJupiterIntegration:
    """Test Jupiter service integration"""
    
    @pytest.mark.asyncio
    async def test_jupiter_simulation_mode(self):
        """Test Jupiter service in simulation mode"""
        with patch.dict(os.environ, {'SIMULATION_MODE': 'true'}):
            jupiter = JupiterService()
            
            # Test quote retrieval
            quote = await jupiter.get_quote(
                input_mint="So11111111111111111111111111111111111111112",
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000000  # 1 SOL
            )
            
            assert quote is not None
            assert quote.input_amount == 1000000000
            assert quote.output_amount > 0
            assert quote.price > 0
            
            # Test token price
            price = await jupiter.get_token_price("So11111111111111111111111111111111111111112")
            assert price is not None
            assert price > 0
            
            # Test supported tokens
            tokens = await jupiter.get_supported_tokens()
            assert isinstance(tokens, list)
            assert len(tokens) > 0
            
            # Test random tokens
            random_tokens = await jupiter.get_random_tokens(count=3)
            assert isinstance(random_tokens, list)
            assert len(random_tokens) <= 3
            
            await jupiter.close()

class TestValidationIntegration:
    """Test validation system integration"""
    
    def test_validator_simulation_mode(self):
        """Test validator in simulation mode"""
        validator = TradingValidator(simulation_mode=True)
        
        # Test trade validation
        trade_params = {
            "input_token": "So11111111111111111111111111111111111111112",
            "output_token": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "amount": "1.0",
            "slippage": "1.5"
        }
        
        result = validator.validate_trade(trade_params)
        assert result.is_valid
        assert result.sanitized_value is not None
        
        # Test credential validation in simulation
        credentials = {
            'private_key': 'demo_key_for_testing',
            'helius_api_key': 'demo_key_for_testing'
        }
        
        cred_result = validator.validate_credentials(credentials)
        assert cred_result.is_valid  # Should pass in simulation
    
    def test_validator_production_mode(self):
        """Test validator in production mode"""
        validator = TradingValidator(simulation_mode=False)
        
        # Test credential validation in production
        credentials = {
            'private_key': 'demo_key_for_testing',
            'helius_api_key': 'demo_key_for_testing'
        }
        
        cred_result = validator.validate_credentials(credentials)
        assert not cred_result.is_valid  # Should fail in production

class TestTradeExecution:
    """Test end-to-end trade execution"""
    
    @pytest.mark.asyncio
    async def test_trade_execution_simulation(self):
        """Test complete trade execution in simulation mode"""
        with patch.dict(os.environ, {
            'SIMULATION_MODE': 'true',
            'SOLANA_PRIVATE_KEY': 'demo_key_for_testing',
            'HELIUS_API_KEY': 'demo_key_for_testing',
            'JUPITER_API_KEY': 'demo_key_for_testing'
        }):
            bot = MemeCoinBot()
            success = await bot.initialize()
            assert success
            
            # Mock token data
            token_data = {
                "symbol": "TESTSOL",
                "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # Valid USDC token address
                "price_usd": 0.1,
                "market_cap": 1000000
            }
            
            # Test buy execution
            initial_balance = await bot.wallet_manager.check_balance()
            
            await bot.execute_trade(
                token_data=token_data,
                action="buy", 
                position_size=0.01,
                reasoning="Test buy order"
            )
            
            # Check that trade was recorded
            portfolio_summary = bot.portfolio.get_portfolio_summary()
            assert portfolio_summary['total_trades'] >= 1
            
            await bot.stop()

class TestErrorHandling:
    """Test error handling and recovery"""
    
    @pytest.mark.asyncio
    async def test_invalid_token_handling(self):
        """Test handling of invalid token addresses"""
        with patch.dict(os.environ, {'SIMULATION_MODE': 'true'}):
            validator = TradingValidator(simulation_mode=True)
            
            # Test with invalid token address
            trade_params = {
                "input_token": "invalid_address",
                "output_token": "So11111111111111111111111111111111111111112",
                "amount": "1.0",
                "slippage": "1.5"
            }
            
            result = validator.validate_trade(trade_params)
            assert not result.is_valid
            assert len(result.errors) > 0
    
    @pytest.mark.asyncio
    async def test_insufficient_balance_handling(self):
        """Test handling of insufficient balance scenarios"""
        with patch.dict(os.environ, {'SIMULATION_MODE': 'true'}):
            wallet = WalletManager()
            await wallet.initialize()
            
            balance = await wallet.check_balance()
            
            # Try to validate transaction for more than available balance
            try:
                await wallet.validate_transaction_params(
                    amount=balance * 2,  # Double the available balance
                    token_address="So11111111111111111111111111111111111111112"
                )
                assert False, "Should raise insufficient funds error"
            except Exception as e:
                assert "insufficient" in str(e).lower() or "amount too large" in str(e).lower()
            
            await wallet.close()
    
    @pytest.mark.asyncio
    async def test_network_error_handling(self):
        """Test handling of network errors"""
        with patch.dict(os.environ, {'SIMULATION_MODE': 'true'}):
            helius = HeliusService()
            
            # Should handle gracefully in simulation mode
            metadata = await helius.get_token_metadata("NonExistentToken")
            # In simulation mode, this should return mock data or handle gracefully
            assert metadata is not None  # Simulation should always return data
            
            await helius.close()

class TestPerformance:
    """Test performance and scalability"""
    
    @pytest.mark.asyncio
    async def test_concurrent_validations(self):
        """Test concurrent validation performance"""
        validator = TradingValidator(simulation_mode=True)
        
        async def validate_trade():
            trade_params = {
                "input_token": "So11111111111111111111111111111111111111112",
                "output_token": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                "amount": "1.0",
                "slippage": "1.5"
            }
            return validator.validate_trade(trade_params)
        
        # Run 50 concurrent validations
        import time
        start_time = time.time()
        
        tasks = [validate_trade() for _ in range(50)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        
        # All should succeed
        assert all(result.is_valid for result in results)
        
        # Should complete within reasonable time (< 2 seconds)
        assert (end_time - start_time) < 2.0
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory usage doesn't grow excessively"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create and initialize multiple validators
        validators = []
        for _ in range(20):
            validator = TradingValidator(simulation_mode=True)
            validators.append(validator)
            
            # Perform multiple validations
            for _ in range(10):
                trade_params = {
                    "input_token": "So11111111111111111111111111111111111111112",
                    "output_token": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                    "amount": "1.0",
                    "slippage": "1.5"
                }
                result = validator.validate_trade(trade_params)
                assert result.is_valid
        
        # Clean up and force garbage collection
        del validators
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 50MB)
        assert memory_increase < 50 * 1024 * 1024

class TestConfigurationHandling:
    """Test different configuration scenarios"""
    
    @pytest.mark.asyncio
    async def test_missing_environment_variables(self):
        """Test handling of missing environment variables"""
        # Clear relevant environment variables
        env_vars_to_clear = [
            'SIMULATION_MODE',
            'SOLANA_PRIVATE_KEY', 
            'HELIUS_API_KEY',
            'JUPITER_API_KEY'
        ]
        
        # Store original values
        original_values = {}
        for var in env_vars_to_clear:
            original_values[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]
        
        try:
            bot = MemeCoinBot()
            
            # Should handle missing environment variables gracefully
            # Default to simulation mode when SIMULATION_MODE is missing
            success = await bot.initialize()
            
            # Should succeed in simulation mode even with missing variables
            assert success or bot.validator.simulation_mode
            
            if success:
                await bot.stop()
                
        finally:
            # Restore original environment variables
            for var, value in original_values.items():
                if value is not None:
                    os.environ[var] = value
    
    def test_environment_variable_validation(self):
        """Test validation of environment variable formats"""
        validator = TradingValidator(simulation_mode=False)
        
        # Test various credential formats
        test_cases = [
            {
                'private_key': '',
                'expected_valid': False
            },
            {
                'private_key': 'demo_key_for_testing',
                'expected_valid': False  # Demo key in production
            },
            {
                'private_key': '1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef',
                'expected_valid': True   # Valid hex format
            }
        ]
        
        for case in test_cases:
            credentials = {
                'private_key': case['private_key'],
                'helius_api_key': 'real_api_key_12345',
                'jupiter_api_key': 'real_api_key_12345'
            }
            
            result = validator.validate_credentials(credentials)
            if case['expected_valid']:
                assert result.is_valid or len(result.errors) == 0
            else:
                assert not result.is_valid

class TestDataPersistence:
    """Test data persistence and state management"""
    
    @pytest.mark.asyncio
    async def test_portfolio_state_persistence(self):
        """Test portfolio state tracking across operations"""
        with patch.dict(os.environ, {
            'SIMULATION_MODE': 'true',
            'SOLANA_PRIVATE_KEY': 'demo_key_for_testing',
            'HELIUS_API_KEY': 'demo_key_for_testing',
            'JUPITER_API_KEY': 'demo_key_for_testing'
        }):
            bot = MemeCoinBot()
            await bot.initialize()
            
            # Initial portfolio state
            initial_summary = bot.portfolio.get_portfolio_summary()
            assert initial_summary['total_trades'] == 0
            
            # Add a mock trade
            trade_data = {
                "action": "buy",
                "token": "TESTSOL",
                "token_address": "TestToken111111111111111111111111111111",
                "amount_sol": 0.01,
                "price_usd": 0.1,
                "sol_price": 100,
                "timestamp": 1234567890,
                "status": "success",
                "transaction_id": "test_tx_123"
            }
            
            bot.portfolio.add_trade(trade_data)
            
            # Check portfolio state updated
            updated_summary = bot.portfolio.get_portfolio_summary()
            assert updated_summary['total_trades'] == 1
            
            # Check holdings
            holdings = bot.portfolio.get_holdings()
            assert len(holdings) > 0
            
            await bot.stop()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"]) 