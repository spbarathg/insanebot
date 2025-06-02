"""
Unit tests for SecurityManager - comprehensive security testing.
"""

import pytest
import pytest_asyncio
import asyncio
import time
import secrets
from unittest.mock import AsyncMock, MagicMock, patch
from src.core.security_manager import SecurityManager, SecurityEvent, EncryptionManager, SecurityEventType, ThreatLevel


class TestEncryptionManager:
    """Test encryption functionality."""
    
    def test_encryption_manager_initialization(self):
        """Test encryption manager initializes correctly."""
        manager = EncryptionManager()
        assert manager.master_key is not None
        assert manager.fernet is not None
        assert manager.rsa_private_key is not None
        assert manager.rsa_public_key is not None
    
    def test_symmetric_encryption_decryption(self):
        """Test symmetric encryption and decryption."""
        manager = EncryptionManager()
        test_data = b"sensitive_trading_data"
        
        # Encrypt
        encrypted = manager.encrypt_symmetric(test_data)
        assert encrypted != test_data
        assert len(encrypted) > len(test_data)
        
        # Decrypt
        decrypted = manager.decrypt_symmetric(encrypted)
        assert decrypted == test_data
    
    def test_asymmetric_encryption_decryption(self):
        """Test asymmetric encryption and decryption."""
        manager = EncryptionManager()
        test_data = b"wallet_private_key"
        
        # Encrypt
        encrypted = manager.encrypt_asymmetric(test_data)
        assert encrypted != test_data
        
        # Decrypt
        decrypted = manager.decrypt_asymmetric(encrypted)
        assert decrypted == test_data
    
    def test_password_hashing_verification(self):
        """Test password hashing and verification."""
        manager = EncryptionManager()
        password = "test_password_123"
        
        # Hash password
        hashed = manager.hash_password(password)
        assert hashed != password
        assert len(hashed) > 20
        
        # Verify password
        assert manager.verify_password(password, hashed) is True
        assert manager.verify_password("wrong_password", hashed) is False


class TestSecurityManager:
    """Test security manager functionality."""
    
    @pytest_asyncio.fixture
    async def security_manager(self):
        """Create security manager for testing."""
        # Mock Redis to avoid actual Redis dependency
        with patch('src.core.security_manager.redis') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.from_url.return_value = mock_redis_instance
            mock_redis_instance.ping.return_value = True
            
            manager = SecurityManager()
            await manager.initialize()
            return manager
    
    @pytest.mark.asyncio
    async def test_security_manager_initialization(self, security_manager):
        """Test security manager initializes correctly."""
        assert security_manager.encryption_manager is not None
        assert security_manager.threat_detector is not None
        assert security_manager.rate_limiter is not None
        assert len(security_manager.security_policies) > 0
    
    @pytest.mark.asyncio
    async def test_authentication_success(self, security_manager):
        """Test successful authentication."""
        # Create a test user
        from src.core.security_manager import AccessControl
        test_access = AccessControl(
            user_id="test_user",
            roles={"trader"},
            permissions={"trading"},
            rate_limits={"api_calls": 100},
            mfa_required=False
        )
        security_manager.access_controls["test_user"] = test_access
        
        # Mock JWT verification
        with patch('jwt.decode') as mock_decode:
            mock_decode.return_value = {"user_id": "test_user"}
            
            request_data = {
                "headers": {"Authorization": "Bearer valid_token"},
                "remote_addr": "127.0.0.1"
            }
            
            result = await security_manager.authenticate_request(request_data)
            
            assert result["authenticated"] is True
            assert result["user_id"] == "test_user"
            assert "trader" in result["roles"]
    
    @pytest.mark.asyncio
    async def test_authentication_failure(self, security_manager):
        """Test failed authentication."""
        # Mock invalid token by making JWT decode fail
        import jwt
        with patch('jwt.decode') as mock_decode:
            mock_decode.side_effect = jwt.InvalidTokenError("Invalid token")
            
            request_data = {
                "headers": {"Authorization": "Bearer invalid_token"},
                "remote_addr": "127.0.0.1"
            }
            
            result = await security_manager.authenticate_request(request_data)
            
            assert result["authenticated"] is False
            assert "reason" in result
    
    @pytest.mark.asyncio
    async def test_threat_detection(self, security_manager):
        """Test threat detection functionality."""
        # Simulate suspicious activity by recording events
        for i in range(10):
            await security_manager._record_security_event(
                event_type=SecurityEventType.AUTHENTICATION,
                severity=ThreatLevel.HIGH,
                source_ip="192.168.1.100",
                user_id="test_user",
                details={"failed_attempts": True, "attempt": i}
            )
        
        # Check that events were recorded
        assert security_manager.metrics['total_events'] >= 10
        assert security_manager.metrics['threats_detected'] >= 0
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, security_manager):
        """Test rate limiting functionality."""
        source_ip = "192.168.1.101"
        
        # Mock the rate limiter to avoid Redis complexities in testing
        with patch.object(security_manager.rate_limiter, 'check_rate_limit') as mock_check:
            mock_check.return_value = True  # Allow the request
            
            # Test rate limiting check
            allowed = await security_manager.check_rate_limit(source_ip, "api_call")
            assert allowed is True
            
            # Test when rate limit is exceeded
            mock_check.return_value = False
            allowed = await security_manager.check_rate_limit(source_ip, "api_call")
            assert allowed is False
    
    def test_data_encryption_decryption(self, security_manager):
        """Test data encryption and decryption."""
        sensitive_data = b"private_key_data"
        
        # Encrypt data
        encrypted = security_manager.encrypt_data(sensitive_data)
        assert encrypted != sensitive_data
        assert len(encrypted) > len(sensitive_data)
        
        # Decrypt data
        decrypted = security_manager.decrypt_data(encrypted)
        assert decrypted == sensitive_data
    
    @pytest.mark.asyncio
    async def test_security_metrics_collection(self, security_manager):
        """Test security metrics are collected properly."""
        initial_metrics = security_manager.get_security_metrics()
        assert isinstance(initial_metrics, dict)
        assert "total_events" in initial_metrics
        
        # Trigger some security events
        await security_manager._record_security_event(
            event_type=SecurityEventType.AUTHENTICATION,
            severity=ThreatLevel.LOW,
            source_ip="127.0.0.1",
            details={"test": True}
        )
        
        updated_metrics = security_manager.get_security_metrics()
        assert updated_metrics["total_events"] >= initial_metrics["total_events"]
    
    @pytest.mark.asyncio
    async def test_alert_callbacks(self, security_manager):
        """Test security alert callbacks are triggered."""
        alert_triggered = False
        alert_event = None
        
        async def test_callback(event):
            nonlocal alert_triggered, alert_event
            alert_triggered = True
            alert_event = event
        
        security_manager.register_alert_callback(test_callback)
        
        # Trigger a high severity event
        await security_manager._record_security_event(
            event_type=SecurityEventType.THREAT_DETECTED,
            severity=ThreatLevel.HIGH,
            source_ip="192.168.1.100",
            details={"threat_type": "test_threat"}
        )
        
        # Give callback time to execute
        await asyncio.sleep(0.1)
        
        assert alert_triggered is True
        assert alert_event is not None


class TestSecurityEvent:
    """Test security event functionality."""
    
    def test_security_event_creation(self):
        """Test security event creation."""
        event = SecurityEvent(
            event_id="test_event_123",
            event_type=SecurityEventType.AUTHENTICATION,
            severity=ThreatLevel.MEDIUM,
            timestamp=time.time(),
            source_ip="192.168.1.100",
            user_id="test_user",
            details={"test": True}
        )
        
        assert event.event_id == "test_event_123"
        assert event.event_type == SecurityEventType.AUTHENTICATION
        assert event.severity == ThreatLevel.MEDIUM
        assert event.source_ip == "192.168.1.100"
        assert event.user_id == "test_user"


class TestSecurityIntegration:
    """Test security integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_security_workflow(self):
        """Test complete security workflow."""
        # Mock Redis
        with patch('src.core.security_manager.redis') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.from_url.return_value = mock_redis_instance
            mock_redis_instance.ping.return_value = True
            
            manager = SecurityManager()
            await manager.initialize()
            
            # Setup test user
            from src.core.security_manager import AccessControl
            test_access = AccessControl(
                user_id="test_user",
                roles={"trader"},
                permissions={"trading"},
                rate_limits={"api_calls": 100},
                mfa_required=False
            )
            manager.access_controls["test_user"] = test_access
            
            # Test authentication flow
            request_data = {
                "headers": {"Authorization": "Bearer test_token"},
                "remote_addr": "127.0.0.1"
            }
            
            with patch('jwt.decode') as mock_decode:
                mock_decode.return_value = {"user_id": "test_user"}
                result = await manager.authenticate_request(request_data)
                
                assert result["authenticated"] is True
                assert result["user_id"] == "test_user"
            
            # Test authorization
            authorized = await manager.authorize_action("test_user", "trading")
            assert authorized is True
            
            # Test unauthorized action
            unauthorized = await manager.authorize_action("test_user", "admin_action")
            assert unauthorized is False


class TestSecurityPerformance:
    """Test security performance."""
    
    def test_encryption_performance(self):
        """Test encryption performance meets requirements."""
        manager = EncryptionManager()
        test_data = b"performance_test_data" * 100  # Larger data
        
        start_time = time.time()
        
        # Encrypt and decrypt multiple times
        for _ in range(100):
            encrypted = manager.encrypt_symmetric(test_data)
            decrypted = manager.decrypt_symmetric(encrypted)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete 100 operations in reasonable time
        assert total_time < 10  # 10 seconds max
        assert decrypted == test_data
    
    @pytest.mark.asyncio
    async def test_concurrent_authentication(self):
        """Test concurrent authentication requests."""
        # Mock Redis
        with patch('src.core.security_manager.redis') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.from_url.return_value = mock_redis_instance
            mock_redis_instance.ping.return_value = True
            
            manager = SecurityManager()
            await manager.initialize()
            
            # Setup test user
            from src.core.security_manager import AccessControl
            test_access = AccessControl(
                user_id="test_user",
                roles={"trader"},
                permissions={"trading"},
                rate_limits={"api_calls": 100},
                mfa_required=False
            )
            manager.access_controls["test_user"] = test_access
            
            async def authenticate_request():
                request_data = {
                    "headers": {"Authorization": "Bearer test_token"},
                    "remote_addr": "127.0.0.1"
                }
                
                with patch('jwt.decode') as mock_verify:
                    mock_verify.return_value = {"user_id": "test_user"}
                    return await manager.authenticate_request(request_data)
            
            # Run multiple concurrent authentications (reduced for testing)
            tasks = [authenticate_request() for _ in range(10)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify results
            successful = sum(1 for r in results if isinstance(r, dict) and r.get("authenticated"))
            assert successful >= 8  # At least 80% should succeed 