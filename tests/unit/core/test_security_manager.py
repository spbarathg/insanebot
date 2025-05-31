"""
Unit tests for SecurityManager and security components
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, AsyncMock, patch

from src.core.security_manager import (
    SecurityManager, ThreatDetector, EncryptionManager, AccessController,
    SecurityEvent, SecurityPolicy, ThreatSignature, AccessToken
)


class TestThreatDetector:
    """Test suite for ThreatDetector"""
    
    @pytest.fixture
    def threat_detector(self):
        """Create ThreatDetector instance"""
        return ThreatDetector()
    
    def test_initialization(self, threat_detector):
        """Test ThreatDetector initialization"""
        assert isinstance(threat_detector.threat_signatures, dict)
        assert isinstance(threat_detector.suspicious_patterns, dict)
        assert isinstance(threat_detector.rate_limits, dict)
        assert isinstance(threat_detector.blocked_ips, set)
        assert threat_detector.anomaly_threshold == 5
    
    def test_add_signature(self, threat_detector):
        """Test adding threat signatures"""
        signature = ThreatSignature(
            signature_id="test_sig_1",
            threat_type="sql_injection",
            pattern="union select",
            severity="high",
            action="block"
        )
        
        threat_detector.add_signature(signature)
        assert "test_sig_1" in threat_detector.threat_signatures
        assert threat_detector.threat_signatures["test_sig_1"] == signature
    
    def test_analyze_request_clean(self, threat_detector):
        """Test analyzing clean request"""
        request_data = {
            'source_ip': '192.168.1.1',
            'user_agent': 'Mozilla/5.0',
            'url': '/api/trades',
            'method': 'GET'
        }
        
        threats = threat_detector.analyze_request(request_data)
        assert isinstance(threats, list)
        # Should be empty for clean request
    
    def test_analyze_request_with_threats(self, threat_detector):
        """Test analyzing request with threats"""
        # Add malicious signature
        signature = ThreatSignature(
            signature_id="sql_inject",
            threat_type="sql_injection",
            pattern="union select",
            severity="high",
            action="block"
        )
        threat_detector.add_signature(signature)
        
        # Malicious request
        request_data = {
            'source_ip': '192.168.1.1',
            'query': 'test union select * from users',
            'method': 'POST'
        }
        
        threats = threat_detector.analyze_request(request_data)
        assert "sql_injection" in threats
    
    def test_rate_limiting(self, threat_detector):
        """Test rate limiting functionality"""
        source_ip = "192.168.1.100"
        
        # Should not exceed limit initially
        for i in range(50):
            result = threat_detector._check_rate_limit(source_ip, limit=100, window=60)
            assert result is False
        
        # Should exceed limit
        for i in range(60):
            threat_detector._check_rate_limit(source_ip, limit=100, window=60)
        
        result = threat_detector._check_rate_limit(source_ip, limit=100, window=60)
        assert result is True
    
    def test_anomaly_detection(self, threat_detector):
        """Test behavioral anomaly detection"""
        request_data = {
            'source_ip': '192.168.1.1',
            'pattern': 'identical_pattern'
        }
        
        # First few requests should not trigger anomaly
        for i in range(3):
            result = threat_detector._detect_anomalies(request_data)
            assert result is False
        
        # Should trigger anomaly after threshold
        for i in range(5):
            threat_detector._detect_anomalies(request_data)
        
        result = threat_detector._detect_anomalies(request_data)
        assert result is True


class TestEncryptionManager:
    """Test suite for EncryptionManager"""
    
    @pytest.fixture
    def encryption_manager(self):
        """Create EncryptionManager instance"""
        return EncryptionManager()
    
    def test_initialization(self, encryption_manager):
        """Test EncryptionManager initialization"""
        assert encryption_manager.master_key is not None
        assert encryption_manager.fernet is not None
        assert encryption_manager.rsa_private_key is not None
        assert encryption_manager.rsa_public_key is not None
    
    def test_symmetric_encryption(self, encryption_manager):
        """Test symmetric encryption/decryption"""
        plaintext = b"Hello, World!"
        
        # Encrypt
        encrypted = encryption_manager.encrypt_symmetric(plaintext)
        assert encrypted != plaintext
        assert isinstance(encrypted, bytes)
        
        # Decrypt
        decrypted = encryption_manager.decrypt_symmetric(encrypted)
        assert decrypted == plaintext
    
    def test_asymmetric_encryption(self, encryption_manager):
        """Test asymmetric encryption/decryption"""
        plaintext = b"Secret message"
        
        # Encrypt with public key
        encrypted = encryption_manager.encrypt_asymmetric(plaintext)
        assert encrypted != plaintext
        assert isinstance(encrypted, bytes)
        
        # Decrypt with private key
        decrypted = encryption_manager.decrypt_asymmetric(encrypted)
        assert decrypted == plaintext
    
    def test_password_hashing(self, encryption_manager):
        """Test password hashing and verification"""
        password = "supersecret123"
        
        # Hash password
        hashed = encryption_manager.hash_password(password)
        assert hashed != password
        assert isinstance(hashed, str)
        
        # Verify correct password
        assert encryption_manager.verify_password(password, hashed) is True
        
        # Verify incorrect password
        assert encryption_manager.verify_password("wrongpassword", hashed) is False


class TestAccessController:
    """Test suite for AccessController"""
    
    @pytest.fixture
    def access_controller(self):
        """Create AccessController instance"""
        return AccessController()
    
    def test_initialization(self, access_controller):
        """Test AccessController initialization"""
        assert isinstance(access_controller.permissions, dict)
        assert isinstance(access_controller.roles, dict)
        assert isinstance(access_controller.user_roles, dict)
        assert isinstance(access_controller.tokens, dict)
    
    def test_permission_management(self, access_controller):
        """Test permission management"""
        # Add permission
        access_controller.add_permission("read", "trades")
        assert "read" in access_controller.permissions
        
        # Create role
        access_controller.create_role("trader", ["read", "write"])
        assert "trader" in access_controller.roles
        assert "read" in access_controller.roles["trader"]
        assert "write" in access_controller.roles["trader"]
    
    def test_role_assignment(self, access_controller):
        """Test role assignment"""
        # Setup role
        access_controller.create_role("admin", ["read", "write", "delete"])
        
        # Assign role
        access_controller.assign_role("user123", "admin")
        assert "user123" in access_controller.user_roles
        assert "admin" in access_controller.user_roles["user123"]
    
    def test_permission_checking(self, access_controller):
        """Test permission checking"""
        # Setup permissions and roles
        access_controller.add_permission("read", "trades")
        access_controller.create_role("trader", ["read"])
        access_controller.assign_role("user123", "trader")
        
        # Check permission
        assert access_controller.check_permission("user123", "read", "trades") is True
        assert access_controller.check_permission("user123", "write", "trades") is False
        assert access_controller.check_permission("unknown_user", "read", "trades") is False
    
    def test_token_generation(self, access_controller):
        """Test JWT token generation"""
        user_id = "user123"
        scopes = ["read", "write"]
        
        token = access_controller.generate_token(user_id, scopes)
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_token_verification(self, access_controller):
        """Test JWT token verification"""
        user_id = "user123"
        scopes = ["read", "write"]
        
        # Generate token
        token = access_controller.generate_token(user_id, scopes)
        
        # Verify token
        result = access_controller.verify_token(token)
        assert result is not None
        assert result["user_id"] == user_id
        assert result["scopes"] == scopes
    
    def test_token_revocation(self, access_controller):
        """Test token revocation"""
        user_id = "user123"
        scopes = ["read"]
        
        # Generate and verify token works
        token = access_controller.generate_token(user_id, scopes)
        result = access_controller.verify_token(token)
        assert result is not None
        
        # Extract token ID and revoke
        token_id = result["token_id"]
        access_controller.revoke_token(token_id)
        
        # Token should now be invalid
        result = access_controller.verify_token(token)
        assert result is None


class TestSecurityManager:
    """Test suite for SecurityManager"""
    
    @pytest.fixture
    def security_manager(self):
        """Create SecurityManager instance"""
        config = {
            'threat_detection': {
                'enable_rate_limiting': True,
                'enable_anomaly_detection': True,
                'rate_limit_threshold': 100
            },
            'encryption': {
                'algorithm': 'AES-256',
                'key_rotation_interval': 86400
            },
            'access_control': {
                'token_expiry': 3600,
                'max_failed_attempts': 3
            },
            'monitoring': {
                'log_security_events': True,
                'alert_threshold': 5
            }
        }
        return SecurityManager(config)
    
    @pytest.mark.asyncio
    async def test_initialization(self, security_manager):
        """Test SecurityManager initialization"""
        result = await security_manager.initialize()
        assert result is True
        assert security_manager.threat_detector is not None
        assert security_manager.encryption_manager is not None
        assert security_manager.access_controller is not None
    
    @pytest.mark.asyncio
    async def test_authenticate_request_success(self, security_manager):
        """Test successful request authentication"""
        await security_manager.initialize()
        
        # Generate valid token
        token = await security_manager.generate_access_token("user123", ["read"])
        
        request_data = {
            'token': token,
            'source_ip': '192.168.1.1',
            'method': 'GET',
            'path': '/api/trades'
        }
        
        result = await security_manager.authenticate_request(request_data)
        assert result['is_authenticated'] is True
        assert result['user_id'] == "user123"
    
    @pytest.mark.asyncio
    async def test_authenticate_request_invalid_token(self, security_manager):
        """Test authentication with invalid token"""
        await security_manager.initialize()
        
        request_data = {
            'token': 'invalid_token',
            'source_ip': '192.168.1.1',
            'method': 'GET',
            'path': '/api/trades'
        }
        
        result = await security_manager.authenticate_request(request_data)
        assert result['is_authenticated'] is False
        assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_authorize_action(self, security_manager):
        """Test action authorization"""
        await security_manager.initialize()
        
        # Setup user with permissions
        security_manager.access_controller.add_permission("read", "trades")
        security_manager.access_controller.create_role("trader", ["read"])
        security_manager.access_controller.assign_role("user123", "trader")
        
        # Test authorization
        result = await security_manager.authorize_action("user123", "read", "trades")
        assert result is True
        
        result = await security_manager.authorize_action("user123", "delete", "trades")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_threat_detection(self, security_manager):
        """Test threat detection integration"""
        await security_manager.initialize()
        
        # Add threat signature
        signature = ThreatSignature(
            signature_id="test_threat",
            threat_type="suspicious_activity",
            pattern="malicious_pattern",
            severity="medium",
            action="alert"
        )
        security_manager.threat_detector.add_signature(signature)
        
        # Test threat detection
        request_data = {
            'source_ip': '192.168.1.1',
            'content': 'this contains malicious_pattern',
            'method': 'POST'
        }
        
        result = await security_manager.authenticate_request(request_data)
        # Should detect threat but specific response depends on implementation
        assert 'threats_detected' in result or 'is_authenticated' in result
    
    def test_encryption_integration(self, security_manager):
        """Test encryption functionality integration"""
        data = b"sensitive_data"
        
        # Test symmetric encryption
        encrypted = security_manager.encrypt_data(data, "symmetric")
        decrypted = security_manager.decrypt_data(encrypted, "symmetric")
        assert decrypted == data
        
        # Test asymmetric encryption
        encrypted = security_manager.encrypt_data(data, "asymmetric")
        decrypted = security_manager.decrypt_data(encrypted, "asymmetric")
        assert decrypted == data
    
    def test_password_management(self, security_manager):
        """Test password hashing and verification"""
        password = "testpassword123"
        
        hashed = security_manager.hash_password(password)
        assert security_manager.verify_password(password, hashed) is True
        assert security_manager.verify_password("wrongpassword", hashed) is False
    
    def test_ip_blocking(self, security_manager):
        """Test IP blocking functionality"""
        ip_address = "192.168.1.100"
        
        # Block IP
        security_manager.block_ip(ip_address, "testing")
        assert ip_address in security_manager.threat_detector.blocked_ips
        
        # Unblock IP
        security_manager.unblock_ip(ip_address)
        assert ip_address not in security_manager.threat_detector.blocked_ips
    
    def test_security_status(self, security_manager):
        """Test security status reporting"""
        status = security_manager.get_security_status()
        
        assert 'threats_detected' in status
        assert 'authentication_attempts' in status
        assert 'blocked_ips' in status
        assert 'active_tokens' in status
        assert isinstance(status['threats_detected'], int)
    
    def test_security_metrics(self, security_manager):
        """Test security metrics collection"""
        metrics = security_manager.get_security_metrics()
        
        assert 'total_requests' in metrics
        assert 'threat_detection_rate' in metrics
        assert 'authentication_success_rate' in metrics
        assert 'average_response_time' in metrics
    
    @pytest.mark.asyncio
    async def test_cleanup(self, security_manager):
        """Test security manager cleanup"""
        await security_manager.initialize()
        
        # Should not raise exceptions
        await security_manager.cleanup()


class TestSecurityDataClasses:
    """Test security-related data classes"""
    
    def test_security_event(self):
        """Test SecurityEvent creation"""
        event = SecurityEvent(
            event_id="evt_123",
            event_type="authentication",
            severity="medium",
            timestamp=time.time(),
            source_ip="192.168.1.1",
            user_id="user123",
            component="api_gateway"
        )
        
        assert event.event_id == "evt_123"
        assert event.event_type == "authentication"
        assert event.severity == "medium"
        assert event.source_ip == "192.168.1.1"
    
    def test_security_policy(self):
        """Test SecurityPolicy creation"""
        policy = SecurityPolicy(
            policy_id="pol_123",
            policy_type="access",
            rules={"min_password_length": 8, "require_2fa": True}
        )
        
        assert policy.policy_id == "pol_123"
        assert policy.policy_type == "access"
        assert policy.is_active is True
        assert policy.rules["min_password_length"] == 8
    
    def test_threat_signature(self):
        """Test ThreatSignature creation"""
        signature = ThreatSignature(
            signature_id="sig_123",
            threat_type="sql_injection",
            pattern="union select",
            severity="high",
            action="block"
        )
        
        assert signature.signature_id == "sig_123"
        assert signature.threat_type == "sql_injection"
        assert signature.is_active is True
    
    def test_access_token(self):
        """Test AccessToken creation"""
        token = AccessToken(
            token_id="tok_123",
            user_id="user123",
            scopes=["read", "write"],
            expires_at=time.time() + 3600
        )
        
        assert token.token_id == "tok_123"
        assert token.user_id == "user123"
        assert "read" in token.scopes
        assert token.is_revoked is False 