"""
Enhanced Security Manager - Production-Grade Security System

Provides comprehensive security features including:
- Multi-layer encryption
- Advanced threat detection
- Audit logging
- Rate limiting
- Access control
"""

import asyncio
import hashlib
import hmac
import secrets
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import ipaddress
import re
import logging

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import bcrypt
import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
import redis.asyncio as redis


class SecurityEventType(Enum):
    """Types of security events."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    ACCESS_VIOLATION = "access_violation"
    THREAT_DETECTED = "threat_detected"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_BREACH_ATTEMPT = "data_breach_attempt"
    SYSTEM_INTRUSION = "system_intrusion"


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_id: str
    event_type: SecurityEventType
    severity: ThreatLevel
    timestamp: float
    source_ip: str
    user_id: Optional[str] = None
    endpoint: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    geolocation: Optional[Dict[str, Any]] = None
    threat_score: float = 0.0


@dataclass
class AccessControl:
    """Access control configuration."""
    user_id: str
    roles: Set[str]
    permissions: Set[str]
    ip_whitelist: Set[str] = field(default_factory=set)
    ip_blacklist: Set[str] = field(default_factory=set)
    rate_limits: Dict[str, int] = field(default_factory=dict)
    mfa_required: bool = True
    session_timeout: int = 3600  # seconds


class EncryptionManager:
    """Advanced encryption and key management."""
    
    def __init__(self):
        self.master_key = self._generate_or_load_master_key()
        self.fernet = Fernet(self.master_key)
        self.rsa_private_key, self.rsa_public_key = self._generate_rsa_keypair()
        self.password_hasher = PasswordHasher(
            time_cost=4,
            memory_cost=65536,
            parallelism=2,
            hash_len=32,
            salt_len=16
        )
    
    def _generate_or_load_master_key(self) -> bytes:
        """Generate or load master encryption key."""
        # In production, this should be loaded from secure key management service
        key_file = ".master_key"
        try:
            with open(key_file, 'rb') as f:
                return f.read()
        except FileNotFoundError:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def _generate_rsa_keypair(self):
        """Generate RSA keypair for asymmetric encryption."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096
        )
        public_key = private_key.public_key()
        return private_key, public_key
    
    def encrypt_symmetric(self, data: bytes) -> bytes:
        """Encrypt data using symmetric encryption."""
        return self.fernet.encrypt(data)
    
    def decrypt_symmetric(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using symmetric encryption."""
        return self.fernet.decrypt(encrypted_data)
    
    def encrypt_asymmetric(self, data: bytes) -> bytes:
        """Encrypt data using asymmetric encryption."""
        return self.rsa_public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    
    def decrypt_asymmetric(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using asymmetric encryption."""
        return self.rsa_private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    
    def hash_password(self, password: str) -> str:
        """Hash password using Argon2."""
        return self.password_hasher.hash(password)
    
    def verify_password(self, password: str, hash: str) -> bool:
        """Verify password against hash."""
        try:
            self.password_hasher.verify(hash, password)
            return True
        except VerifyMismatchError:
            return False
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure token."""
        return secrets.token_urlsafe(length)


class ThreatDetector:
    """Advanced threat detection system."""
    
    def __init__(self):
        self.suspicious_patterns = self._load_suspicious_patterns()
        self.ip_reputation_cache = {}
        self.behavioral_baselines = {}
        self.anomaly_thresholds = {
            'login_frequency': 10,  # per hour
            'api_calls': 1000,  # per hour
            'failed_attempts': 5,  # consecutive
            'unusual_locations': 3,  # different countries per day
        }
    
    def _load_suspicious_patterns(self) -> List[re.Pattern]:
        """Load patterns for detecting suspicious activity."""
        patterns = [
            r'union\s+select',  # SQL injection
            r'<script.*?>',  # XSS
            r'\.\./',  # Directory traversal
            r'exec\s*\(',  # Code injection
            r'eval\s*\(',  # Code injection
            r'system\s*\(',  # System command injection
        ]
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    async def analyze_request(self, request_data: Dict[str, Any]) -> float:
        """Analyze request for threats and return threat score (0-1)."""
        threat_score = 0.0
        
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            for value in self._extract_string_values(request_data):
                if pattern.search(str(value)):
                    threat_score += 0.3
        
        # Check IP reputation
        ip_score = await self._check_ip_reputation(request_data.get('remote_addr'))
        threat_score += ip_score
        
        # Check behavioral anomalies
        behavioral_score = await self._check_behavioral_anomalies(request_data)
        threat_score += behavioral_score
        
        # Check rate patterns
        rate_score = await self._check_rate_patterns(request_data)
        threat_score += rate_score
        
        return min(threat_score, 1.0)
    
    def _extract_string_values(self, data: Any) -> List[str]:
        """Extract all string values from nested data structure."""
        values = []
        if isinstance(data, str):
            values.append(data)
        elif isinstance(data, dict):
            for value in data.values():
                values.extend(self._extract_string_values(value))
        elif isinstance(data, (list, tuple)):
            for item in data:
                values.extend(self._extract_string_values(item))
        return values
    
    async def _check_ip_reputation(self, ip_address: str) -> float:
        """Check IP address reputation."""
        if not ip_address:
            return 0.0
        
        # Check cache first
        if ip_address in self.ip_reputation_cache:
            return self.ip_reputation_cache[ip_address]
        
        # Basic checks
        score = 0.0
        
        # Check if IP is in private ranges
        try:
            ip_obj = ipaddress.ip_address(ip_address)
            if ip_obj.is_private:
                score = 0.0  # Private IPs are generally safe
            elif ip_obj.is_loopback:
                score = 0.0  # Loopback is safe
        except ValueError:
            score = 0.2  # Invalid IP format is suspicious
        
        # Cache result
        self.ip_reputation_cache[ip_address] = score
        return score
    
    async def _check_behavioral_anomalies(self, request_data: Dict[str, Any]) -> float:
        """Check for behavioral anomalies."""
        # Placeholder for behavioral analysis
        # In production, this would analyze user behavior patterns
        return 0.0
    
    async def _check_rate_patterns(self, request_data: Dict[str, Any]) -> float:
        """Check for suspicious rate patterns."""
        # Placeholder for rate pattern analysis
        # In production, this would analyze request frequency patterns
        return 0.0


class RateLimiter:
    """Advanced rate limiting system."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.default_limits = {
            'api_calls': (100, 3600),  # 100 per hour
            'login_attempts': (5, 300),  # 5 per 5 minutes
            'password_resets': (3, 3600),  # 3 per hour
            'trades': (50, 3600),  # 50 trades per hour
        }
    
    async def check_rate_limit(
        self, 
        identifier: str, 
        action: str,
        custom_limit: Optional[tuple] = None
    ) -> bool:
        """Check if action is within rate limits."""
        limit, window = custom_limit or self.default_limits.get(action, (100, 3600))
        
        key = f"rate_limit:{action}:{identifier}"
        current_time = int(time.time())
        window_start = current_time - window
        
        # Use Redis sorted set for sliding window
        pipe = self.redis.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Count current entries
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(current_time): current_time})
        
        # Set expiry
        pipe.expire(key, window)
        
        results = await pipe.execute()
        current_count = results[1]
        
        return current_count < limit
    
    async def get_rate_limit_status(
        self, 
        identifier: str, 
        action: str
    ) -> Dict[str, int]:
        """Get current rate limit status."""
        limit, window = self.default_limits.get(action, (100, 3600))
        key = f"rate_limit:{action}:{identifier}"
        
        current_time = int(time.time())
        window_start = current_time - window
        
        # Clean old entries and count
        await self.redis.zremrangebyscore(key, 0, window_start)
        current_count = await self.redis.zcard(key)
        
        return {
            'limit': limit,
            'remaining': max(0, limit - current_count),
            'reset_time': current_time + window,
            'window_seconds': window
        }


class SecurityAuditor:
    """Security audit and compliance system."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.audit_retention_days = 90
    
    async def log_security_event(self, event: SecurityEvent):
        """Log security event for audit trail."""
        event_data = {
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'severity': event.severity.value,
            'timestamp': event.timestamp,
            'source_ip': event.source_ip,
            'user_id': event.user_id,
            'endpoint': event.endpoint,
            'user_agent': event.user_agent,
            'details': event.details,
            'geolocation': event.geolocation,
            'threat_score': event.threat_score
        }
        
        # Store in Redis with expiry
        key = f"security_audit:{event.event_id}"
        await self.redis.setex(
            key, 
            self.audit_retention_days * 24 * 3600,
            json.dumps(event_data, default=str)
        )
        
        # Also store in time-series for analytics
        ts_key = f"security_events:{event.event_type.value}"
        await self.redis.zadd(
            ts_key,
            {event.event_id: event.timestamp}
        )
    
    async def get_security_metrics(
        self, 
        start_time: float, 
        end_time: float
    ) -> Dict[str, Any]:
        """Get security metrics for time period."""
        metrics = {
            'total_events': 0,
            'events_by_type': {},
            'events_by_severity': {},
            'top_source_ips': {},
            'threat_score_distribution': [],
        }
        
        # Get events in time range
        for event_type in SecurityEventType:
            ts_key = f"security_events:{event_type.value}"
            event_ids = await self.redis.zrangebyscore(
                ts_key, start_time, end_time
            )
            
            metrics['events_by_type'][event_type.value] = len(event_ids)
            metrics['total_events'] += len(event_ids)
        
        return metrics


class SecurityManager:
    """Main security management system."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.encryption_manager = EncryptionManager()
        self.threat_detector = ThreatDetector()
        self.redis = redis.from_url(redis_url)
        self.rate_limiter = RateLimiter(self.redis)
        self.auditor = SecurityAuditor(self.redis)
        self.access_controls: Dict[str, AccessControl] = {}
        self.security_policies = self._load_security_policies()
        self.alert_callbacks = []
        
        # Initialize security metrics
        self.metrics = {
            'total_events': 0,
            'threats_detected': 0,
            'blocked_attempts': 0,
            'successful_authentications': 0,
            'failed_authentications': 0,
        }
    
    async def initialize(self):
        """Initialize security manager."""
        await self.redis.ping()
        self._setup_default_policies()
    
    def _load_security_policies(self) -> Dict[str, Any]:
        """Load security policies configuration."""
        return {
            'password_policy': {
                'min_length': 12,
                'require_uppercase': True,
                'require_lowercase': True,
                'require_numbers': True,
                'require_special_chars': True,
                'max_age_days': 90,
            },
            'session_policy': {
                'timeout_seconds': 3600,
                'max_concurrent_sessions': 5,
                'require_mfa': True,
            },
            'access_policy': {
                'max_failed_attempts': 5,
                'lockout_duration_minutes': 30,
                'require_ip_whitelist': False,
            }
        }
    
    def _setup_default_policies(self):
        """Setup default security policies."""
        # Default admin access control
        admin_access = AccessControl(
            user_id="admin",
            roles={"admin", "trader", "viewer"},
            permissions={"all"},
            rate_limits={"api_calls": 1000},
            mfa_required=True
        )
        self.access_controls["admin"] = admin_access
    
    async def authenticate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate incoming request."""
        start_time = time.time()
        
        # Extract authentication token
        auth_header = request_data.get('headers', {}).get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return {
                'authenticated': False,
                'reason': 'missing_token',
                'threat_score': 0.2
            }
        
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        
        # Verify token
        try:
            payload = jwt.decode(
                token, 
                'your-secret-key',  # In production, use proper key management
                algorithms=['HS256']
            )
            user_id = payload.get('user_id')
            
            if not user_id:
                return {
                    'authenticated': False,
                    'reason': 'invalid_token',
                    'threat_score': 0.3
                }
            
            # Check if user exists and is active
            if user_id not in self.access_controls:
                return {
                    'authenticated': False,
                    'reason': 'user_not_found',
                    'threat_score': 0.4
                }
            
            # Analyze request for threats
            threat_score = await self.threat_detector.analyze_request(request_data)
            
            # Log successful authentication
            await self._record_security_event(
                event_type=SecurityEventType.AUTHENTICATION,
                severity=ThreatLevel.LOW,
                source_ip=request_data.get('remote_addr', ''),
                user_id=user_id,
                details={'success': True, 'threat_score': threat_score}
            )
            
            self.metrics['successful_authentications'] += 1
            
            return {
                'authenticated': True,
                'user_id': user_id,
                'roles': list(self.access_controls[user_id].roles),
                'scopes': list(self.access_controls[user_id].permissions),
                'threat_score': threat_score,
                'auth_time_ms': (time.time() - start_time) * 1000
            }
            
        except jwt.ExpiredSignatureError:
            await self._record_security_event(
                event_type=SecurityEventType.AUTHENTICATION,
                severity=ThreatLevel.MEDIUM,
                source_ip=request_data.get('remote_addr', ''),
                details={'error': 'expired_token'}
            )
            return {
                'authenticated': False,
                'reason': 'expired_token',
                'threat_score': 0.5
            }
        
        except jwt.InvalidTokenError:
            await self._record_security_event(
                event_type=SecurityEventType.AUTHENTICATION,
                severity=ThreatLevel.HIGH,
                source_ip=request_data.get('remote_addr', ''),
                details={'error': 'invalid_token'}
            )
            return {
                'authenticated': False,
                'reason': 'invalid_token',
                'threat_score': 0.7
            }
    
    async def authorize_action(
        self, 
        user_id: str, 
        action: str, 
        resource_data: Dict[str, Any] = None
    ) -> bool:
        """Authorize user action on resource."""
        if user_id not in self.access_controls:
            await self._record_security_event(
                event_type=SecurityEventType.AUTHORIZATION,
                severity=ThreatLevel.HIGH,
                user_id=user_id,
                details={'action': action, 'error': 'user_not_found'}
            )
            return False
        
        access_control = self.access_controls[user_id]
        
        # Check if user has required permissions
        if 'all' in access_control.permissions or action in access_control.permissions:
            return True
        
        # Check role-based permissions
        required_roles = self._get_required_roles(action)
        if access_control.roles.intersection(required_roles):
            return True
        
        # Log unauthorized access attempt
        await self._record_security_event(
            event_type=SecurityEventType.ACCESS_VIOLATION,
            severity=ThreatLevel.MEDIUM,
            user_id=user_id,
            details={
                'action': action, 
                'user_roles': list(access_control.roles),
                'required_roles': list(required_roles)
            }
        )
        
        return False
    
    def _get_required_roles(self, action: str) -> Set[str]:
        """Get required roles for action."""
        role_mapping = {
            'execute_trade': {'trader', 'admin'},
            'view_portfolio': {'viewer', 'trader', 'admin'},
            'modify_settings': {'admin'},
            'view_logs': {'admin'},
        }
        return role_mapping.get(action, {'admin'})
    
    async def check_rate_limit(self, identifier: str, action: str) -> bool:
        """Check rate limits for identifier and action."""
        allowed = await self.rate_limiter.check_rate_limit(identifier, action)
        
        if not allowed:
            await self._record_security_event(
                event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                severity=ThreatLevel.MEDIUM,
                source_ip=identifier,
                details={'action': action}
            )
            self.metrics['blocked_attempts'] += 1
        
        return allowed
    
    async def assess_threat_level(self, source_ip: str) -> float:
        """Assess overall threat level for source IP."""
        # Get recent security events for this IP
        recent_events = await self._get_recent_events_for_ip(source_ip)
        
        if not recent_events:
            return 0.0
        
        # Calculate threat score based on recent activity
        threat_score = 0.0
        for event in recent_events:
            if event['severity'] == 'critical':
                threat_score += 0.4
            elif event['severity'] == 'high':
                threat_score += 0.3
            elif event['severity'] == 'medium':
                threat_score += 0.2
            else:
                threat_score += 0.1
        
        return min(threat_score, 1.0)
    
    async def _get_recent_events_for_ip(
        self, 
        source_ip: str, 
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get recent security events for IP address."""
        # This would query the audit log for recent events
        # Simplified implementation
        return []
    
    async def _record_security_event(
        self,
        event_type: SecurityEventType,
        severity: ThreatLevel,
        source_ip: str = '',
        user_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        details: Dict[str, Any] = None
    ):
        """Record a security event."""
        event = SecurityEvent(
            event_id=self.encryption_manager.generate_secure_token(16),
            event_type=event_type,
            severity=severity,
            timestamp=time.time(),
            source_ip=source_ip,
            user_id=user_id,
            endpoint=endpoint,
            details=details or {}
        )
        
        # Log to audit system
        await self.auditor.log_security_event(event)
        
        # Update metrics
        self.metrics['total_events'] += 1
        if event.threat_score > 0.5:
            self.metrics['threats_detected'] += 1
        
        # Trigger alerts for high severity events
        if severity in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            await self._trigger_security_alert(event)
    
    async def _trigger_security_alert(self, event: SecurityEvent):
        """Trigger security alert for high severity events."""
        for callback in self.alert_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logging.error(f"Failed to trigger security alert: {e}")
    
    def register_alert_callback(self, callback):
        """Register callback for security alerts."""
        self.alert_callbacks.append(callback)
    
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt sensitive data."""
        return self.encryption_manager.encrypt_symmetric(data)
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt sensitive data."""
        return self.encryption_manager.decrypt_symmetric(encrypted_data)
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get current security metrics."""
        return self.metrics.copy()
    
    async def shutdown(self):
        """Shutdown security manager."""
        await self.redis.close() 