"""
SecurityManager - Comprehensive security management for Ant Bot System

Provides centralized security controls, threat detection, access management,
API security, encryption, and security monitoring with real-time alerts.
"""

import os
import time
import hmac
import hashlib
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timedelta
import ipaddress
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import jwt
import secrets
import bcrypt

logger = logging.getLogger(__name__)

@dataclass
class SecurityEvent:
    """Represents a security event"""
    event_id: str
    event_type: str  # 'authentication', 'authorization', 'threat', 'anomaly', 'access'
    severity: str   # 'low', 'medium', 'high', 'critical'
    timestamp: float
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    component: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    threat_indicators: List[str] = field(default_factory=list)

@dataclass
class SecurityPolicy:
    """Security policy definition"""
    policy_id: str
    policy_type: str  # 'access', 'encryption', 'authentication', 'authorization'
    rules: Dict[str, Any]
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

@dataclass
class ThreatSignature:
    """Threat detection signature"""
    signature_id: str
    threat_type: str
    pattern: str
    severity: str
    action: str  # 'alert', 'block', 'quarantine'
    is_active: bool = True

@dataclass
class AccessToken:
    """Access token for API authentication"""
    token_id: str
    user_id: str
    scopes: List[str]
    expires_at: float
    is_revoked: bool = False
    created_at: float = field(default_factory=time.time)

class ThreatDetector:
    """Advanced threat detection system"""
    
    def __init__(self):
        self.threat_signatures = {}
        self.suspicious_patterns = defaultdict(int)
        self.rate_limits = defaultdict(list)
        self.blocked_ips = set()
        self.anomaly_threshold = 5
        
    def add_signature(self, signature: ThreatSignature):
        """Add threat detection signature"""
        self.threat_signatures[signature.signature_id] = signature
    
    def analyze_request(self, request_data: Dict[str, Any]) -> List[str]:
        """Analyze request for threats"""
        threats = []
        
        # Check against known signatures
        for signature in self.threat_signatures.values():
            if not signature.is_active:
                continue
                
            if self._matches_signature(request_data, signature):
                threats.append(signature.threat_type)
        
        # Rate limiting analysis
        source_ip = request_data.get('source_ip')
        if source_ip and self._check_rate_limit(source_ip):
            threats.append('rate_limit_exceeded')
        
        # Anomaly detection
        if self._detect_anomalies(request_data):
            threats.append('behavioral_anomaly')
        
        return threats
    
    def _matches_signature(self, request_data: Dict[str, Any], signature: ThreatSignature) -> bool:
        """Check if request matches threat signature"""
        # Simple pattern matching - in production this would be more sophisticated
        request_str = json.dumps(request_data, default=str).lower()
        return signature.pattern.lower() in request_str
    
    def _check_rate_limit(self, source_ip: str, limit: int = 100, window: int = 60) -> bool:
        """Check if IP exceeds rate limit"""
        current_time = time.time()
        
        # Clean old entries
        self.rate_limits[source_ip] = [
            t for t in self.rate_limits[source_ip] 
            if current_time - t < window
        ]
        
        # Add current request
        self.rate_limits[source_ip].append(current_time)
        
        return len(self.rate_limits[source_ip]) > limit
    
    def _detect_anomalies(self, request_data: Dict[str, Any]) -> bool:
        """Detect behavioral anomalies"""
        # Simple anomaly detection based on request patterns
        request_hash = hashlib.md5(
            json.dumps(request_data, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]
        
        self.suspicious_patterns[request_hash] += 1
        return self.suspicious_patterns[request_hash] > self.anomaly_threshold

class EncryptionManager:
    """Advanced encryption and key management"""
    
    def __init__(self, master_key: Optional[bytes] = None):
        self.master_key = master_key or self._generate_master_key()
        self.fernet = Fernet(self.master_key)
        self.rsa_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.rsa_public_key = self.rsa_private_key.public_key()
        
    def _generate_master_key(self) -> bytes:
        """Generate new master encryption key"""
        return Fernet.generate_key()
    
    def encrypt_symmetric(self, data: bytes) -> bytes:
        """Encrypt data using symmetric encryption"""
        return self.fernet.encrypt(data)
    
    def decrypt_symmetric(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using symmetric encryption"""
        return self.fernet.decrypt(encrypted_data)
    
    def encrypt_asymmetric(self, data: bytes) -> bytes:
        """Encrypt data using asymmetric encryption"""
        return self.rsa_public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    
    def decrypt_asymmetric(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using asymmetric encryption"""
        return self.rsa_private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

class AccessController:
    """Advanced access control and authorization"""
    
    def __init__(self):
        self.permissions = defaultdict(set)
        self.roles = defaultdict(set)
        self.user_roles = defaultdict(set)
        self.active_tokens = {}
        self.revoked_tokens = set()
        
    def add_permission(self, permission: str, resource: str):
        """Add permission for resource"""
        self.permissions[resource].add(permission)
    
    def create_role(self, role_name: str, permissions: List[str]):
        """Create role with permissions"""
        self.roles[role_name] = set(permissions)
    
    def assign_role(self, user_id: str, role_name: str):
        """Assign role to user"""
        self.user_roles[user_id].add(role_name)
    
    def check_permission(self, user_id: str, permission: str, resource: str = None) -> bool:
        """Check if user has permission"""
        user_permissions = set()
        
        # Get permissions from user roles
        for role in self.user_roles[user_id]:
            user_permissions.update(self.roles[role])
        
        # Check permission
        if resource:
            return permission in self.permissions[resource] and permission in user_permissions
        return permission in user_permissions
    
    def generate_token(self, user_id: str, scopes: List[str], expires_in: int = 3600) -> str:
        """Generate JWT access token"""
        payload = {
            'user_id': user_id,
            'scopes': scopes,
            'exp': time.time() + expires_in,
            'iat': time.time(),
            'jti': secrets.token_urlsafe(16)
        }
        
        token = jwt.encode(payload, self.master_secret, algorithm='HS256')
        
        # Store token info
        token_info = AccessToken(
            token_id=payload['jti'],
            user_id=user_id,
            scopes=scopes,
            expires_at=payload['exp']
        )
        self.active_tokens[payload['jti']] = token_info
        
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.master_secret, algorithms=['HS256'])
            token_id = payload.get('jti')
            
            # Check if token is revoked
            if token_id in self.revoked_tokens:
                return None
            
            # Check if token exists and is not expired
            if token_id in self.active_tokens:
                token_info = self.active_tokens[token_id]
                if not token_info.is_revoked and token_info.expires_at > time.time():
                    return payload
            
            return None
            
        except jwt.InvalidTokenError:
            return None
    
    def revoke_token(self, token_id: str):
        """Revoke access token"""
        self.revoked_tokens.add(token_id)
        if token_id in self.active_tokens:
            self.active_tokens[token_id].is_revoked = True

class SecurityManager:
    """Comprehensive security management system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Import existing security components
        from .security import KeyManager, IPWhitelist
        self.key_manager = KeyManager()
        self.ip_whitelist = IPWhitelist()
        
        # Core security components
        self.threat_detector = ThreatDetector()
        self.encryption_manager = EncryptionManager()
        self.access_controller = AccessController()
        
        # Security monitoring
        self.security_events: deque = deque(maxlen=self.config.get('event_history_size', 10000))
        self.security_policies: Dict[str, SecurityPolicy] = {}
        self.alert_callbacks: List[Callable] = []
        
        # Threat intelligence
        self.threat_intel_feeds = []
        self.suspicious_activities = defaultdict(int)
        self.blocked_entities = {
            'ips': set(),
            'users': set(),
            'tokens': set()
        }
        
        # Security metrics
        self.security_metrics = {
            'total_events': 0,
            'threats_blocked': 0,
            'failed_authentications': 0,
            'successful_authentications': 0,
            'policy_violations': 0
        }
        
        # Background tasks
        self.monitoring_task = None
        self.cleanup_task = None
        self.is_running = False
        
        self._initialized = False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default security configuration"""
        return {
            "event_history_size": 10000,
            "threat_detection_enabled": True,
            "encryption_required": True,
            "token_expiry": 3600,
            "max_failed_attempts": 5,
            "lockout_duration": 300,
            "audit_logging": True,
            "real_time_monitoring": True,
            "threat_intel_enabled": True,
            "security_headers": {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "X-XSS-Protection": "1; mode=block",
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
            }
        }
    
    async def initialize(self) -> bool:
        """Initialize the security manager"""
        try:
            # Setup default security policies
            await self._setup_default_policies()
            
            # Initialize threat signatures
            await self._setup_threat_signatures()
            
            # Setup access control
            await self._setup_access_control()
            
            # Load threat intelligence
            if self.config.get('threat_intel_enabled', True):
                await self._load_threat_intelligence()
            
            # Start monitoring tasks
            await self._start_monitoring()
            
            self._initialized = True
            logger.info("SecurityManager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SecurityManager: {e}")
            return False
    
    async def _setup_default_policies(self):
        """Setup default security policies"""
        # Authentication policy
        auth_policy = SecurityPolicy(
            policy_id="default_authentication",
            policy_type="authentication",
            rules={
                "require_2fa": False,
                "password_min_length": 8,
                "password_complexity": True,
                "session_timeout": 3600,
                "max_concurrent_sessions": 5
            }
        )
        self.security_policies[auth_policy.policy_id] = auth_policy
        
        # Encryption policy
        encryption_policy = SecurityPolicy(
            policy_id="default_encryption",
            policy_type="encryption",
            rules={
                "encrypt_at_rest": True,
                "encrypt_in_transit": True,
                "key_rotation_interval": 86400,
                "cipher_suite": "AES-256-GCM"
            }
        )
        self.security_policies[encryption_policy.policy_id] = encryption_policy
        
        # Access control policy
        access_policy = SecurityPolicy(
            policy_id="default_access",
            policy_type="access",
            rules={
                "default_deny": True,
                "principle_of_least_privilege": True,
                "session_validation": True,
                "ip_whitelist_required": False
            }
        )
        self.security_policies[access_policy.policy_id] = access_policy
    
    async def _setup_threat_signatures(self):
        """Setup threat detection signatures"""
        signatures = [
            ThreatSignature(
                signature_id="sql_injection",
                threat_type="sql_injection",
                pattern="union select|drop table|insert into|delete from",
                severity="high",
                action="block"
            ),
            ThreatSignature(
                signature_id="xss_attempt",
                threat_type="cross_site_scripting",
                pattern="<script|javascript:|onerror=|onload=",
                severity="high",
                action="block"
            ),
            ThreatSignature(
                signature_id="path_traversal",
                threat_type="directory_traversal",
                pattern="../|..\\|%2e%2e%2f|%2e%2e%5c",
                severity="medium",
                action="alert"
            ),
            ThreatSignature(
                signature_id="command_injection",
                threat_type="command_injection",
                pattern="|;|&&|`|$(",
                severity="high",
                action="block"
            )
        ]
        
        for signature in signatures:
            self.threat_detector.add_signature(signature)
    
    async def _setup_access_control(self):
        """Setup access control roles and permissions"""
        # Define permissions
        permissions = [
            "read", "write", "delete", "execute", "admin",
            "trade", "view_metrics", "configure_system", "manage_users"
        ]
        
        # Create roles
        self.access_controller.create_role("admin", permissions)
        self.access_controller.create_role("trader", ["read", "write", "trade", "view_metrics"])
        self.access_controller.create_role("viewer", ["read", "view_metrics"])
        self.access_controller.create_role("system", ["read", "write", "execute", "configure_system"])
        
        # Setup master secret for JWT
        self.access_controller.master_secret = os.getenv('JWT_SECRET', secrets.token_urlsafe(32))
    
    async def _load_threat_intelligence(self):
        """Load threat intelligence feeds"""
        # This would typically load from external threat intel sources
        # For now, we'll use placeholder data
        pass
    
    async def _start_monitoring(self):
        """Start security monitoring tasks"""
        if self.config.get('real_time_monitoring', True):
            self.is_running = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _monitoring_loop(self):
        """Main security monitoring loop"""
        while self.is_running:
            try:
                # Monitor for suspicious activities
                await self._analyze_security_events()
                
                # Check for policy violations
                await self._check_policy_compliance()
                
                # Update threat intelligence
                if self.config.get('threat_intel_enabled', True):
                    await self._update_threat_intelligence()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in security monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self):
        """Cleanup expired tokens and old events"""
        while self.is_running:
            try:
                # Cleanup expired tokens
                current_time = time.time()
                expired_tokens = [
                    token_id for token_id, token_info in self.access_controller.active_tokens.items()
                    if token_info.expires_at < current_time
                ]
                
                for token_id in expired_tokens:
                    del self.access_controller.active_tokens[token_id]
                
                # Cleanup old security events (keep last 24 hours)
                cutoff_time = current_time - 86400
                while self.security_events and self.security_events[0].timestamp < cutoff_time:
                    self.security_events.popleft()
                
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                logger.error(f"Error in security cleanup loop: {e}")
                await asyncio.sleep(3600)
    
    async def authenticate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate incoming request"""
        try:
            # Extract authentication data
            token = request_data.get('token') or request_data.get('authorization', '').replace('Bearer ', '')
            source_ip = request_data.get('source_ip')
            user_agent = request_data.get('user_agent')
            
            # Check IP whitelist if required
            access_policy = self.security_policies.get('default_access')
            if access_policy and access_policy.rules.get('ip_whitelist_required', False):
                if not self.ip_whitelist.is_allowed(source_ip):
                    await self._record_security_event(
                        event_type="access",
                        severity="medium",
                        source_ip=source_ip,
                        details={"reason": "ip_not_whitelisted"}
                    )
                    return {"authenticated": False, "reason": "ip_not_allowed"}
            
            # Threat detection
            if self.config.get('threat_detection_enabled', True):
                threats = self.threat_detector.analyze_request(request_data)
                if threats:
                    await self._record_security_event(
                        event_type="threat",
                        severity="high",
                        source_ip=source_ip,
                        details={"threats": threats}
                    )
                    return {"authenticated": False, "reason": "threat_detected", "threats": threats}
            
            # Token verification
            if token:
                payload = self.access_controller.verify_token(token)
                if payload:
                    self.security_metrics['successful_authentications'] += 1
                    await self._record_security_event(
                        event_type="authentication",
                        severity="low",
                        source_ip=source_ip,
                        user_id=payload.get('user_id'),
                        details={"method": "token", "success": True}
                    )
                    return {
                        "authenticated": True,
                        "user_id": payload.get('user_id'),
                        "scopes": payload.get('scopes', [])
                    }
                else:
                    self.security_metrics['failed_authentications'] += 1
                    await self._record_security_event(
                        event_type="authentication",
                        severity="medium",
                        source_ip=source_ip,
                        details={"method": "token", "success": False, "reason": "invalid_token"}
                    )
                    return {"authenticated": False, "reason": "invalid_token"}
            
            return {"authenticated": False, "reason": "no_credentials"}
            
        except Exception as e:
            logger.error(f"Error in authentication: {e}")
            return {"authenticated": False, "reason": "authentication_error"}
    
    async def authorize_action(self, user_id: str, action: str, resource: str = None) -> bool:
        """Authorize user action"""
        try:
            # Check if user has required permission
            has_permission = self.access_controller.check_permission(user_id, action, resource)
            
            if not has_permission:
                await self._record_security_event(
                    event_type="authorization",
                    severity="medium",
                    user_id=user_id,
                    details={
                        "action": action,
                        "resource": resource,
                        "result": "denied"
                    }
                )
                self.security_metrics['policy_violations'] += 1
            
            return has_permission
            
        except Exception as e:
            logger.error(f"Error in authorization: {e}")
            return False
    
    async def generate_access_token(self, user_id: str, scopes: List[str]) -> str:
        """Generate access token for user"""
        try:
            token = self.access_controller.generate_token(user_id, scopes)
            
            await self._record_security_event(
                event_type="authentication",
                severity="low",
                user_id=user_id,
                details={
                    "action": "token_generated",
                    "scopes": scopes
                }
            )
            
            return token
            
        except Exception as e:
            logger.error(f"Error generating token: {e}")
            raise
    
    async def revoke_access_token(self, token_id: str, user_id: str = None):
        """Revoke access token"""
        try:
            self.access_controller.revoke_token(token_id)
            
            await self._record_security_event(
                event_type="authentication",
                severity="low",
                user_id=user_id,
                details={
                    "action": "token_revoked",
                    "token_id": token_id
                }
            )
            
        except Exception as e:
            logger.error(f"Error revoking token: {e}")
            raise
    
    def encrypt_data(self, data: bytes, method: str = "symmetric") -> bytes:
        """Encrypt sensitive data"""
        try:
            if method == "symmetric":
                return self.encryption_manager.encrypt_symmetric(data)
            elif method == "asymmetric":
                return self.encryption_manager.encrypt_asymmetric(data)
            else:
                raise ValueError(f"Unsupported encryption method: {method}")
                
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: bytes, method: str = "symmetric") -> bytes:
        """Decrypt sensitive data"""
        try:
            if method == "symmetric":
                return self.encryption_manager.decrypt_symmetric(encrypted_data)
            elif method == "asymmetric":
                return self.encryption_manager.decrypt_asymmetric(encrypted_data)
            else:
                raise ValueError(f"Unsupported decryption method: {method}")
                
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            raise
    
    def hash_password(self, password: str) -> str:
        """Hash password securely"""
        return self.encryption_manager.hash_password(password)
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return self.encryption_manager.verify_password(password, hashed)
    
    async def _record_security_event(
        self,
        event_type: str,
        severity: str,
        source_ip: str = None,
        user_id: str = None,
        component: str = None,
        details: Dict[str, Any] = None
    ):
        """Record security event"""
        event = SecurityEvent(
            event_id=secrets.token_urlsafe(16),
            event_type=event_type,
            severity=severity,
            timestamp=time.time(),
            source_ip=source_ip,
            user_id=user_id,
            component=component,
            details=details or {}
        )
        
        self.security_events.append(event)
        self.security_metrics['total_events'] += 1
        
        # Trigger alerts for high/critical severity events
        if severity in ['high', 'critical']:
            for callback in self.alert_callbacks:
                try:
                    await callback(event)
                except Exception as e:
                    logger.error(f"Error in security alert callback: {e}")
    
    async def _analyze_security_events(self):
        """Analyze recent security events for patterns"""
        if len(self.security_events) < 10:
            return
        
        recent_events = list(self.security_events)[-100:]  # Last 100 events
        
        # Look for patterns indicating potential attacks
        ip_activity = defaultdict(int)
        user_activity = defaultdict(int)
        
        for event in recent_events:
            if event.source_ip:
                ip_activity[event.source_ip] += 1
            if event.user_id:
                user_activity[event.user_id] += 1
        
        # Alert on suspicious IP activity
        for ip, count in ip_activity.items():
            if count > 20:  # More than 20 events from same IP
                await self._record_security_event(
                    event_type="anomaly",
                    severity="medium",
                    source_ip=ip,
                    details={"reason": "excessive_activity", "event_count": count}
                )
    
    async def _check_policy_compliance(self):
        """Check compliance with security policies"""
        # This would implement policy compliance checking
        pass
    
    async def _update_threat_intelligence(self):
        """Update threat intelligence data"""
        # This would update from external threat intel feeds
        pass
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for security alerts"""
        self.alert_callbacks.append(callback)
    
    def block_ip(self, ip_address: str, reason: str = "security_violation"):
        """Block IP address"""
        self.blocked_entities['ips'].add(ip_address)
        self.threat_detector.blocked_ips.add(ip_address)
    
    def unblock_ip(self, ip_address: str):
        """Unblock IP address"""
        self.blocked_entities['ips'].discard(ip_address)
        self.threat_detector.blocked_ips.discard(ip_address)
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        recent_events = [e for e in self.security_events if time.time() - e.timestamp < 3600]
        
        threat_levels = defaultdict(int)
        for event in recent_events:
            threat_levels[event.severity] += 1
        
        return {
            "overall_status": "secure" if threat_levels.get('critical', 0) == 0 else "at_risk",
            "total_events_24h": len([e for e in self.security_events if time.time() - e.timestamp < 86400]),
            "recent_threats": len([e for e in recent_events if e.event_type == 'threat']),
            "blocked_entities": {
                "ips": len(self.blocked_entities['ips']),
                "users": len(self.blocked_entities['users']),
                "tokens": len(self.blocked_entities['tokens'])
            },
            "active_tokens": len(self.access_controller.active_tokens),
            "security_metrics": self.security_metrics.copy(),
            "threat_levels": dict(threat_levels),
            "policies_active": len([p for p in self.security_policies.values() if p.is_active])
        }
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get detailed security metrics"""
        return {
            "events": {
                "total": len(self.security_events),
                "by_type": defaultdict(int),
                "by_severity": defaultdict(int)
            },
            "authentication": {
                "successful": self.security_metrics['successful_authentications'],
                "failed": self.security_metrics['failed_authentications'],
                "active_tokens": len(self.access_controller.active_tokens)
            },
            "threats": {
                "blocked": self.security_metrics['threats_blocked'],
                "signatures_active": len([s for s in self.threat_detector.threat_signatures.values() if s.is_active])
            },
            "policies": {
                "total": len(self.security_policies),
                "active": len([p for p in self.security_policies.values() if p.is_active]),
                "violations": self.security_metrics['policy_violations']
            }
        }
    
    async def cleanup(self):
        """Cleanup security manager resources"""
        try:
            self.is_running = False
            
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Clear sensitive data
            self.access_controller.active_tokens.clear()
            self.security_events.clear()
            
            logger.info("SecurityManager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during SecurityManager cleanup: {e}") 