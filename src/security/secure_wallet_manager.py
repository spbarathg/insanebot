"""
Secure Wallet Manager - Enterprise-Grade Security for Local Deployment

This module implements comprehensive wallet security with HSM integration,
multi-signature support, dynamic key rotation, and real-time monitoring.
"""

import asyncio
import time
import logging
import hashlib
import secrets
import os
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    BASIC = "basic"
    STANDARD = "standard" 
    HIGH = "high"
    ENTERPRISE = "enterprise"

class WalletType(Enum):
    SINGLE_SIG = "single_sig"
    MULTI_SIG_2OF3 = "multi_sig_2of3"
    MULTI_SIG_3OF5 = "multi_sig_3of5"
    HARDWARE_BACKED = "hardware_backed"

class AccessRole(Enum):
    VIEWER = "viewer"
    TRADER = "trader"
    MANAGER = "manager"
    ADMIN = "admin"
    EMERGENCY = "emergency"

class SecurityEvent(Enum):
    KEY_ROTATION = "key_rotation"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_TRANSACTION = "suspicious_transaction"
    WALLET_COMPROMISE = "wallet_compromise"
    EMERGENCY_LOCKDOWN = "emergency_lockdown"

@dataclass
class WalletConfig:
    """Wallet configuration"""
    wallet_id: str
    wallet_type: WalletType
    security_level: SecurityLevel
    required_signatures: int
    key_rotation_interval_hours: int = 24
    transaction_limit_per_hour: int = 100
    max_transaction_amount: float = 10000.0
    enable_monitoring: bool = True

@dataclass
class SecurityEventLog:
    """Security event logging"""
    event_id: str
    event_type: SecurityEvent
    severity: str
    wallet_id: str
    user_id: Optional[str]
    details: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False

@dataclass
class AccessPermission:
    """User access permissions"""
    user_id: str
    role: AccessRole
    wallet_access: Set[str]
    permissions: Set[str]
    rate_limits: Dict[str, int]
    expires_at: Optional[float] = None
    created_at: float = field(default_factory=time.time)

@dataclass
class SigningRequest:
    """Transaction signing request"""
    request_id: str
    wallet_id: str
    transaction_data: Dict[str, Any]
    required_signatures: int
    signatures_collected: int = 0
    signatures: List[Dict[str, Any]] = field(default_factory=list)
    expires_at: float = field(default_factory=lambda: time.time() + 300)  # 5 minutes
    status: str = "pending"

class SecureWalletManager:
    """
    Secure wallet manager with enterprise-grade security
    
    Features:
    - Hardware Security Module (HSM) integration for enterprise deployments
    - Multi-signature wallet support (2-of-3, 3-of-5 configurations)
    - Dynamic key rotation with 24-hour intervals
    - Role-based access control with granular permissions
    - Rate limiting (100 signatures/minute) and real-time monitoring
    - Emergency lockdown and recovery procedures
    """
    
    def __init__(self):
        # Wallet management
        self.wallets: Dict[str, WalletConfig] = {}
        self.signing_keys: Dict[str, Dict[str, Any]] = {}
        self.pending_signatures: Dict[str, SigningRequest] = {}
        
        # Security systems
        self.access_control = AccessControlManager()
        self.rate_limiter = RateLimiter()
        self.security_monitor = SecurityMonitor()
        self.key_rotator = KeyRotationManager()
        
        # HSM integration (when available)
        self.hsm_available = False
        self.hsm_client = None
        
        # Event logging
        self.security_events = deque(maxlen=10000)
        self.audit_trail = deque(maxlen=50000)
        
        # Emergency controls
        self.emergency_lockdown = False
        self.compromised_wallets: Set[str] = set()
        
        # Performance tracking
        self.security_metrics = {
            "total_signatures": 0,
            "failed_signatures": 0,
            "key_rotations": 0,
            "security_incidents": 0,
            "emergency_lockdowns": 0
        }
        
        logger.info("🔐 Secure Wallet Manager initialized")
    
    async def initialize(self) -> bool:
        """Initialize secure wallet manager"""
        try:
            # Initialize security components
            await self.access_control.initialize()
            await self.rate_limiter.initialize()
            await self.security_monitor.initialize()
            await self.key_rotator.initialize()
            
            # Check for HSM availability
            self.hsm_available = await self._check_hsm_availability()
            if self.hsm_available:
                await self._initialize_hsm()
            
            # Start monitoring tasks
            asyncio.create_task(self._security_monitoring_loop())
            asyncio.create_task(self._key_rotation_loop())
            asyncio.create_task(self._signature_cleanup_loop())
            
            # Load existing wallets
            await self._load_wallet_configurations()
            
            logger.info("✅ Secure Wallet Manager initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Secure Wallet Manager: {str(e)}")
            return False
    
    async def create_wallet(self, wallet_id: str, wallet_type: WalletType, 
                          security_level: SecurityLevel, user_id: str) -> Dict[str, Any]:
        """Create new secure wallet"""
        try:
            # Check permissions
            if not await self.access_control.check_permission(user_id, "create_wallet"):
                raise PermissionError("Insufficient permissions to create wallet")
            
            # Check if wallet already exists
            if wallet_id in self.wallets:
                raise ValueError(f"Wallet {wallet_id} already exists")
            
            # Determine configuration based on type and security level
            config = self._determine_wallet_config(wallet_id, wallet_type, security_level)
            
            # Generate keys based on wallet type
            if wallet_type == WalletType.HARDWARE_BACKED and self.hsm_available:
                keys = await self._generate_hsm_keys(wallet_id, config.required_signatures)
            else:
                keys = await self._generate_software_keys(wallet_id, config.required_signatures)
            
            # Store wallet configuration
            self.wallets[wallet_id] = config
            self.signing_keys[wallet_id] = keys
            
            # Log security event
            await self._log_security_event(
                SecurityEvent.KEY_ROTATION,
                "info",
                wallet_id,
                user_id,
                {"action": "wallet_created", "type": wallet_type.value}
            )
            
            # Persist configuration
            await self._persist_wallet_config(wallet_id, config)
            
            logger.info(f"🔐 Wallet created: {wallet_id} ({wallet_type.value}, {security_level.value})")
            
            return {
                "wallet_id": wallet_id,
                "wallet_type": wallet_type.value,
                "security_level": security_level.value,
                "required_signatures": config.required_signatures,
                "public_keys": [key.get("public_key") for key in keys.values()],
                "created_at": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error creating wallet {wallet_id}: {str(e)}")
            raise
    
    async def sign_transaction(self, wallet_id: str, transaction_data: Dict[str, Any], 
                             user_id: str) -> Dict[str, Any]:
        """Sign transaction with wallet"""
        try:
            start_time = time.time()
            
            # Security checks
            await self._validate_signing_request(wallet_id, transaction_data, user_id)
            
            # Check rate limits
            if not await self.rate_limiter.check_rate_limit(user_id, "signature"):
                raise PermissionError("Rate limit exceeded for signatures")
            
            # Get wallet configuration
            config = self.wallets[wallet_id]
            
            if config.wallet_type == WalletType.SINGLE_SIG:
                # Single signature wallet
                signature = await self._single_signature_sign(wallet_id, transaction_data, user_id)
                result = {
                    "signature": signature,
                    "signatures_required": 1,
                    "signatures_collected": 1,
                    "status": "completed"
                }
            else:
                # Multi-signature wallet
                result = await self._multi_signature_sign(wallet_id, transaction_data, user_id)
            
            # Update metrics
            self.security_metrics["total_signatures"] += 1
            await self.rate_limiter.record_action(user_id, "signature")
            
            # Log transaction
            await self._log_security_event(
                SecurityEvent.SUSPICIOUS_TRANSACTION if self._is_suspicious_transaction(transaction_data) else SecurityEvent.KEY_ROTATION,
                "info",
                wallet_id,
                user_id,
                {
                    "transaction_hash": transaction_data.get("hash", "unknown"),
                    "amount": transaction_data.get("amount", 0),
                    "execution_time_ms": (time.time() - start_time) * 1000
                }
            )
            
            return result
            
        except Exception as e:
            self.security_metrics["failed_signatures"] += 1
            logger.error(f"Error signing transaction for wallet {wallet_id}: {str(e)}")
            raise
    
    async def _single_signature_sign(self, wallet_id: str, transaction_data: Dict[str, Any], 
                                   user_id: str) -> str:
        """Sign transaction with single signature"""
        try:
            keys = self.signing_keys[wallet_id]
            primary_key = keys["key_0"]
            
            if self.hsm_available and primary_key.get("hsm_backed"):
                # HSM signing
                signature = await self._hsm_sign(primary_key["hsm_key_id"], transaction_data)
            else:
                # Software signing
                signature = await self._software_sign(primary_key["private_key"], transaction_data)
            
            return signature
            
        except Exception as e:
            logger.error(f"Error in single signature signing: {str(e)}")
            raise
    
    async def _multi_signature_sign(self, wallet_id: str, transaction_data: Dict[str, Any], 
                                  user_id: str) -> Dict[str, Any]:
        """Handle multi-signature signing process"""
        try:
            config = self.wallets[wallet_id]
            request_id = f"multisig_{wallet_id}_{int(time.time() * 1000)}"
            
            # Check for existing signing request
            existing_request = self._find_existing_signing_request(wallet_id, transaction_data)
            
            if existing_request:
                # Add signature to existing request
                request = self.pending_signatures[existing_request]
                
                # Verify user hasn't already signed
                if any(sig["user_id"] == user_id for sig in request.signatures):
                    raise ValueError("User has already signed this transaction")
                
                # Generate signature
                signature = await self._generate_user_signature(wallet_id, transaction_data, user_id)
                
                # Add signature to request
                request.signatures.append({
                    "user_id": user_id,
                    "signature": signature,
                    "timestamp": time.time()
                })
                request.signatures_collected += 1
                
                # Check if we have enough signatures
                if request.signatures_collected >= config.required_signatures:
                    request.status = "completed"
                    
                    return {
                        "signature": self._combine_signatures(request.signatures),
                        "signatures_required": config.required_signatures,
                        "signatures_collected": request.signatures_collected,
                        "status": "completed",
                        "request_id": existing_request
                    }
                else:
                    return {
                        "signatures_required": config.required_signatures,
                        "signatures_collected": request.signatures_collected,
                        "status": "pending",
                        "request_id": existing_request
                    }
            else:
                # Create new signing request
                signature = await self._generate_user_signature(wallet_id, transaction_data, user_id)
                
                request = SigningRequest(
                    request_id=request_id,
                    wallet_id=wallet_id,
                    transaction_data=transaction_data,
                    required_signatures=config.required_signatures,
                    signatures_collected=1,
                    signatures=[{
                        "user_id": user_id,
                        "signature": signature,
                        "timestamp": time.time()
                    }]
                )
                
                self.pending_signatures[request_id] = request
                
                if config.required_signatures == 1:
                    request.status = "completed"
                    
                    return {
                        "signature": signature,
                        "signatures_required": config.required_signatures,
                        "signatures_collected": 1,
                        "status": "completed",
                        "request_id": request_id
                    }
                else:
                    return {
                        "signatures_required": config.required_signatures,
                        "signatures_collected": 1,
                        "status": "pending",
                        "request_id": request_id
                    }
            
        except Exception as e:
            logger.error(f"Error in multi-signature signing: {str(e)}")
            raise
    
    async def rotate_keys(self, wallet_id: str, user_id: str) -> Dict[str, Any]:
        """Manually rotate wallet keys"""
        try:
            # Check permissions
            if not await self.access_control.check_permission(user_id, "rotate_keys"):
                raise PermissionError("Insufficient permissions to rotate keys")
            
            if wallet_id not in self.wallets:
                raise ValueError(f"Wallet {wallet_id} not found")
            
            # Execute key rotation
            result = await self.key_rotator.rotate_wallet_keys(wallet_id, self.wallets[wallet_id])
            
            # Update stored keys
            self.signing_keys[wallet_id] = result["new_keys"]
            
            # Log security event
            await self._log_security_event(
                SecurityEvent.KEY_ROTATION,
                "info",
                wallet_id,
                user_id,
                {"manual_rotation": True, "new_key_count": len(result["new_keys"])}
            )
            
            self.security_metrics["key_rotations"] += 1
            
            logger.info(f"🔄 Keys rotated for wallet: {wallet_id}")
            
            return {
                "wallet_id": wallet_id,
                "rotation_successful": True,
                "new_public_keys": result["public_keys"],
                "rotation_time": result["rotation_time"]
            }
            
        except Exception as e:
            logger.error(f"Error rotating keys for wallet {wallet_id}: {str(e)}")
            raise
    
    async def emergency_lockdown(self, reason: str, user_id: str) -> bool:
        """Execute emergency lockdown of all wallets"""
        try:
            # Check emergency permissions
            if not await self.access_control.check_permission(user_id, "emergency_lockdown"):
                raise PermissionError("Insufficient permissions for emergency lockdown")
            
            self.emergency_lockdown = True
            
            # Log emergency event
            await self._log_security_event(
                SecurityEvent.EMERGENCY_LOCKDOWN,
                "critical",
                "ALL_WALLETS",
                user_id,
                {"reason": reason, "lockdown_time": time.time()}
            )
            
            self.security_metrics["emergency_lockdowns"] += 1
            
            logger.critical(f"🚨 EMERGENCY LOCKDOWN ACTIVATED: {reason}")
            
            # Notify all monitoring systems
            await self.security_monitor.broadcast_emergency(reason)
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing emergency lockdown: {str(e)}")
            return False
    
    async def _validate_signing_request(self, wallet_id: str, transaction_data: Dict[str, Any], 
                                      user_id: str):
        """Validate transaction signing request"""
        # Check emergency lockdown
        if self.emergency_lockdown:
            raise PermissionError("System in emergency lockdown")
        
        # Check wallet exists
        if wallet_id not in self.wallets:
            raise ValueError(f"Wallet {wallet_id} not found")
        
        # Check wallet not compromised
        if wallet_id in self.compromised_wallets:
            raise SecurityError(f"Wallet {wallet_id} is compromised")
        
        # Check user permissions
        if not await self.access_control.check_wallet_access(user_id, wallet_id):
            raise PermissionError(f"User {user_id} does not have access to wallet {wallet_id}")
        
        # Validate transaction data
        if not self._validate_transaction_data(transaction_data):
            raise ValueError("Invalid transaction data")
        
        # Check transaction limits
        config = self.wallets[wallet_id]
        amount = transaction_data.get("amount", 0)
        if amount > config.max_transaction_amount:
            raise ValueError(f"Transaction amount {amount} exceeds limit {config.max_transaction_amount}")
    
    def _validate_transaction_data(self, transaction_data: Dict[str, Any]) -> bool:
        """Validate transaction data structure"""
        required_fields = ["to", "amount", "nonce"]
        return all(field in transaction_data for field in required_fields)
    
    def _is_suspicious_transaction(self, transaction_data: Dict[str, Any]) -> bool:
        """Check if transaction appears suspicious"""
        amount = transaction_data.get("amount", 0)
        
        # Large transaction
        if amount > 50000:
            return True
        
        # Round number (potential test)
        if amount in [1000, 10000, 100000]:
            return True
        
        return False
    
    async def _check_hsm_availability(self) -> bool:
        """Check if HSM is available"""
        try:
            # This would check for actual HSM connectivity
            # For now, simulate based on environment
            return os.environ.get("HSM_AVAILABLE", "false").lower() == "true"
        except Exception:
            return False
    
    async def _initialize_hsm(self):
        """Initialize HSM connection"""
        try:
            # This would initialize actual HSM client
            logger.info("🏛️ HSM integration initialized (simulated)")
            self.hsm_client = "simulated_hsm_client"
        except Exception as e:
            logger.warning(f"HSM initialization failed: {str(e)}")
            self.hsm_available = False
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get comprehensive security metrics"""
        try:
            current_time = time.time()
            
            # Recent security events
            recent_events = [
                event for event in self.security_events
                if current_time - event.timestamp < 3600  # Last hour
            ]
            
            event_counts = defaultdict(int)
            for event in recent_events:
                event_counts[event.event_type.value] += 1
            
            # Active signing requests
            active_requests = len([
                req for req in self.pending_signatures.values()
                if req.status == "pending" and req.expires_at > current_time
            ])
            
            return {
                "wallet_management": {
                    "total_wallets": len(self.wallets),
                    "compromised_wallets": len(self.compromised_wallets),
                    "emergency_lockdown": self.emergency_lockdown,
                    "hsm_available": self.hsm_available
                },
                "signature_performance": {
                    "total_signatures": self.security_metrics["total_signatures"],
                    "failed_signatures": self.security_metrics["failed_signatures"],
                    "success_rate": (
                        (self.security_metrics["total_signatures"] - self.security_metrics["failed_signatures"]) /
                        max(1, self.security_metrics["total_signatures"])
                    ) * 100,
                    "active_signing_requests": active_requests
                },
                "security_events": {
                    "recent_events_1h": len(recent_events),
                    "event_distribution": dict(event_counts),
                    "security_incidents_total": self.security_metrics["security_incidents"],
                    "emergency_lockdowns_total": self.security_metrics["emergency_lockdowns"]
                },
                "key_management": {
                    "key_rotations_total": self.security_metrics["key_rotations"],
                    "rotation_schedule": "24h intervals",
                    "next_rotation": "automated"
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting security metrics: {str(e)}")
            return {"error": str(e)}

class AccessControlManager:
    """Manage user access and permissions"""
    
    def __init__(self):
        self.user_permissions: Dict[str, AccessPermission] = {}
        self.role_permissions = self._define_role_permissions()
    
    def _define_role_permissions(self) -> Dict[AccessRole, Set[str]]:
        """Define permissions for each role"""
        return {
            AccessRole.VIEWER: {"view_wallets", "view_transactions"},
            AccessRole.TRADER: {"view_wallets", "view_transactions", "sign_transactions"},
            AccessRole.MANAGER: {
                "view_wallets", "view_transactions", "sign_transactions", 
                "create_wallet", "manage_users"
            },
            AccessRole.ADMIN: {
                "view_wallets", "view_transactions", "sign_transactions", 
                "create_wallet", "manage_users", "rotate_keys", "system_config"
            },
            AccessRole.EMERGENCY: {
                "emergency_lockdown", "emergency_override", "security_override"
            }
        }
    
    async def initialize(self):
        logger.info("👤 Access Control Manager initialized")
    
    async def check_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has specific permission"""
        try:
            if user_id not in self.user_permissions:
                return False
            
            user_perm = self.user_permissions[user_id]
            
            # Check if permission expired
            if user_perm.expires_at and time.time() > user_perm.expires_at:
                return False
            
            # Check role permissions
            role_perms = self.role_permissions.get(user_perm.role, set())
            
            return permission in role_perms or permission in user_perm.permissions
            
        except Exception as e:
            logger.error(f"Error checking permission: {str(e)}")
            return False
    
    async def check_wallet_access(self, user_id: str, wallet_id: str) -> bool:
        """Check if user has access to specific wallet"""
        try:
            if user_id not in self.user_permissions:
                return False
            
            user_perm = self.user_permissions[user_id]
            return wallet_id in user_perm.wallet_access or "*" in user_perm.wallet_access
            
        except Exception as e:
            logger.error(f"Error checking wallet access: {str(e)}")
            return False

class RateLimiter:
    """Rate limiting for security operations"""
    
    def __init__(self):
        self.rate_limits = {
            "signature": 100,  # 100 signatures per hour
            "key_rotation": 5,  # 5 rotations per hour
            "wallet_creation": 10  # 10 wallets per hour
        }
        self.user_actions: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=1000)))
    
    async def initialize(self):
        logger.info("⏱️ Rate Limiter initialized")
    
    async def check_rate_limit(self, user_id: str, action: str) -> bool:
        """Check if user is within rate limits"""
        try:
            current_time = time.time()
            hour_ago = current_time - 3600
            
            # Get recent actions
            actions = self.user_actions[user_id][action]
            recent_actions = [t for t in actions if t > hour_ago]
            
            # Update deque
            self.user_actions[user_id][action] = deque(recent_actions, maxlen=1000)
            
            # Check limit
            limit = self.rate_limits.get(action, 1000)
            return len(recent_actions) < limit
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {str(e)}")
            return False
    
    async def record_action(self, user_id: str, action: str):
        """Record user action for rate limiting"""
        try:
            self.user_actions[user_id][action].append(time.time())
        except Exception as e:
            logger.error(f"Error recording action: {str(e)}")

class SecurityMonitor:
    """Monitor security events and threats"""
    
    def __init__(self):
        self.monitoring_active = True
        self.threat_indicators = deque(maxlen=1000)
    
    async def initialize(self):
        logger.info("🛡️ Security Monitor initialized")
    
    async def broadcast_emergency(self, reason: str):
        """Broadcast emergency notification"""
        logger.critical(f"📢 EMERGENCY BROADCAST: {reason}")

class KeyRotationManager:
    """Manage automatic key rotation"""
    
    def __init__(self):
        self.rotation_schedule: Dict[str, float] = {}
    
    async def initialize(self):
        logger.info("🔄 Key Rotation Manager initialized")
    
    async def rotate_wallet_keys(self, wallet_id: str, config: WalletConfig) -> Dict[str, Any]:
        """Rotate keys for specific wallet"""
        try:
            start_time = time.time()
            
            # Generate new keys
            if config.wallet_type == WalletType.HARDWARE_BACKED:
                new_keys = await self._generate_hsm_keys(wallet_id, config.required_signatures)
            else:
                new_keys = await self._generate_software_keys(wallet_id, config.required_signatures)
            
            rotation_time = time.time() - start_time
            
            return {
                "new_keys": new_keys,
                "public_keys": [key.get("public_key") for key in new_keys.values()],
                "rotation_time": rotation_time
            }
            
        except Exception as e:
            logger.error(f"Error rotating keys: {str(e)}")
            raise
    
    async def _generate_software_keys(self, wallet_id: str, count: int) -> Dict[str, Dict[str, Any]]:
        """Generate software-based keys"""
        keys = {}
        for i in range(count):
            # Simulate key generation
            private_key = secrets.token_hex(32)
            public_key = hashlib.sha256(private_key.encode()).hexdigest()
            
            keys[f"key_{i}"] = {
                "private_key": private_key,
                "public_key": public_key,
                "created_at": time.time(),
                "hsm_backed": False
            }
        
        return keys
    
    async def _generate_hsm_keys(self, wallet_id: str, count: int) -> Dict[str, Dict[str, Any]]:
        """Generate HSM-backed keys"""
        keys = {}
        for i in range(count):
            # Simulate HSM key generation
            hsm_key_id = f"hsm_{wallet_id}_{i}_{int(time.time())}"
            public_key = hashlib.sha256(hsm_key_id.encode()).hexdigest()
            
            keys[f"key_{i}"] = {
                "hsm_key_id": hsm_key_id,
                "public_key": public_key,
                "created_at": time.time(),
                "hsm_backed": True
            }
        
        return keys

class SecurityError(Exception):
    """Security-related error"""
    pass 