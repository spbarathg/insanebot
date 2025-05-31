"""
IP Whitelist - Network Access Control System

Manages allowed IP addresses and network ranges for secure access control
to the trading system components.
"""

import logging
import ipaddress
import time
import asyncio
from typing import Set, Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import re

logger = logging.getLogger(__name__)

@dataclass
class IPEntry:
    """Information about a whitelisted IP or network"""
    ip_address: str
    network: Union[ipaddress.IPv4Network, ipaddress.IPv6Network]
    added_at: float
    added_by: str
    description: str = ""
    expires_at: Optional[float] = None
    is_active: bool = True
    access_count: int = 0
    last_access: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AccessAttempt:
    """Record of an access attempt"""
    ip_address: str
    timestamp: float
    allowed: bool
    reason: str
    user_agent: Optional[str] = None
    endpoint: Optional[str] = None

class IPWhitelist:
    """
    IP address whitelist management system
    
    Features:
    - IPv4 and IPv6 support
    - Network range support (CIDR notation)
    - Temporary and permanent entries
    - Access logging and monitoring
    - Automatic cleanup of expired entries
    - Emergency lockdown mode
    """
    
    def __init__(self):
        # Whitelist storage
        self._whitelist: Dict[str, IPEntry] = {}
        self._networks: List[IPEntry] = []  # For CIDR ranges
        
        # Access tracking
        self.access_history: deque = deque(maxlen=10000)
        self.blocked_attempts: deque = deque(maxlen=1000)
        self.access_stats = defaultdict(int)
        
        # Security settings
        self.default_expiry_hours = 24
        self.emergency_lockdown = False
        self.allow_localhost = True
        self.allow_private_networks = False
        
        # Statistics
        self.total_checks = 0
        self.allowed_count = 0
        self.blocked_count = 0
        
        # Initialize with localhost if enabled
        if self.allow_localhost:
            self._add_default_entries()
        
        logger.info("🛡️ IPWhitelist initialized - Network access control active")
    
    def _add_default_entries(self):
        """Add default safe entries"""
        localhost_entries = [
            ("127.0.0.1/32", "IPv4 localhost"),
            ("::1/128", "IPv6 localhost"),
        ]
        
        for ip_range, description in localhost_entries:
            try:
                self.add_ip_range(
                    ip_range=ip_range,
                    added_by="system",
                    description=f"Default: {description}",
                    permanent=True
                )
            except Exception as e:
                logger.warning(f"Failed to add default entry {ip_range}: {e}")
    
    def add_ip_address(self, ip_address: str, added_by: str, 
                      description: str = "", hours_valid: Optional[int] = None,
                      permanent: bool = False) -> bool:
        """Add a single IP address to the whitelist"""
        try:
            # Validate IP address
            ip_obj = ipaddress.ip_address(ip_address)
            
            # Create network object for single IP
            if isinstance(ip_obj, ipaddress.IPv4Address):
                network = ipaddress.IPv4Network(f"{ip_address}/32")
            else:
                network = ipaddress.IPv6Network(f"{ip_address}/128")
            
            # Calculate expiry
            expires_at = None
            if not permanent:
                hours = hours_valid or self.default_expiry_hours
                expires_at = time.time() + (hours * 3600)
            
            # Create entry
            entry = IPEntry(
                ip_address=ip_address,
                network=network,
                added_at=time.time(),
                added_by=added_by,
                description=description,
                expires_at=expires_at
            )
            
            self._whitelist[ip_address] = entry
            
            logger.info(f"✅ Added IP to whitelist: {ip_address} by {added_by}")
            if expires_at:
                logger.info(f"   Expires: {time.ctime(expires_at)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add IP {ip_address}: {e}")
            return False
    
    def add_ip_range(self, ip_range: str, added_by: str,
                    description: str = "", hours_valid: Optional[int] = None,
                    permanent: bool = False) -> bool:
        """Add an IP range (CIDR notation) to the whitelist"""
        try:
            # Validate network range
            network = ipaddress.ip_network(ip_range, strict=False)
            
            # Calculate expiry
            expires_at = None
            if not permanent:
                hours = hours_valid or self.default_expiry_hours
                expires_at = time.time() + (hours * 3600)
            
            # Create entry
            entry = IPEntry(
                ip_address=ip_range,
                network=network,
                added_at=time.time(),
                added_by=added_by,
                description=description,
                expires_at=expires_at
            )
            
            self._networks.append(entry)
            
            logger.info(f"✅ Added IP range to whitelist: {ip_range} by {added_by}")
            if expires_at:
                logger.info(f"   Expires: {time.ctime(expires_at)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add IP range {ip_range}: {e}")
            return False
    
    def remove_ip(self, ip_address: str, removed_by: str) -> bool:
        """Remove an IP address from the whitelist"""
        try:
            if ip_address in self._whitelist:
                del self._whitelist[ip_address]
                logger.info(f"🗑️ Removed IP from whitelist: {ip_address} by {removed_by}")
                return True
            
            # Check if it's a network range
            for i, entry in enumerate(self._networks):
                if entry.ip_address == ip_address:
                    del self._networks[i]
                    logger.info(f"🗑️ Removed IP range from whitelist: {ip_address} by {removed_by}")
                    return True
            
            logger.warning(f"IP not found in whitelist: {ip_address}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove IP {ip_address}: {e}")
            return False
    
    def is_allowed(self, ip_address: str, endpoint: str = None, 
                  user_agent: str = None) -> bool:
        """Check if an IP address is allowed access"""
        self.total_checks += 1
        
        try:
            # Emergency lockdown - deny all
            if self.emergency_lockdown:
                self._record_access_attempt(ip_address, False, "Emergency lockdown", 
                                          endpoint, user_agent)
                self.blocked_count += 1
                return False
            
            # Parse IP address
            try:
                ip_obj = ipaddress.ip_address(ip_address)
            except ValueError:
                self._record_access_attempt(ip_address, False, "Invalid IP format",
                                          endpoint, user_agent)
                self.blocked_count += 1
                return False
            
            # Check if it's a private network and allowed
            if self.allow_private_networks and ip_obj.is_private:
                self._record_access_attempt(ip_address, True, "Private network allowed",
                                          endpoint, user_agent)
                self.allowed_count += 1
                return True
            
            # Check exact IP matches
            if ip_address in self._whitelist:
                entry = self._whitelist[ip_address]
                
                # Check if entry is active and not expired
                if not entry.is_active:
                    self._record_access_attempt(ip_address, False, "IP entry inactive",
                                              endpoint, user_agent)
                    self.blocked_count += 1
                    return False
                
                if entry.expires_at and time.time() > entry.expires_at:
                    # Entry expired, remove it
                    del self._whitelist[ip_address]
                    self._record_access_attempt(ip_address, False, "IP entry expired",
                                              endpoint, user_agent)
                    self.blocked_count += 1
                    return False
                
                # Update access stats
                entry.access_count += 1
                entry.last_access = time.time()
                self.access_stats[ip_address] += 1
                
                self._record_access_attempt(ip_address, True, "Exact IP match",
                                          endpoint, user_agent)
                self.allowed_count += 1
                return True
            
            # Check network ranges
            for entry in list(self._networks):
                if not entry.is_active:
                    continue
                
                # Check if expired
                if entry.expires_at and time.time() > entry.expires_at:
                    self._networks.remove(entry)
                    continue
                
                # Check if IP is in network range
                if ip_obj in entry.network:
                    # Update access stats
                    entry.access_count += 1
                    entry.last_access = time.time()
                    self.access_stats[entry.ip_address] += 1
                    
                    self._record_access_attempt(ip_address, True, 
                                              f"Network range match: {entry.ip_address}",
                                              endpoint, user_agent)
                    self.allowed_count += 1
                    return True
            
            # No match found
            self._record_access_attempt(ip_address, False, "IP not whitelisted",
                                      endpoint, user_agent)
            self.blocked_count += 1
            return False
            
        except Exception as e:
            logger.error(f"Error checking IP {ip_address}: {e}")
            self._record_access_attempt(ip_address, False, f"Check error: {str(e)}",
                                      endpoint, user_agent)
            self.blocked_count += 1
            return False
    
    def _record_access_attempt(self, ip_address: str, allowed: bool, reason: str,
                             endpoint: str = None, user_agent: str = None):
        """Record an access attempt"""
        attempt = AccessAttempt(
            ip_address=ip_address,
            timestamp=time.time(),
            allowed=allowed,
            reason=reason,
            user_agent=user_agent,
            endpoint=endpoint
        )
        
        if allowed:
            self.access_history.append(attempt)
        else:
            self.blocked_attempts.append(attempt)
            logger.warning(f"🚫 Blocked access from {ip_address}: {reason}")
    
    def enable_emergency_lockdown(self, enabled_by: str, reason: str = ""):
        """Enable emergency lockdown mode (deny all access)"""
        self.emergency_lockdown = True
        logger.critical(f"🚨 EMERGENCY LOCKDOWN ACTIVATED by {enabled_by}: {reason}")
    
    def disable_emergency_lockdown(self, disabled_by: str):
        """Disable emergency lockdown mode"""
        self.emergency_lockdown = False
        logger.info(f"✅ Emergency lockdown disabled by {disabled_by}")
    
    def cleanup_expired_entries(self) -> int:
        """Remove expired entries from the whitelist"""
        current_time = time.time()
        removed_count = 0
        
        # Clean up single IPs
        for ip_address in list(self._whitelist.keys()):
            entry = self._whitelist[ip_address]
            if entry.expires_at and current_time > entry.expires_at:
                del self._whitelist[ip_address]
                removed_count += 1
                logger.debug(f"🗑️ Removed expired IP: {ip_address}")
        
        # Clean up network ranges
        original_count = len(self._networks)
        self._networks = [entry for entry in self._networks 
                         if not (entry.expires_at and current_time > entry.expires_at)]
        removed_count += original_count - len(self._networks)
        
        if removed_count > 0:
            logger.info(f"🗑️ Cleaned up {removed_count} expired whitelist entries")
        
        return removed_count
    
    def get_whitelist_entries(self, include_expired: bool = False) -> List[IPEntry]:
        """Get all whitelist entries"""
        entries = []
        current_time = time.time()
        
        # Add single IPs
        for entry in self._whitelist.values():
            if include_expired or not entry.expires_at or entry.expires_at > current_time:
                entries.append(entry)
        
        # Add network ranges
        for entry in self._networks:
            if include_expired or not entry.expires_at or entry.expires_at > current_time:
                entries.append(entry)
        
        return entries
    
    def get_recent_access_attempts(self, count: int = 100, blocked_only: bool = False) -> List[AccessAttempt]:
        """Get recent access attempts"""
        if blocked_only:
            return list(self.blocked_attempts)[-count:]
        else:
            return list(self.access_history)[-count:]
    
    def get_access_stats(self) -> Dict[str, int]:
        """Get access statistics by IP"""
        return dict(self.access_stats)
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export whitelist configuration for backup"""
        entries = []
        
        for entry in self.get_whitelist_entries(include_expired=True):
            entries.append({
                "ip_address": entry.ip_address,
                "added_at": entry.added_at,
                "added_by": entry.added_by,
                "description": entry.description,
                "expires_at": entry.expires_at,
                "is_active": entry.is_active,
                "access_count": entry.access_count,
                "last_access": entry.last_access,
                "metadata": entry.metadata
            })
        
        return {
            "entries": entries,
            "settings": {
                "default_expiry_hours": self.default_expiry_hours,
                "allow_localhost": self.allow_localhost,
                "allow_private_networks": self.allow_private_networks,
                "emergency_lockdown": self.emergency_lockdown
            },
            "exported_at": time.time()
        }
    
    def import_configuration(self, config: Dict[str, Any], imported_by: str) -> bool:
        """Import whitelist configuration from backup"""
        try:
            # Import settings
            settings = config.get("settings", {})
            self.default_expiry_hours = settings.get("default_expiry_hours", 24)
            self.allow_localhost = settings.get("allow_localhost", True)
            self.allow_private_networks = settings.get("allow_private_networks", False)
            
            # Import entries
            entries = config.get("entries", [])
            imported_count = 0
            
            for entry_data in entries:
                ip_address = entry_data["ip_address"]
                
                # Determine if it's a single IP or range
                if "/" in ip_address:
                    success = self.add_ip_range(
                        ip_range=ip_address,
                        added_by=imported_by,
                        description=f"Imported: {entry_data.get('description', '')}",
                        permanent=entry_data.get("expires_at") is None
                    )
                else:
                    success = self.add_ip_address(
                        ip_address=ip_address,
                        added_by=imported_by,
                        description=f"Imported: {entry_data.get('description', '')}",
                        permanent=entry_data.get("expires_at") is None
                    )
                
                if success:
                    imported_count += 1
            
            logger.info(f"✅ Imported {imported_count} whitelist entries")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import whitelist configuration: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get whitelist status"""
        active_entries = len(self.get_whitelist_entries(include_expired=False))
        expired_entries = len(self.get_whitelist_entries(include_expired=True)) - active_entries
        
        return {
            "active_entries": active_entries,
            "expired_entries": expired_entries,
            "total_checks": self.total_checks,
            "allowed_count": self.allowed_count,
            "blocked_count": self.blocked_count,
            "success_rate": (self.allowed_count / max(1, self.total_checks)) * 100,
            "emergency_lockdown": self.emergency_lockdown,
            "allow_localhost": self.allow_localhost,
            "allow_private_networks": self.allow_private_networks,
            "recent_blocked_attempts": len(self.blocked_attempts)
        } 