"""
Health check and recovery system module.
"""
import time
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from enum import Enum
import json

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class HealthCheck:
    def __init__(self, name: str, check_func: callable, interval: int = 60):
        """Initialize health check with name and check function."""
        self.name = name
        self.check_func = check_func
        self.interval = interval
        self.last_check = None
        self.last_status = ServiceStatus.UNKNOWN
        self.last_error = None
        self.consecutive_failures = 0

    async def run_check(self) -> ServiceStatus:
        """Run the health check and return status."""
        try:
            result = await self.check_func()
            self.last_check = datetime.now()
            self.last_status = ServiceStatus.HEALTHY if result else ServiceStatus.UNHEALTHY
            self.last_error = None
            self.consecutive_failures = 0
            return self.last_status
        except Exception as e:
            self.last_check = datetime.now()
            self.last_status = ServiceStatus.UNHEALTHY
            self.last_error = str(e)
            self.consecutive_failures += 1
            logger.error(f"Health check {self.name} failed: {str(e)}")
            return self.last_status

class RecoveryAction:
    def __init__(self, name: str, action_func: callable, max_attempts: int = 3):
        """Initialize recovery action with name and action function."""
        self.name = name
        self.action_func = action_func
        self.max_attempts = max_attempts
        self.attempts = 0
        self.last_attempt = None
        self.last_error = None

    async def execute(self) -> bool:
        """Execute the recovery action."""
        if self.attempts >= self.max_attempts:
            logger.error(f"Recovery action {self.name} exceeded max attempts")
            return False

        try:
            await self.action_func()
            self.attempts = 0
            self.last_attempt = datetime.now()
            self.last_error = None
            return True
        except Exception as e:
            self.attempts += 1
            self.last_attempt = datetime.now()
            self.last_error = str(e)
            logger.error(f"Recovery action {self.name} failed: {str(e)}")
            return False

class HealthMonitor:
    def __init__(self):
        """Initialize health monitor."""
        self.checks: Dict[str, HealthCheck] = {}
        self.recovery_actions: Dict[str, List[RecoveryAction]] = {}
        self.status_history: List[Dict[str, Any]] = []
        self.max_history = 1000
        self._lock = asyncio.Lock()

    def add_check(self, check: HealthCheck) -> None:
        """Add a health check."""
        self.checks[check.name] = check

    def add_recovery_action(self, check_name: str, action: RecoveryAction) -> None:
        """Add a recovery action for a health check."""
        if check_name not in self.recovery_actions:
            self.recovery_actions[check_name] = []
        self.recovery_actions[check_name].append(action)

    async def run_checks(self) -> Dict[str, ServiceStatus]:
        """Run all health checks."""
        results = {}
        async with self._lock:
            for check in self.checks.values():
                status = await check.run_check()
                results[check.name] = status

                # Record status in history
                self.status_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "check": check.name,
                    "status": status.value,
                    "error": check.last_error
                })

                # Trim history if too long
                if len(self.status_history) > self.max_history:
                    self.status_history = self.status_history[-self.max_history:]

                # Trigger recovery if needed
                if status == ServiceStatus.UNHEALTHY:
                    await self._trigger_recovery(check.name)

        return results

    async def _trigger_recovery(self, check_name: str) -> None:
        """Trigger recovery actions for a failed check."""
        if check_name not in self.recovery_actions:
            return

        for action in self.recovery_actions[check_name]:
            if await action.execute():
                logger.info(f"Recovery action {action.name} succeeded for {check_name}")
                break
            else:
                logger.warning(f"Recovery action {action.name} failed for {check_name}")

    def get_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return {
            "timestamp": datetime.now().isoformat(),
            "checks": {
                name: {
                    "status": check.last_status.value,
                    "last_check": check.last_check.isoformat() if check.last_check else None,
                    "error": check.last_error,
                    "consecutive_failures": check.consecutive_failures
                }
                for name, check in self.checks.items()
            }
        }

    def get_history(self, check_name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get status history, optionally filtered by check name."""
        history = self.status_history
        if check_name:
            history = [h for h in history if h["check"] == check_name]
        return history[-limit:]

# Initialize global instance
health_monitor = HealthMonitor()

# Example health checks
async def check_rpc_connection():
    """Check RPC connection health."""
    # Implement actual RPC connection check
    return True

async def check_wallet_balance():
    """Check wallet balance health."""
    # Implement actual wallet balance check
    return True

async def check_market_data():
    """Check market data health."""
    # Implement actual market data check
    return True

# Example recovery actions
async def reconnect_rpc():
    """Recover RPC connection."""
    # Implement actual RPC reconnection
    pass

async def refresh_wallet():
    """Recover wallet connection."""
    # Implement actual wallet refresh
    pass

async def refresh_market_data():
    """Recover market data connection."""
    # Implement actual market data refresh
    pass

# Initialize health checks and recovery actions
health_monitor.add_check(HealthCheck("rpc_connection", check_rpc_connection))
health_monitor.add_check(HealthCheck("wallet_balance", check_wallet_balance))
health_monitor.add_check(HealthCheck("market_data", check_market_data))

health_monitor.add_recovery_action("rpc_connection", RecoveryAction("reconnect_rpc", reconnect_rpc))
health_monitor.add_recovery_action("wallet_balance", RecoveryAction("refresh_wallet", refresh_wallet))
health_monitor.add_recovery_action("market_data", RecoveryAction("refresh_market_data", refresh_market_data))

# Periodic health check task
async def run_health_checks():
    """Periodic task to run health checks."""
    while True:
        try:
            await health_monitor.run_checks()
        except Exception as e:
            logger.error(f"Health check task failed: {str(e)}")
        await asyncio.sleep(60)  # Run every minute 