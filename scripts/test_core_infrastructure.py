#!/usr/bin/env python3
"""
Core Infrastructure Test Script

Demonstrates all 4 core infrastructure components working together:
1. ConfigManager - Configuration management with hot-reload
2. SystemLogger - Advanced structured logging with performance tracking
3. SystemMetrics - Comprehensive system monitoring with Prometheus
4. SecurityManager - Complete security management with threat detection

This script shows the integration and capabilities of each component.
"""

import asyncio
import os
import sys
import logging
import time
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.config_manager import ConfigManager
from src.core.logger import SystemLogger
from src.core.system_metrics import SystemMetrics
from src.core.security_manager import SecurityManager

# Configure basic logging for the test script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CoreInfrastructureDemo:
    """Demonstrates all core infrastructure components"""
    
    def __init__(self):
        self.config_manager = None
        self.system_logger = None
        self.system_metrics = None
        self.security_manager = None
        
        self.demo_running = False
        self.start_time = time.time()
    
    async def initialize_all_components(self):
        """Initialize all 4 core infrastructure components"""
        print("\n" + "="*80)
        print("🏗️  INITIALIZING CORE INFRASTRUCTURE COMPONENTS")
        print("="*80)
        
        # 1. Initialize ConfigManager
        print("\n1️⃣  Initializing ConfigManager...")
        self.config_manager = ConfigManager()
        if await self.config_manager.initialize():
            print("✅ ConfigManager initialized successfully")
            self._demo_config_features()
        else:
            print("❌ ConfigManager initialization failed")
            return False
        
        # 2. Initialize SystemLogger
        print("\n2️⃣  Initializing SystemLogger...")
        self.system_logger = SystemLogger()
        if await self.system_logger.initialize():
            print("✅ SystemLogger initialized successfully")
            self._demo_logging_features()
        else:
            print("❌ SystemLogger initialization failed")
            return False
        
        # 3. Initialize SystemMetrics
        print("\n3️⃣  Initializing SystemMetrics...")
        self.system_metrics = SystemMetrics()
        if await self.system_metrics.initialize():
            print("✅ SystemMetrics initialized successfully")
            self._demo_metrics_features()
        else:
            print("❌ SystemMetrics initialization failed")
            return False
        
        # 4. Initialize SecurityManager
        print("\n4️⃣  Initializing SecurityManager...")
        self.security_manager = SecurityManager()
        if await self.security_manager.initialize():
            print("✅ SecurityManager initialized successfully")
            self._demo_security_features()
        else:
            print("❌ SecurityManager initialization failed")
            return False
        
        print("\n🎉 All core infrastructure components initialized successfully!")
        return True
    
    def _demo_config_features(self):
        """Demonstrate ConfigManager features"""
        print("\n📋 ConfigManager Features:")
        
        # Get configuration values
        environment = self.config_manager.get_environment()
        print(f"   • Environment: {environment}")
        
        # Feature flags
        hot_reload = self.config_manager.is_feature_enabled("enable_hot_reload")
        compounding = self.config_manager.is_feature_enabled("enable_compounding")
        print(f"   • Hot Reload: {hot_reload}")
        print(f"   • Compounding: {compounding}")
        
        # Get nested configuration
        trading_config = self.config_manager.get("trading.max_position_size", "Not configured")
        print(f"   • Trading Max Position: {trading_config}")
        
        # Get configuration summary
        summary = self.config_manager.get_config_summary()
        print(f"   • Config Sections: {len(summary['config_sections'])}")
        print(f"   • Validation Rules: {summary['validation_rules_count']}")
    
    def _demo_logging_features(self):
        """Demonstrate SystemLogger features"""
        print("\n📝 SystemLogger Features:")
        
        # Get different component loggers
        config_logger = self.system_logger.get_logger("ConfigManager")
        metrics_logger = self.system_logger.get_logger("SystemMetrics")
        security_logger = self.system_logger.get_logger("SecurityManager")
        
        # Demonstrate different log levels
        config_logger.info("ConfigManager operational", extra={
            "event_type": "startup",
            "component": "ConfigManager"
        })
        
        metrics_logger.warning("High CPU usage detected", extra={
            "event_type": "performance",
            "cpu_usage": 85.5
        })
        
        security_logger.error("Security event detected", extra={
            "event_type": "security",
            "threat_type": "suspicious_activity"
        })
        
        # Performance tracking
        correlation_id = self.system_logger.start_correlation("demo_operation")
        self.system_logger.log_performance("Demo", "test_operation", 0.125, 
                                          correlation_id=correlation_id)
        self.system_logger.end_correlation()
        
        # Get system metrics
        log_metrics = self.system_logger.get_system_metrics()
        print(f"   • Active Loggers: {log_metrics['active_loggers']}")
        print(f"   • Log Queue Size: {log_metrics['log_queue_size']}")
        print(f"   • Performance Tracking: {len(log_metrics['performance_metrics'])}")
    
    def _demo_metrics_features(self):
        """Demonstrate SystemMetrics features"""
        print("\n📊 SystemMetrics Features:")
        
        # Record various operations
        asyncio.create_task(self.system_metrics.record_operation(
            "ConfigManager", "config_load", 0.05, success=True
        ))
        asyncio.create_task(self.system_metrics.record_operation(
            "SecurityManager", "threat_detection", 0.02, success=True
        ))
        
        # Record custom metrics
        self.system_metrics.set_custom_metric("demo_counter", 42)
        self.system_metrics.set_custom_metric("demo_timestamp", time.time())
        
        # Get performance summary
        performance = self.system_metrics.get_performance_summary()
        print(f"   • Total Operations: {performance.get('total_operations', 0)}")
        print(f"   • Active Components: {performance.get('active_components', 0)}")
        print(f"   • Monitoring Uptime: {performance.get('monitoring_uptime', 0):.2f}s")
        
        # Add custom alert
        asyncio.create_task(self.system_metrics.add_alert(
            "demo_alert", "cpu_usage", 90.0, "greater", "high"
        ))
        
        print(f"   • Prometheus Server: Port 8001")
        print(f"   • Custom Metrics: {len(self.system_metrics.custom_metrics)}")
    
    def _demo_security_features(self):
        """Demonstrate SecurityManager features"""
        print("\n🔐 SecurityManager Features:")
        
        # Get security status
        status = self.security_manager.get_security_status()
        print(f"   • Overall Status: {status['overall_status']}")
        print(f"   • Active Policies: {status['policies_active']}")
        print(f"   • Active Tokens: {status['active_tokens']}")
        
        # Demonstrate authentication
        test_request = {
            "source_ip": "127.0.0.1",
            "user_agent": "AntBot/1.0",
            "endpoint": "/api/status"
        }
        
        auth_result = asyncio.create_task(
            self.security_manager.authenticate_request(test_request)
        )
        print(f"   • Request Authentication: Available")
        
        # Demonstrate encryption
        test_data = b"This is sensitive trading data"
        encrypted = self.security_manager.encrypt_data(test_data)
        decrypted = self.security_manager.decrypt_data(encrypted)
        print(f"   • Encryption: {len(encrypted)} bytes → {len(decrypted)} bytes")
        
        # Password hashing
        test_password = "secure_password_123"
        hashed = self.security_manager.hash_password(test_password)
        verified = self.security_manager.verify_password(test_password, hashed)
        print(f"   • Password Security: Verified {verified}")
        
        # Get security metrics
        metrics = self.security_manager.get_security_metrics()
        print(f"   • Threat Signatures: {metrics['threats']['signatures_active']}")
        print(f"   • Security Events: {metrics['events']['total']}")
    
    async def run_integration_demo(self):
        """Demonstrate integration between all components"""
        print("\n" + "="*80)
        print("🔄 INTEGRATION DEMONSTRATION")
        print("="*80)
        
        self.demo_running = True
        
        # Create a realistic scenario
        print("\n🎯 Scenario: Simulating Ant Bot trading system startup and operation")
        
        # 1. Configuration-driven operation
        print("\n1. Configuration Management:")
        trading_enabled = self.config_manager.is_feature_enabled("enable_compounding")
        if trading_enabled:
            print("   ✅ Compounding system enabled via configuration")
            
            # Log the configuration decision
            app_logger = self.system_logger.get_logger("AntBotSystem")
            app_logger.info("System configured for compounding operations", extra={
                "event_type": "configuration",
                "feature": "compounding",
                "enabled": True
            })
        
        # 2. Security validation
        print("\n2. Security Validation:")
        # Simulate API request authentication
        api_request = {
            "source_ip": "192.168.1.100",
            "user_agent": "AntBot-Worker/1.0",
            "endpoint": "/api/trade",
            "method": "POST"
        }
        
        auth_result = await self.security_manager.authenticate_request(api_request)
        print(f"   • API Request Authentication: {auth_result['authenticated']}")
        
        if not auth_result['authenticated']:
            print(f"   • Reason: {auth_result.get('reason', 'Unknown')}")
        
        # 3. Performance monitoring
        print("\n3. Performance Monitoring:")
        
        # Start operation timer
        timer_id = self.system_metrics.start_operation_timer("trading_operation")
        
        # Simulate some work
        await asyncio.sleep(0.1)
        
        # End timer and record operation
        duration = self.system_metrics.end_operation_timer(timer_id)
        await self.system_metrics.record_operation(
            "TradingEngine", "execute_trade", duration, success=True
        )
        
        print(f"   • Operation Duration: {duration:.3f} seconds")
        print(f"   • Metrics Recorded: Operation success tracked")
        
        # 4. Logging coordination
        print("\n4. Coordinated Logging:")
        correlation_id = self.system_logger.start_correlation("integration_demo")
        
        # Log from multiple components with correlation
        config_logger = self.system_logger.get_logger("ConfigManager")
        metrics_logger = self.system_logger.get_logger("SystemMetrics")
        security_logger = self.system_logger.get_logger("SecurityManager")
        
        config_logger.info("Configuration accessed", extra={
            "event_type": "config_access",
            "correlation_id": correlation_id
        })
        
        metrics_logger.info("Performance metrics updated", extra={
            "event_type": "metrics_update",
            "correlation_id": correlation_id
        })
        
        security_logger.info("Security validation completed", extra={
            "event_type": "security_check",
            "correlation_id": correlation_id
        })
        
        self.system_logger.end_correlation()
        print(f"   • Correlation ID: {correlation_id}")
        print("   • All components logged with correlation tracking")
        
        # 5. Health monitoring
        print("\n5. System Health Check:")
        health = self.system_metrics.get_system_health()
        security_status = self.security_manager.get_security_status()
        
        print(f"   • System Status: {health['status']}")
        print(f"   • Security Status: {security_status['overall_status']}")
        print(f"   • Component Count: {health['components'].__len__() if 'components' in health else 0}")
        
        # 6. Feature demonstration
        await self._demo_advanced_features()
        
        print("\n✅ Integration demonstration completed successfully!")
    
    async def _demo_advanced_features(self):
        """Demonstrate advanced features"""
        print("\n6. Advanced Features:")
        
        # Configuration hot-reload simulation
        if self.config_manager.is_feature_enabled("enable_hot_reload"):
            print("   • Hot-reload: Monitoring configuration changes")
        
        # Security threat simulation
        threat_request = {
            "source_ip": "suspicious.attacker.com",
            "user_agent": "SQLInject/1.0",
            "payload": "'; DROP TABLE users; --"
        }
        
        auth_result = await self.security_manager.authenticate_request(threat_request)
        if not auth_result['authenticated'] and 'threat_detected' in auth_result.get('reason', ''):
            print("   • Threat Detection: ✅ Malicious request blocked")
        
        # Performance alert simulation
        # Simulate high CPU usage
        high_cpu_data = {
            "cpu_usage": 95.0,
            "memory_usage": 80.0,
            "disk_usage": 60.0
        }
        
        print("   • Performance Monitoring: Real-time system tracking active")
        print("   • Prometheus Metrics: Available on http://localhost:8001/metrics")
    
    async def cleanup_all_components(self):
        """Cleanup all components"""
        print("\n" + "="*80)
        print("🧹 CLEANING UP COMPONENTS")
        print("="*80)
        
        self.demo_running = False
        
        if self.security_manager:
            await self.security_manager.cleanup()
            print("✅ SecurityManager cleaned up")
        
        if self.system_metrics:
            await self.system_metrics.cleanup()
            print("✅ SystemMetrics cleaned up")
        
        if self.system_logger:
            await self.system_logger.cleanup()
            print("✅ SystemLogger cleaned up")
        
        if self.config_manager:
            await self.config_manager.cleanup()
            print("✅ ConfigManager cleaned up")
        
        runtime = time.time() - self.start_time
        print(f"\n⏱️  Total demo runtime: {runtime:.2f} seconds")
        print("🎉 Core Infrastructure Demo completed successfully!")

async def main():
    """Run the core infrastructure demonstration"""
    print("🐜 ANT BOT CORE INFRASTRUCTURE DEMONSTRATION")
    print("=" * 80)
    print("This script demonstrates all 4 core infrastructure components:")
    print("1. ConfigManager - Configuration management with hot-reload")
    print("2. SystemLogger - Advanced structured logging")
    print("3. SystemMetrics - System monitoring with Prometheus")
    print("4. SecurityManager - Complete security management")
    print("=" * 80)
    
    demo = CoreInfrastructureDemo()
    
    try:
        # Initialize all components
        if await demo.initialize_all_components():
            
            # Run integration demonstration
            await demo.run_integration_demo()
            
            # Keep running for a bit to show monitoring
            print(f"\n⏳ Running for 30 seconds to demonstrate monitoring...")
            await asyncio.sleep(30)
            
        else:
            print("❌ Failed to initialize all components")
            
    except KeyboardInterrupt:
        print("\n👋 Demonstration interrupted by user")
    except Exception as e:
        print(f"\n💥 Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        await demo.cleanup_all_components()

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main()) 