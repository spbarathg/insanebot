#!/usr/bin/env python3
"""
Validation script for Solana trading bot deployment.
Tests monitoring, failure scenarios, and memory usage.
"""
import os
import sys
import time
import json
import logging
import asyncio
import argparse
import requests
import psutil
from pathlib import Path

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent))

# Import project modules
from src.utils.config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("validator")

class DeploymentValidator:
    """Validates the trading bot deployment"""
    
    def __init__(self):
        self.grafana_url = "http://localhost:3000"
        self.prometheus_url = "http://localhost:9090"
        self.loki_url = "http://localhost:3100"
        self.trading_bot_url = "http://localhost:8000"
        self.results = {
            "monitoring": {},
            "failure_tests": {},
            "performance": {}
        }
        
    async def validate_all(self):
        """Run all validation tests"""
        logger.info("Starting deployment validation...")
        
        # Check monitoring services
        self.check_monitoring_services()
        
        # Check Grafana dashboards
        self.check_grafana_dashboards()
        
        # Check Prometheus metrics
        self.check_prometheus_metrics()
        
        # Run failure tests
        await self.run_failure_tests()
        
        # Measure performance
        await self.measure_performance()
        
        # Print summary
        self.print_summary()
        
    def check_monitoring_services(self):
        """Check if monitoring services are running"""
        logger.info("Checking monitoring services...")
        
        services = {
            "Grafana": self.grafana_url,
            "Prometheus": self.prometheus_url,
            "Loki": self.loki_url,
            "Trading Bot": self.trading_bot_url
        }
        
        for name, url in services.items():
            try:
                response = requests.get(f"{url}/", timeout=5)
                if response.status_code < 400:
                    logger.info(f"✅ {name} is accessible")
                    self.results["monitoring"][name] = "PASS"
                else:
                    logger.error(f"❌ {name} returned status code {response.status_code}")
                    self.results["monitoring"][name] = f"FAIL: Status {response.status_code}"
            except Exception as e:
                logger.error(f"❌ Cannot connect to {name}: {str(e)}")
                self.results["monitoring"][name] = f"FAIL: {str(e)}"
                
    def check_grafana_dashboards(self):
        """Check if Grafana dashboards are properly configured"""
        logger.info("Checking Grafana dashboards...")
        
        try:
            # Try to authenticate with default credentials
            auth = ("admin", "admin")
            response = requests.get(f"{self.grafana_url}/api/dashboards", auth=auth, timeout=5)
            
            if response.status_code == 200:
                dashboards = response.json()
                logger.info(f"Found {len(dashboards)} dashboards")
                self.results["monitoring"]["Grafana Dashboards"] = f"PASS: {len(dashboards)} dashboards found"
            else:
                logger.warning(f"Could not get dashboards: {response.status_code}")
                self.results["monitoring"]["Grafana Dashboards"] = f"WARNING: Status {response.status_code}"
        except Exception as e:
            logger.error(f"❌ Error checking Grafana dashboards: {str(e)}")
            self.results["monitoring"]["Grafana Dashboards"] = f"FAIL: {str(e)}"
            
    def check_prometheus_metrics(self):
        """Check if Prometheus is collecting expected metrics"""
        logger.info("Checking Prometheus metrics...")
        
        expected_metrics = [
            "up",
            "container_memory_usage_bytes",
            "container_cpu_usage_seconds_total",
            "ai_model_predictions_total",
            "ai_model_accuracy"
        ]
        
        for metric in expected_metrics:
            try:
                response = requests.get(
                    f"{self.prometheus_url}/api/v1/query",
                    params={"query": metric},
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data["data"]["result"]:
                        logger.info(f"✅ Metric '{metric}' is available")
                        self.results["monitoring"][f"Metric: {metric}"] = "PASS"
                    else:
                        logger.warning(f"⚠️ Metric '{metric}' returned no data")
                        self.results["monitoring"][f"Metric: {metric}"] = "WARNING: No data"
                else:
                    logger.error(f"❌ Failed to query metric '{metric}': {response.status_code}")
                    self.results["monitoring"][f"Metric: {metric}"] = f"FAIL: Status {response.status_code}"
            except Exception as e:
                logger.error(f"❌ Error querying metric '{metric}': {str(e)}")
                self.results["monitoring"][f"Metric: {metric}"] = f"FAIL: {str(e)}"
                
    async def run_failure_tests(self):
        """Run failure scenario tests"""
        logger.info("Running failure scenario tests...")
        
        # Test 1: LLM timeout
        logger.info("Testing LLM timeout scenario...")
        try:
            # Check health endpoint
            response = requests.get(f"{self.trading_bot_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Bot health endpoint is accessible")
                self.results["failure_tests"]["Bot Health"] = "PASS"
                
                # Test token analysis with timeout
                start_time = time.time()
                test_response = requests.post(
                    f"{self.trading_bot_url}/api/analyze",
                    json={"token": "TokenWithLongProcessingTime"},
                    timeout=10
                )
                
                elapsed = time.time() - start_time
                if test_response.status_code == 200:
                    if elapsed < settings.LLM_INFERENCE_TIMEOUT + 2:
                        logger.info(f"✅ Timeout test passed: {elapsed:.2f}s (limit: {settings.LLM_INFERENCE_TIMEOUT}s)")
                        self.results["failure_tests"]["LLM Timeout"] = f"PASS: {elapsed:.2f}s"
                    else:
                        logger.warning(f"⚠️ Timeout test questionable: {elapsed:.2f}s (limit: {settings.LLM_INFERENCE_TIMEOUT}s)")
                        self.results["failure_tests"]["LLM Timeout"] = f"WARNING: {elapsed:.2f}s"
                else:
                    logger.error(f"❌ Timeout test failed: {test_response.status_code}")
                    self.results["failure_tests"]["LLM Timeout"] = f"FAIL: Status {test_response.status_code}"
            else:
                logger.error(f"❌ Bot health check failed: {response.status_code}")
                self.results["failure_tests"]["Bot Health"] = f"FAIL: Status {response.status_code}"
        except Exception as e:
            logger.error(f"❌ Error during failure tests: {str(e)}")
            self.results["failure_tests"]["Error"] = f"FAIL: {str(e)}"
            
    async def measure_performance(self):
        """Measure system performance during load"""
        logger.info("Measuring system performance...")
        
        try:
            # Get current resource usage
            process = psutil.Process()
            
            # Memory usage
            memory_info = process.memory_info()
            memory_usage_mb = memory_info.rss / (1024 * 1024)
            logger.info(f"Memory usage: {memory_usage_mb:.2f} MB")
            self.results["performance"]["Memory Usage"] = f"{memory_usage_mb:.2f} MB"
            
            # CPU usage
            cpu_percent = process.cpu_percent(interval=1.0)
            logger.info(f"CPU usage: {cpu_percent:.2f}%")
            self.results["performance"]["CPU Usage"] = f"{cpu_percent:.2f}%"
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            logger.info(f"Disk read: {disk_io.read_bytes / (1024*1024):.2f} MB")
            logger.info(f"Disk write: {disk_io.write_bytes / (1024*1024):.2f} MB")
            self.results["performance"]["Disk Read"] = f"{disk_io.read_bytes / (1024*1024):.2f} MB"
            self.results["performance"]["Disk Write"] = f"{disk_io.write_bytes / (1024*1024):.2f} MB"
            
            # Network I/O
            net_io = psutil.net_io_counters()
            logger.info(f"Network sent: {net_io.bytes_sent / (1024*1024):.2f} MB")
            logger.info(f"Network received: {net_io.bytes_recv / (1024*1024):.2f} MB")
            self.results["performance"]["Network Sent"] = f"{net_io.bytes_sent / (1024*1024):.2f} MB"
            self.results["performance"]["Network Received"] = f"{net_io.bytes_recv / (1024*1024):.2f} MB"
            
        except Exception as e:
            logger.error(f"❌ Error measuring performance: {str(e)}")
            self.results["performance"]["Error"] = f"FAIL: {str(e)}"
            
    def print_summary(self):
        """Print validation summary"""
        logger.info("\n" + "="*50)
        logger.info("DEPLOYMENT VALIDATION SUMMARY")
        logger.info("="*50)
        
        # Count results
        total = 0
        passed = 0
        warnings = 0
        failed = 0
        
        for category, tests in self.results.items():
            logger.info(f"\n{category.upper()}:")
            for test, result in tests.items():
                total += 1
                if result.startswith("PASS"):
                    passed += 1
                    logger.info(f"✅ {test}: {result}")
                elif result.startswith("WARNING"):
                    warnings += 1
                    logger.info(f"⚠️ {test}: {result}")
                else:
                    failed += 1
                    logger.info(f"❌ {test}: {result}")
        
        # Calculate percentage
        readiness = (passed / total) * 100 if total > 0 else 0
        
        logger.info("\n" + "="*50)
        logger.info(f"RESULTS: {passed}/{total} tests passed ({readiness:.1f}% readiness)")
        if warnings > 0:
            logger.info(f"WARNINGS: {warnings} tests have warnings")
        if failed > 0:
            logger.info(f"FAILURES: {failed} tests failed")
            
        logger.info("="*50)
        
        # Save results to file
        with open("validation_results.json", "w") as f:
            json.dump({
                "results": self.results,
                "summary": {
                    "total": total,
                    "passed": passed,
                    "warnings": warnings,
                    "failed": failed,
                    "readiness": readiness
                }
            }, f, indent=2)
            
        logger.info(f"Results saved to validation_results.json")
        
        # Return overall success
        return failed == 0

async def main():
    parser = argparse.ArgumentParser(description="Validate Solana trading bot deployment")
    args = parser.parse_args()
    
    validator = DeploymentValidator()
    await validator.validate_all()

if __name__ == "__main__":
    asyncio.run(main()) 