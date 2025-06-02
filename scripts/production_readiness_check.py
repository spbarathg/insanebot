#!/usr/bin/env python3
"""
Enhanced Ant Bot - Production Readiness Validation

Comprehensive validation script to verify system readiness for production deployment.
Checks all critical components, dependencies, security, performance, and operational aspects.
"""

import os
import sys
import time
import json
import asyncio
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import importlib.util

# Color codes for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


@dataclass
class CheckResult:
    """Result of a readiness check."""
    name: str
    status: str  # "PASS", "FAIL", "WARN"
    message: str
    details: Optional[str] = None
    execution_time: float = 0.0
    severity: str = "info"  # "critical", "high", "medium", "low", "info"


class ProductionReadinessChecker:
    """Comprehensive production readiness validation."""
    
    def __init__(self):
        self.results: List[CheckResult] = []
        self.start_time = time.time()
        self.logger = self._setup_logging()
        self.critical_failures = 0
        self.high_failures = 0
        self.warnings = 0
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for readiness checks."""
        logger = logging.getLogger("production_readiness")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _run_check(self, check_name: str, check_func, severity: str = "medium"):
        """Run a single check and record results."""
        start_time = time.time()
        try:
            result = check_func()
            execution_time = time.time() - start_time
            
            if isinstance(result, tuple):
                status, message, details = result
            elif isinstance(result, bool):
                status = "PASS" if result else "FAIL"
                message = f"{check_name} {'passed' if result else 'failed'}"
                details = None
            else:
                status = "PASS"
                message = str(result)
                details = None
            
            check_result = CheckResult(
                name=check_name,
                status=status,
                message=message,
                details=details,
                execution_time=execution_time,
                severity=severity
            )
            
            self.results.append(check_result)
            
            # Update failure counters
            if status == "FAIL":
                if severity == "critical":
                    self.critical_failures += 1
                elif severity == "high":
                    self.high_failures += 1
            elif status == "WARN":
                self.warnings += 1
            
            return check_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            check_result = CheckResult(
                name=check_name,
                status="FAIL",
                message=f"Check failed with exception: {str(e)}",
                details=str(e),
                execution_time=execution_time,
                severity=severity
            )
            self.results.append(check_result)
            
            if severity == "critical":
                self.critical_failures += 1
            elif severity == "high":
                self.high_failures += 1
                
            return check_result
    
    def check_python_version(self) -> Tuple[str, str, Optional[str]]:
        """Check Python version compatibility."""
        version = sys.version_info
        required_major, required_minor = 3, 8
        
        if version.major < required_major or (version.major == required_major and version.minor < required_minor):
            return "FAIL", f"Python {version.major}.{version.minor} detected, requires >= {required_major}.{required_minor}", None
        
        return "PASS", f"Python {version.major}.{version.minor}.{version.micro} is compatible", None
    
    def check_required_files(self) -> Tuple[str, str, Optional[str]]:
        """Check for required project files."""
        required_files = [
            "requirements.txt",
            "docker-compose.prod.yml",
            "Dockerfile",
            "pytest.ini",
            "env.template",
            "launch_production.sh"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            return "FAIL", f"Missing required files: {', '.join(missing_files)}", None
        
        return "PASS", "All required files present", None
    
    def check_environment_variables(self) -> Tuple[str, str, Optional[str]]:
        """Check for required environment variables."""
        required_env_vars = [
            "DB_HOST", "DB_PORT", "DB_NAME", "DB_USERNAME", "DB_PASSWORD",
            "REDIS_HOST", "REDIS_PORT",
            "HELIUS_API_KEY", "JUPITER_API_KEY",
            "SOLANA_RPC_URL", "PRIVATE_KEY",
            "JWT_SECRET"
        ]
        
        missing_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            return "WARN", f"Missing environment variables: {', '.join(missing_vars)}", "Check env.template for required variables"
        
        return "PASS", "All required environment variables set", None
    
    def check_dependencies(self) -> Tuple[str, str, Optional[str]]:
        """Check if all dependencies are installable."""
        try:
            # Test import of critical dependencies
            critical_imports = [
                "aiohttp", "asyncpg", "redis", "cryptography",
                "pandas", "numpy", "fastapi", "pydantic",
                "pytest", "prometheus_client"
            ]
            
            missing_imports = []
            for module in critical_imports:
                try:
                    importlib.import_module(module)
                except ImportError:
                    missing_imports.append(module)
            
            if missing_imports:
                return "FAIL", f"Missing critical dependencies: {', '.join(missing_imports)}", "Run: pip install -r requirements.txt"
            
            return "PASS", "All critical dependencies available", None
            
        except Exception as e:
            return "FAIL", f"Dependency check failed: {str(e)}", None
    
    def check_test_coverage(self) -> Tuple[str, str, Optional[str]]:
        """Check test coverage and quality."""
        test_dirs = ["tests/unit", "tests/integration", "tests/load"]
        
        # Check if test directories exist
        missing_dirs = [d for d in test_dirs if not os.path.exists(d)]
        if missing_dirs:
            return "WARN", f"Missing test directories: {', '.join(missing_dirs)}", "Some test types may not be available"
        
        # Count test files
        test_files = []
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                test_files.extend(list(Path(test_dir).glob("**/test_*.py")))
        
        if len(test_files) < 10:
            return "WARN", f"Only {len(test_files)} test files found", "Consider adding more comprehensive tests"
        
        return "PASS", f"Found {len(test_files)} test files across test suites", None
    
    def check_docker_configuration(self) -> Tuple[str, str, Optional[str]]:
        """Check Docker configuration."""
        try:
            # Check if Docker is available
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return "FAIL", "Docker not available", "Install Docker for containerized deployment"
            
            # Check docker-compose
            result = subprocess.run(
                ["docker-compose", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return "WARN", "Docker Compose not available", "Install docker-compose for easier container management"
            
            return "PASS", "Docker and Docker Compose available", None
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return "FAIL", "Docker not available or not responding", None
    
    def check_security_configuration(self) -> Tuple[str, str, Optional[str]]:
        """Check security configuration."""
        security_issues = []
        
        # Check for default secrets
        jwt_secret = os.getenv("JWT_SECRET", "")
        if not jwt_secret or len(jwt_secret) < 32:
            security_issues.append("JWT_SECRET not set or too short")
        
        # Check private key security
        private_key = os.getenv("PRIVATE_KEY", "")
        if not private_key:
            security_issues.append("PRIVATE_KEY not set")
        
        # Check for SSL/TLS configuration
        if not os.getenv("SSL_CERT_PATH") and not os.getenv("SSL_KEY_PATH"):
            security_issues.append("SSL certificates not configured")
        
        if security_issues:
            return "WARN", f"Security issues: {'; '.join(security_issues)}", "Review security configuration"
        
        return "PASS", "Security configuration looks good", None
    
    def check_monitoring_configuration(self) -> Tuple[str, str, Optional[str]]:
        """Check monitoring and observability setup."""
        monitoring_files = [
            "monitoring/prometheus.yml",
            "monitoring/alert_rules.yml",
            "monitoring/grafana"
        ]
        
        missing_files = [f for f in monitoring_files if not os.path.exists(f)]
        
        if missing_files:
            return "WARN", f"Missing monitoring files: {', '.join(missing_files)}", "Some monitoring capabilities may be limited"
        
        return "PASS", "Monitoring configuration complete", None
    
    def check_backup_configuration(self) -> Tuple[str, str, Optional[str]]:
        """Check backup and disaster recovery setup."""
        backup_script = "scripts/automated_backup.py"
        dr_docs = "docs/runbooks/disaster_recovery.md"
        
        issues = []
        if not os.path.exists(backup_script):
            issues.append("Backup script not found")
        
        if not os.path.exists(dr_docs):
            issues.append("Disaster recovery documentation missing")
        
        if issues:
            return "WARN", f"Backup/DR issues: {'; '.join(issues)}", "Backup and DR setup incomplete"
        
        return "PASS", "Backup and disaster recovery configured", None
    
    def check_performance_configuration(self) -> Tuple[str, str, Optional[str]]:
        """Check performance and scalability configuration."""
        load_tests = Path("tests/load")
        
        if not load_tests.exists():
            return "WARN", "Load tests not found", "Performance validation may be limited"
        
        load_test_files = list(load_tests.glob("test_*.py"))
        if len(load_test_files) < 1:
            return "WARN", "No load test files found", "Add load tests for performance validation"
        
        return "PASS", f"Found {len(load_test_files)} load test files", None
    
    def check_database_migration(self) -> Tuple[str, str, Optional[str]]:
        """Check database schema and migrations."""
        # This would check if database migrations are present and valid
        # For now, just check if migration directory exists
        
        migration_patterns = [
            "migrations/",
            "alembic/",
            "schema.sql",
            "database/"
        ]
        
        found_migrations = [p for p in migration_patterns if os.path.exists(p)]
        
        if not found_migrations:
            return "WARN", "No database migration files found", "Consider adding database schema management"
        
        return "PASS", f"Database schema management found: {', '.join(found_migrations)}", None
    
    def check_logging_configuration(self) -> Tuple[str, str, Optional[str]]:
        """Check logging configuration."""
        # Check if logging directory exists and is writable
        log_dir = Path("logs")
        if not log_dir.exists():
            try:
                log_dir.mkdir()
                return "PASS", "Logging directory created", None
            except PermissionError:
                return "FAIL", "Cannot create logs directory - permission denied", None
        
        # Check if directory is writable
        test_file = log_dir / "test_write"
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            test_file.unlink()
            return "PASS", "Logging directory is writable", None
        except PermissionError:
            return "FAIL", "Logs directory not writable", None
    
    def check_configuration_validation(self) -> Tuple[str, str, Optional[str]]:
        """Check configuration file validation."""
        config_files = [
            "config/trading_config.yaml",
            "config/risk_config.yaml",
            "config/system_config.yaml"
        ]
        
        found_configs = [f for f in config_files if os.path.exists(f)]
        
        if len(found_configs) < len(config_files) / 2:
            return "WARN", f"Only {len(found_configs)} of {len(config_files)} config files found", "Some features may use defaults"
        
        return "PASS", f"Configuration files found: {len(found_configs)}/{len(config_files)}", None
    
    def check_resource_limits(self) -> Tuple[str, str, Optional[str]]:
        """Check system resource requirements."""
        try:
            import psutil
            
            # Check memory
            memory = psutil.virtual_memory()
            if memory.total < 4 * 1024 * 1024 * 1024:  # 4GB
                return "WARN", f"Low memory: {memory.total // (1024**3)}GB available", "Recommended: 4GB+ for production"
            
            # Check CPU cores
            cpu_count = psutil.cpu_count()
            if cpu_count < 2:
                return "WARN", f"Low CPU cores: {cpu_count}", "Recommended: 2+ cores for production"
            
            # Check disk space
            disk = psutil.disk_usage('/')
            free_gb = disk.free // (1024**3)
            if free_gb < 10:
                return "WARN", f"Low disk space: {free_gb}GB free", "Recommended: 20GB+ free space"
            
            return "PASS", f"Resources OK: {memory.total // (1024**3)}GB RAM, {cpu_count} CPUs, {free_gb}GB free", None
            
        except ImportError:
            return "WARN", "Cannot check system resources (psutil not available)", None
    
    def run_all_checks(self):
        """Run all production readiness checks."""
        print(f"{Colors.BOLD}{Colors.BLUE}🚀 Enhanced Ant Bot - Production Readiness Check{Colors.END}\n")
        print(f"{Colors.CYAN}Starting comprehensive validation...{Colors.END}\n")
        
        # Critical checks (must pass)
        print(f"{Colors.BOLD}🔥 CRITICAL CHECKS{Colors.END}")
        self._run_check("Python Version", self.check_python_version, "critical")
        self._run_check("Required Files", self.check_required_files, "critical")
        self._run_check("Dependencies", self.check_dependencies, "critical")
        self._run_check("Docker Configuration", self.check_docker_configuration, "critical")
        
        # High priority checks
        print(f"\n{Colors.BOLD}⚡ HIGH PRIORITY CHECKS{Colors.END}")
        self._run_check("Environment Variables", self.check_environment_variables, "high")
        self._run_check("Security Configuration", self.check_security_configuration, "high")
        self._run_check("Test Coverage", self.check_test_coverage, "high")
        self._run_check("Logging Configuration", self.check_logging_configuration, "high")
        
        # Medium priority checks
        print(f"\n{Colors.BOLD}📊 OPERATIONAL CHECKS{Colors.END}")
        self._run_check("Monitoring Configuration", self.check_monitoring_configuration, "medium")
        self._run_check("Backup Configuration", self.check_backup_configuration, "medium")
        self._run_check("Performance Configuration", self.check_performance_configuration, "medium")
        self._run_check("Database Migration", self.check_database_migration, "medium")
        
        # Low priority checks
        print(f"\n{Colors.BOLD}🔧 ADDITIONAL CHECKS{Colors.END}")
        self._run_check("Configuration Validation", self.check_configuration_validation, "low")
        self._run_check("Resource Limits", self.check_resource_limits, "low")
        
        # Display results
        self._display_results()
        
        # Generate report
        self._generate_report()
        
        return self._calculate_readiness_score()
    
    def _display_results(self):
        """Display check results in a formatted way."""
        print(f"\n{Colors.BOLD}📋 RESULTS SUMMARY{Colors.END}")
        print("=" * 80)
        
        # Group results by status
        passed = [r for r in self.results if r.status == "PASS"]
        failed = [r for r in self.results if r.status == "FAIL"]
        warnings = [r for r in self.results if r.status == "WARN"]
        
        # Display summary
        print(f"{Colors.GREEN}✅ PASSED: {len(passed)}{Colors.END}")
        print(f"{Colors.RED}❌ FAILED: {len(failed)}{Colors.END}")
        print(f"{Colors.YELLOW}⚠️  WARNINGS: {len(warnings)}{Colors.END}")
        print()
        
        # Display detailed results
        for result in self.results:
            if result.status == "PASS":
                icon = f"{Colors.GREEN}✅{Colors.END}"
            elif result.status == "FAIL":
                icon = f"{Colors.RED}❌{Colors.END}"
            else:
                icon = f"{Colors.YELLOW}⚠️{Colors.END}"
            
            severity_color = {
                "critical": Colors.RED,
                "high": Colors.MAGENTA,
                "medium": Colors.YELLOW,
                "low": Colors.CYAN,
                "info": Colors.WHITE
            }.get(result.severity, Colors.WHITE)
            
            print(f"{icon} {severity_color}[{result.severity.upper()}]{Colors.END} {result.name}")
            print(f"    {result.message}")
            if result.details:
                print(f"    💡 {Colors.CYAN}{result.details}{Colors.END}")
            print(f"    ⏱️  {result.execution_time:.3f}s")
            print()
    
    def _calculate_readiness_score(self) -> int:
        """Calculate overall readiness score (0-100)."""
        total_checks = len(self.results)
        if total_checks == 0:
            return 0
        
        # Weighted scoring
        score = 100
        
        # Critical failures are heavily penalized
        score -= self.critical_failures * 25
        
        # High priority failures
        score -= self.high_failures * 15
        
        # Warnings reduce score moderately
        score -= self.warnings * 5
        
        # Ensure score doesn't go below 0
        score = max(0, score)
        
        return score
    
    def _generate_report(self):
        """Generate detailed readiness report."""
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "total_execution_time": time.time() - self.start_time,
            "summary": {
                "total_checks": len(self.results),
                "passed": len([r for r in self.results if r.status == "PASS"]),
                "failed": len([r for r in self.results if r.status == "FAIL"]),
                "warnings": len([r for r in self.results if r.status == "WARN"]),
                "critical_failures": self.critical_failures,
                "high_failures": self.high_failures
            },
            "readiness_score": self._calculate_readiness_score(),
            "checks": [
                {
                    "name": r.name,
                    "status": r.status,
                    "severity": r.severity,
                    "message": r.message,
                    "details": r.details,
                    "execution_time": r.execution_time
                }
                for r in self.results
            ]
        }
        
        # Save report
        report_file = f"production_readiness_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"📊 Detailed report saved to: {Colors.CYAN}{report_file}{Colors.END}")
        
        return report_data
    
    def get_recommendations(self) -> List[str]:
        """Get prioritized recommendations for improving readiness."""
        recommendations = []
        
        # Critical issues first
        critical_failures = [r for r in self.results if r.status == "FAIL" and r.severity == "critical"]
        for failure in critical_failures:
            recommendations.append(f"🔥 CRITICAL: Fix {failure.name} - {failure.message}")
        
        # High priority issues
        high_failures = [r for r in self.results if r.status == "FAIL" and r.severity == "high"]
        for failure in high_failures:
            recommendations.append(f"⚡ HIGH: Address {failure.name} - {failure.message}")
        
        # Warnings with details
        warnings_with_details = [r for r in self.results if r.status == "WARN" and r.details]
        for warning in warnings_with_details:
            recommendations.append(f"⚠️  IMPROVE: {warning.name} - {warning.details}")
        
        return recommendations


def main():
    """Main function to run production readiness checks."""
    checker = ProductionReadinessChecker()
    
    try:
        score = checker.run_all_checks()
        
        print(f"\n{Colors.BOLD}🎯 PRODUCTION READINESS SCORE: {score}/100{Colors.END}")
        
        if score >= 90:
            print(f"{Colors.GREEN}{Colors.BOLD}🎉 EXCELLENT! System is production-ready!{Colors.END}")
            exit_code = 0
        elif score >= 75:
            print(f"{Colors.YELLOW}{Colors.BOLD}👍 GOOD! Minor improvements recommended.{Colors.END}")
            exit_code = 0
        elif score >= 60:
            print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  OK! Some issues need attention.{Colors.END}")
            exit_code = 1
        else:
            print(f"{Colors.RED}{Colors.BOLD}❌ NOT READY! Critical issues must be resolved.{Colors.END}")
            exit_code = 2
        
        # Display recommendations
        recommendations = checker.get_recommendations()
        if recommendations:
            print(f"\n{Colors.BOLD}📝 RECOMMENDATIONS{Colors.END}")
            print("-" * 50)
            for i, rec in enumerate(recommendations[:10], 1):  # Show top 10
                print(f"{i}. {rec}")
            
            if len(recommendations) > 10:
                print(f"... and {len(recommendations) - 10} more (see full report)")
        
        print(f"\n{Colors.CYAN}Total execution time: {time.time() - checker.start_time:.2f}s{Colors.END}")
        
        return exit_code
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}❌ Readiness check interrupted by user{Colors.END}")
        return 130
    except Exception as e:
        print(f"\n{Colors.RED}❌ Readiness check failed with error: {str(e)}{Colors.END}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)