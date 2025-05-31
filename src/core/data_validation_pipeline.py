"""
Data Quality Validation Pipeline - Lightweight Adversarial-Resistant Design

This module implements practical data quality validation with adversarial attack detection
optimized for local deployment environments.
"""

import asyncio
import time
import logging
import statistics
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AttackType(Enum):
    DATA_POISONING = "data_poisoning"
    SCHEMA_MANIPULATION = "schema_manipulation" 
    STATISTICAL_ANOMALY = "statistical_anomaly"
    TEMPORAL_MANIPULATION = "temporal_manipulation"

@dataclass
class ValidationResult:
    is_valid: bool
    quality_score: float
    issues: List[str] = field(default_factory=list)
    severity: ValidationSeverity = ValidationSeverity.INFO
    attack_indicators: List[AttackType] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

@dataclass
class DataProfile:
    """Statistical profile for detecting anomalies"""
    field_name: str
    mean: float
    std: float
    min_val: float
    max_val: float
    sample_count: int
    last_updated: float

class DataQualityValidator:
    """Lightweight data quality validation with adversarial detection"""
    
    def __init__(self):
        # Statistical baselines for anomaly detection
        self.data_profiles: Dict[str, DataProfile] = {}
        self.validation_history = deque(maxlen=1000)
        
        # Adversarial detection parameters
        self.anomaly_threshold = 3.0  # Standard deviations
        self.poisoning_detection_window = 100  # Recent samples to analyze
        self.schema_validation_rules = self._init_schema_rules()
        
        # Performance tracking
        self.validation_metrics = {
            "total_validations": 0,
            "failed_validations": 0,
            "attacks_detected": 0,
            "false_positives": 0
        }
        
    def _init_schema_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize schema validation rules for token data"""
        return {
            "price": {"type": float, "min": 0.0, "max": 1000000.0, "required": True},
            "volume_24h": {"type": float, "min": 0.0, "max": 1e12, "required": True},
            "liquidity": {"type": float, "min": 100.0, "max": 1e12, "required": True},
            "market_cap": {"type": float, "min": 1000.0, "max": 1e15, "required": False},
            "timestamp": {"type": float, "min": 0.0, "max": time.time() + 86400, "required": True},
            "token_address": {"type": str, "min_length": 32, "max_length": 44, "required": True}
        }
    
    async def validate_data(self, token_address: str, data: Dict[str, Any]) -> ValidationResult:
        """Comprehensive data validation with adversarial detection"""
        try:
            start_time = time.time()
            issues = []
            attack_indicators = []
            quality_score = 1.0
            
            # 1. Schema Validation
            schema_issues = await self._validate_schema(data)
            if schema_issues:
                issues.extend(schema_issues)
                quality_score -= 0.3
            
            # 2. Statistical Anomaly Detection
            anomaly_result = await self._detect_statistical_anomalies(token_address, data)
            if anomaly_result.get("anomalies"):
                issues.extend(anomaly_result["anomalies"])
                attack_indicators.extend(anomaly_result.get("attack_types", []))
                quality_score -= 0.4
            
            # 3. Temporal Consistency Check
            temporal_issues = await self._validate_temporal_consistency(data)
            if temporal_issues:
                issues.extend(temporal_issues)
                quality_score -= 0.2
            
            # 4. Cross-Field Consistency
            consistency_issues = await self._validate_cross_field_consistency(data)
            if consistency_issues:
                issues.extend(consistency_issues)
                quality_score -= 0.3
            
            # 5. Adversarial Pattern Detection
            adversarial_result = await self._detect_adversarial_patterns(token_address, data)
            if adversarial_result.get("detected"):
                issues.extend(adversarial_result["patterns"])
                attack_indicators.extend(adversarial_result.get("attack_types", []))
                quality_score -= 0.5
            
            # Update profiles with valid data
            if quality_score > 0.6:
                await self._update_data_profiles(token_address, data)
            
            # Determine severity
            severity = self._determine_severity(quality_score, attack_indicators)
            
            # Create validation result
            result = ValidationResult(
                is_valid=quality_score > 0.5,
                quality_score=max(0.0, quality_score),
                issues=issues,
                severity=severity,
                attack_indicators=attack_indicators
            )
            
            # Update metrics
            self.validation_metrics["total_validations"] += 1
            if not result.is_valid:
                self.validation_metrics["failed_validations"] += 1
            if attack_indicators:
                self.validation_metrics["attacks_detected"] += 1
            
            # Store validation history
            self.validation_history.append(result)
            
            validation_time = time.time() - start_time
            logger.debug(f"Data validation completed in {validation_time*1000:.1f}ms "
                        f"(Quality: {quality_score:.2f}, Issues: {len(issues)})")
            
            return result
            
        except Exception as e:
            logger.error(f"Data validation error: {str(e)}")
            return ValidationResult(
                is_valid=False,
                quality_score=0.0,
                issues=[f"Validation system error: {str(e)}"],
                severity=ValidationSeverity.CRITICAL
            )
    
    async def _validate_schema(self, data: Dict[str, Any]) -> List[str]:
        """Validate data against expected schema"""
        issues = []
        
        for field, rules in self.schema_validation_rules.items():
            if rules.get("required", False) and field not in data:
                issues.append(f"Missing required field: {field}")
                continue
            
            if field in data:
                value = data[field]
                expected_type = rules.get("type")
                
                # Type validation
                if expected_type and not isinstance(value, expected_type):
                    issues.append(f"Invalid type for {field}: expected {expected_type.__name__}")
                    continue
                
                # Range validation for numeric fields
                if isinstance(value, (int, float)):
                    if "min" in rules and value < rules["min"]:
                        issues.append(f"{field} below minimum: {value} < {rules['min']}")
                    if "max" in rules and value > rules["max"]:
                        issues.append(f"{field} above maximum: {value} > {rules['max']}")
                
                # String length validation
                if isinstance(value, str):
                    if "min_length" in rules and len(value) < rules["min_length"]:
                        issues.append(f"{field} too short: {len(value)} < {rules['min_length']}")
                    if "max_length" in rules and len(value) > rules["max_length"]:
                        issues.append(f"{field} too long: {len(value)} > {rules['max_length']}")
        
        return issues
    
    async def _detect_statistical_anomalies(self, token_address: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect statistical anomalies that could indicate data poisoning"""
        anomalies = []
        attack_types = []
        
        numeric_fields = ["price", "volume_24h", "liquidity", "market_cap"]
        
        for field in numeric_fields:
            if field not in data:
                continue
                
            value = data[field]
            profile_key = f"{token_address}_{field}"
            
            if profile_key in self.data_profiles:
                profile = self.data_profiles[profile_key]
                
                # Calculate z-score
                if profile.std > 0:
                    z_score = abs(value - profile.mean) / profile.std
                    
                    if z_score > self.anomaly_threshold:
                        anomalies.append(f"Statistical anomaly in {field}: z-score {z_score:.2f}")
                        
                        # Classify potential attack type
                        if z_score > 5.0:
                            attack_types.append(AttackType.DATA_POISONING)
                        else:
                            attack_types.append(AttackType.STATISTICAL_ANOMALY)
        
        return {
            "anomalies": anomalies,
            "attack_types": attack_types
        }
    
    async def _validate_temporal_consistency(self, data: Dict[str, Any]) -> List[str]:
        """Validate temporal consistency of data"""
        issues = []
        
        if "timestamp" in data:
            timestamp = data["timestamp"]
            current_time = time.time()
            
            # Check if timestamp is too far in the future
            if timestamp > current_time + 300:  # 5 minutes tolerance
                issues.append(f"Timestamp too far in future: {timestamp - current_time:.0f}s ahead")
            
            # Check if timestamp is too old
            if timestamp < current_time - 3600:  # 1 hour tolerance
                issues.append(f"Timestamp too old: {current_time - timestamp:.0f}s ago")
        
        return issues
    
    async def _validate_cross_field_consistency(self, data: Dict[str, Any]) -> List[str]:
        """Validate consistency between related fields"""
        issues = []
        
        # Market cap vs price consistency
        if all(field in data for field in ["price", "market_cap"]):
            price = data["price"]
            market_cap = data["market_cap"]
            
            # Basic sanity check: market cap should be reasonable relative to price
            if price > 0 and market_cap > 0:
                implied_supply = market_cap / price
                
                # Check for unreasonable token supply
                if implied_supply < 1000:  # Too few tokens
                    issues.append("Implied token supply suspiciously low")
                elif implied_supply > 1e15:  # Too many tokens
                    issues.append("Implied token supply suspiciously high")
        
        # Volume vs liquidity consistency
        if all(field in data for field in ["volume_24h", "liquidity"]):
            volume = data["volume_24h"]
            liquidity = data["liquidity"]
            
            # Volume shouldn't be dramatically higher than liquidity
            if volume > liquidity * 100:  # 100x liquidity is suspicious
                issues.append("Volume/liquidity ratio suspicious")
        
        return issues
    
    async def _detect_adversarial_patterns(self, token_address: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect patterns indicating coordinated adversarial attacks"""
        patterns = []
        attack_types = []
        
        # Check for coordinated data manipulation patterns
        recent_validations = [
            v for v in self.validation_history 
            if time.time() - v.timestamp < 300  # Last 5 minutes
        ]
        
        if len(recent_validations) > 20:  # Sufficient data for pattern analysis
            # Check for suspicious uniformity in data
            price_variations = []
            for validation in recent_validations[-10:]:  # Last 10 validations
                if validation.is_valid and hasattr(validation, 'data'):
                    # This would need to store original data in ValidationResult
                    pass
            
            # Pattern: Sudden spike in validation failures
            recent_failures = sum(1 for v in recent_validations if not v.is_valid)
            failure_rate = recent_failures / len(recent_validations)
            
            if failure_rate > 0.5:  # More than 50% failures
                patterns.append("High validation failure rate detected")
                attack_types.append(AttackType.SCHEMA_MANIPULATION)
        
        return {
            "detected": len(patterns) > 0,
            "patterns": patterns,
            "attack_types": attack_types
        }
    
    async def _update_data_profiles(self, token_address: str, data: Dict[str, Any]):
        """Update statistical profiles for normal data"""
        numeric_fields = ["price", "volume_24h", "liquidity", "market_cap"]
        
        for field in numeric_fields:
            if field not in data:
                continue
                
            value = data[field]
            profile_key = f"{token_address}_{field}"
            
            if profile_key in self.data_profiles:
                profile = self.data_profiles[profile_key]
                
                # Update running statistics
                n = profile.sample_count
                old_mean = profile.mean
                
                # Online algorithm for mean and variance
                new_mean = (n * old_mean + value) / (n + 1)
                new_variance = ((n - 1) * profile.std**2 + (value - old_mean) * (value - new_mean)) / n if n > 1 else 0
                
                profile.mean = new_mean
                profile.std = max(0.01, new_variance**0.5)  # Minimum std to avoid division by zero
                profile.min_val = min(profile.min_val, value)
                profile.max_val = max(profile.max_val, value)
                profile.sample_count += 1
                profile.last_updated = time.time()
            else:
                # Create new profile
                self.data_profiles[profile_key] = DataProfile(
                    field_name=field,
                    mean=value,
                    std=0.01,  # Small initial std
                    min_val=value,
                    max_val=value,
                    sample_count=1,
                    last_updated=time.time()
                )
    
    def _determine_severity(self, quality_score: float, attack_indicators: List[AttackType]) -> ValidationSeverity:
        """Determine severity based on quality score and attack indicators"""
        if AttackType.DATA_POISONING in attack_indicators:
            return ValidationSeverity.CRITICAL
        elif any(attack in attack_indicators for attack in [AttackType.SCHEMA_MANIPULATION]):
            return ValidationSeverity.ERROR
        elif quality_score < 0.3:
            return ValidationSeverity.ERROR
        elif quality_score < 0.6:
            return ValidationSeverity.WARNING
        else:
            return ValidationSeverity.INFO
    
    def get_validation_metrics(self) -> Dict[str, Any]:
        """Get validation system metrics"""
        total = self.validation_metrics["total_validations"]
        if total == 0:
            return {"error": "No validations performed yet"}
        
        return {
            "total_validations": total,
            "success_rate": 1.0 - (self.validation_metrics["failed_validations"] / total),
            "attack_detection_rate": self.validation_metrics["attacks_detected"] / total,
            "false_positive_rate": self.validation_metrics["false_positives"] / max(1, self.validation_metrics["attacks_detected"]),
            "active_profiles": len(self.data_profiles),
            "validation_history_size": len(self.validation_history)
        } 