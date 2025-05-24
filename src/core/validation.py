"""
Comprehensive input validation for the Solana trading bot.
Prevents invalid trades, security vulnerabilities, and data corruption.
"""
import re
import json
from typing import Any, Dict, List, Optional, Union, Tuple
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import base58

class ValidationError(Exception):
    """Raised when validation fails"""
    pass

class SecurityLevel(Enum):
    """Security levels for validation"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Result of validation operation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_value: Any = None
    security_level: SecurityLevel = SecurityLevel.LOW

class AddressValidator:
    """Validates Solana addresses and tokens"""
    
    # Known malicious/scam token addresses (blacklist)
    BLACKLISTED_TOKENS = {
        "ScamToken1111111111111111111111111111111",
        "FakeUSDC1111111111111111111111111111111111",
        "RugPull111111111111111111111111111111111",
    }
    
    # Known safe token addresses (whitelist)
    WHITELISTED_TOKENS = {
        "So11111111111111111111111111111111111111112",  # SOL
        "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
        "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
    }
    
    @staticmethod
    def is_valid_solana_address(address: str) -> bool:
        """Validate Solana address format"""
        try:
            if not address or not isinstance(address, str):
                return False
            
            # Solana addresses are 32-44 characters in base58
            if not 32 <= len(address) <= 44:
                return False
            
            # Try to parse as Solana public key using solders
            try:
                from solders.pubkey import Pubkey
                Pubkey.from_string(address)
                return True
            except Exception:
                # Fallback to base58 validation
                try:
                    import base58
                    decoded = base58.b58decode(address)
                    if len(decoded) != 32:  # Solana addresses are 32 bytes
                        return False
                    return True
                except Exception:
                    return False
                
        except Exception:
            return False
    
    @classmethod
    def validate_token_address(cls, address: str, allow_unknown: bool = True) -> ValidationResult:
        """Comprehensive token address validation"""
        errors = []
        warnings = []
        security_level = SecurityLevel.LOW
        
        # Basic format validation
        if not cls.is_valid_solana_address(address):
            errors.append(f"Invalid Solana address format: {address}")
            return ValidationResult(False, errors, warnings, security_level=SecurityLevel.CRITICAL)
        
        # Blacklist check
        if address in cls.BLACKLISTED_TOKENS:
            errors.append(f"Token address is blacklisted (known scam): {address}")
            return ValidationResult(False, errors, warnings, security_level=SecurityLevel.CRITICAL)
        
        # Whitelist check
        if address in cls.WHITELISTED_TOKENS:
            security_level = SecurityLevel.LOW
        elif not allow_unknown:
            warnings.append(f"Token address not in whitelist: {address}")
            security_level = SecurityLevel.MEDIUM
        else:
            warnings.append(f"Unknown token address - proceed with caution: {address}")
            security_level = SecurityLevel.HIGH
        
        return ValidationResult(True, errors, warnings, address, security_level)

class AmountValidator:
    """Validates trading amounts and financial values"""
    
    # Trading limits
    MIN_SOL_AMOUNT = Decimal('0.000001')  # 1 lamport
    MAX_SOL_AMOUNT = Decimal('1000')      # 1000 SOL max per trade
    MIN_USD_VALUE = Decimal('0.01')       # $0.01 minimum
    MAX_USD_VALUE = Decimal('100000')     # $100k maximum
    
    @staticmethod
    def validate_amount(amount: Union[str, int, float, Decimal], 
                       min_amount: Optional[Decimal] = None,
                       max_amount: Optional[Decimal] = None,
                       allow_zero: bool = False) -> ValidationResult:
        """Validate trading amount"""
        errors = []
        warnings = []
        security_level = SecurityLevel.LOW
        
        try:
            # Convert to Decimal for precise validation
            if isinstance(amount, str):
                # Remove any whitespace and validate string format
                amount = amount.strip()
                if not re.match(r'^[0-9]*\.?[0-9]+$', amount):
                    errors.append(f"Invalid amount format: {amount}")
                    return ValidationResult(False, errors, warnings, security_level=SecurityLevel.HIGH)
                
                amount_decimal = Decimal(amount)
            else:
                amount_decimal = Decimal(str(amount))
            
            # Check for negative amounts
            if amount_decimal < 0:
                errors.append(f"Amount cannot be negative: {amount_decimal}")
                return ValidationResult(False, errors, warnings, security_level=SecurityLevel.HIGH)
            
            # Check for zero amounts BEFORE minimum check
            if amount_decimal == 0:
                if not allow_zero:
                    errors.append("Amount cannot be zero")
                    return ValidationResult(False, errors, warnings, security_level=SecurityLevel.MEDIUM)
                else:
                    # Zero is allowed, return success
                    return ValidationResult(True, errors, warnings, amount_decimal, security_level)
            
            # Check minimum amount (skip if zero is allowed and amount is zero)
            min_amount = min_amount or AmountValidator.MIN_SOL_AMOUNT
            if amount_decimal < min_amount:
                errors.append(f"Amount {amount_decimal} is below minimum {min_amount}")
                return ValidationResult(False, errors, warnings, security_level=SecurityLevel.MEDIUM)
            
            # Check maximum amount
            max_amount = max_amount or AmountValidator.MAX_SOL_AMOUNT
            if amount_decimal > max_amount:
                errors.append(f"Amount {amount_decimal} exceeds maximum {max_amount}")
                return ValidationResult(False, errors, warnings, security_level=SecurityLevel.HIGH)
            
            # Warning for large amounts
            if amount_decimal > AmountValidator.MAX_SOL_AMOUNT * Decimal('0.1'):  # >10% of max
                warnings.append(f"Large amount detected: {amount_decimal}")
                security_level = SecurityLevel.MEDIUM
            
            return ValidationResult(True, errors, warnings, amount_decimal, security_level)
            
        except (InvalidOperation, ValueError) as e:
            errors.append(f"Invalid amount value: {amount} - {str(e)}")
            return ValidationResult(False, errors, warnings, security_level=SecurityLevel.HIGH)

class PercentageValidator:
    """Validates percentage values for slippage, fees, etc."""
    
    @staticmethod
    def validate_percentage(percentage: Union[str, int, float], 
                          min_pct: float = 0.0,
                          max_pct: float = 100.0,
                          allow_zero: bool = True) -> ValidationResult:
        """Validate percentage value"""
        errors = []
        warnings = []
        security_level = SecurityLevel.LOW
        
        try:
            # Convert to float
            if isinstance(percentage, str):
                percentage = percentage.strip().rstrip('%')
                pct_value = float(percentage)
            else:
                pct_value = float(percentage)
            
            # Check range
            if pct_value < min_pct:
                errors.append(f"Percentage {pct_value}% is below minimum {min_pct}%")
                return ValidationResult(False, errors, warnings, security_level=SecurityLevel.MEDIUM)
            
            if pct_value > max_pct:
                errors.append(f"Percentage {pct_value}% exceeds maximum {max_pct}%")
                return ValidationResult(False, errors, warnings, security_level=SecurityLevel.HIGH)
            
            # Check for zero
            if pct_value == 0 and not allow_zero:
                errors.append("Percentage cannot be zero")
                return ValidationResult(False, errors, warnings, security_level=SecurityLevel.MEDIUM)
            
            # Warnings for extreme values
            if pct_value > 50:
                warnings.append(f"High percentage value: {pct_value}%")
                security_level = SecurityLevel.MEDIUM
            
            return ValidationResult(True, errors, warnings, pct_value, security_level)
            
        except (ValueError, TypeError) as e:
            errors.append(f"Invalid percentage value: {percentage} - {str(e)}")
            return ValidationResult(False, errors, warnings, security_level=SecurityLevel.HIGH)

class SlippageValidator:
    """Validates slippage tolerance settings"""
    
    MIN_SLIPPAGE = 0.1   # 0.1%
    MAX_SLIPPAGE = 20.0  # 20%
    RECOMMENDED_MAX = 5.0  # 5%
    
    @classmethod
    def validate_slippage(cls, slippage: Union[str, int, float]) -> ValidationResult:
        """Validate slippage tolerance"""
        result = PercentageValidator.validate_percentage(
            slippage, 
            min_pct=cls.MIN_SLIPPAGE, 
            max_pct=cls.MAX_SLIPPAGE,
            allow_zero=False
        )
        
        if result.is_valid and result.sanitized_value > cls.RECOMMENDED_MAX:
            result.warnings.append(f"Slippage {result.sanitized_value}% is higher than recommended maximum {cls.RECOMMENDED_MAX}%")
            result.security_level = SecurityLevel.MEDIUM
        
        return result

class TradeParameterValidator:
    """Validates complete trade parameters"""
    
    @staticmethod
    def validate_trade_params(
        input_token: str,
        output_token: str, 
        amount: Union[str, int, float],
        slippage: Union[str, int, float],
        max_price_impact: Optional[Union[str, int, float]] = None
    ) -> ValidationResult:
        """Validate complete set of trade parameters"""
        errors = []
        warnings = []
        security_level = SecurityLevel.LOW
        
        # Validate input token
        input_result = AddressValidator.validate_token_address(input_token)
        if not input_result.is_valid:
            errors.extend(input_result.errors)
            return ValidationResult(False, errors, warnings, security_level=SecurityLevel.CRITICAL)
        
        errors.extend(input_result.errors)
        warnings.extend(input_result.warnings)
        security_level = max(security_level, input_result.security_level, key=lambda x: x.value)
        
        # Validate output token
        output_result = AddressValidator.validate_token_address(output_token)
        if not output_result.is_valid:
            errors.extend(output_result.errors)
            return ValidationResult(False, errors, warnings, security_level=SecurityLevel.CRITICAL)
        
        errors.extend(output_result.errors)
        warnings.extend(output_result.warnings)
        security_level = max(security_level, output_result.security_level, key=lambda x: x.value)
        
        # Check for same token swap
        if input_token == output_token:
            errors.append("Input and output tokens cannot be the same")
            return ValidationResult(False, errors, warnings, security_level=SecurityLevel.HIGH)
        
        # Validate amount
        amount_result = AmountValidator.validate_amount(amount)
        if not amount_result.is_valid:
            errors.extend(amount_result.errors)
            return ValidationResult(False, errors, warnings, security_level=SecurityLevel.HIGH)
        
        warnings.extend(amount_result.warnings)
        security_level = max(security_level, amount_result.security_level, key=lambda x: x.value)
        
        # Validate slippage
        slippage_result = SlippageValidator.validate_slippage(slippage)
        if not slippage_result.is_valid:
            errors.extend(slippage_result.errors)
            return ValidationResult(False, errors, warnings, security_level=SecurityLevel.HIGH)
        
        warnings.extend(slippage_result.warnings)
        security_level = max(security_level, slippage_result.security_level, key=lambda x: x.value)
        
        # Validate max price impact if provided
        if max_price_impact is not None:
            impact_result = PercentageValidator.validate_percentage(
                max_price_impact, min_pct=0.0, max_pct=50.0
            )
            if not impact_result.is_valid:
                errors.extend(impact_result.errors)
                return ValidationResult(False, errors, warnings, security_level=SecurityLevel.HIGH)
            
            warnings.extend(impact_result.warnings)
            security_level = max(security_level, impact_result.security_level, key=lambda x: x.value)
        
        # Prepare sanitized parameters
        sanitized_params = {
            "input_token": input_token,
            "output_token": output_token,
            "amount": amount_result.sanitized_value,
            "slippage": slippage_result.sanitized_value,
            "max_price_impact": impact_result.sanitized_value if max_price_impact else None
        }
        
        return ValidationResult(True, errors, warnings, sanitized_params, security_level)

class SecurityValidator:
    """Validates security-sensitive operations"""
    
    @staticmethod
    def validate_private_key(private_key: str) -> ValidationResult:
        """Validate private key format and security"""
        errors = []
        warnings = []
        security_level = SecurityLevel.CRITICAL
        
        if not private_key or not isinstance(private_key, str):
            errors.append("Private key cannot be empty")
            return ValidationResult(False, errors, warnings, security_level=security_level)
        
        # Check for demo/test keys
        if private_key in ["0000000000000000000000000000000000000000000000000000000000000000", 
                          "demo_key_for_testing", "test_key"]:
            errors.append("Cannot use demo/test private key in production")
            return ValidationResult(False, errors, warnings, security_level=security_level)
        
        # Validate format
        try:
            if private_key.startswith('[') and private_key.endswith(']'):
                # Array format
                key_array = json.loads(private_key)
                if not isinstance(key_array, list) or len(key_array) != 64:
                    errors.append("Invalid private key array format")
                    return ValidationResult(False, errors, warnings, security_level=security_level)
            elif len(private_key) == 88:
                # Base58 format
                base58.b58decode(private_key)
            elif private_key.startswith('0x') and len(private_key) == 66:
                # Hex format with prefix
                bytes.fromhex(private_key[2:])
            elif len(private_key) == 64:
                # Hex format without prefix
                bytes.fromhex(private_key)
            else:
                errors.append("Invalid private key format")
                return ValidationResult(False, errors, warnings, security_level=security_level)
                
        except Exception as e:
            errors.append(f"Private key validation failed: {str(e)}")
            return ValidationResult(False, errors, warnings, security_level=security_level)
        
        return ValidationResult(True, errors, warnings, "**REDACTED**", SecurityLevel.LOW)
    
    @staticmethod
    def validate_api_key(api_key: str, service_name: str) -> ValidationResult:
        """Validate API key format"""
        errors = []
        warnings = []
        security_level = SecurityLevel.HIGH
        
        if not api_key or not isinstance(api_key, str):
            errors.append(f"{service_name} API key cannot be empty")
            return ValidationResult(False, errors, warnings, security_level=security_level)
        
        # Check for demo keys
        demo_keys = ["abc123example_replace_with_real_api_key", "demo_key_for_testing", "test_key"]
        if api_key in demo_keys:
            errors.append(f"Cannot use demo {service_name} API key in production")
            return ValidationResult(False, errors, warnings, security_level=security_level)
        
        # Basic format validation
        if len(api_key) < 8:
            errors.append(f"{service_name} API key appears too short")
            return ValidationResult(False, errors, warnings, security_level=security_level)
        
        return ValidationResult(True, errors, warnings, f"{api_key[:8]}...", SecurityLevel.LOW)

class InputSanitizer:
    """Sanitizes user inputs to prevent injection attacks"""
    
    @staticmethod
    def sanitize_string(input_str: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        if not isinstance(input_str, str):
            return str(input_str)
        
        # Remove null bytes and control characters
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', input_str)
        
        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        # Remove potentially dangerous patterns
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'data:',
            r'vbscript:',
        ]
        
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        return sanitized.strip()

# Main validation interface
class TradingValidator:
    """Main interface for all trading validations"""
    
    def __init__(self, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode
        self.address_validator = AddressValidator()
        self.amount_validator = AmountValidator()
        self.trade_validator = TradeParameterValidator()
        self.security_validator = SecurityValidator()
        self.sanitizer = InputSanitizer()
    
    def validate_trade(self, trade_params: Dict[str, Any]) -> ValidationResult:
        """Validate complete trade parameters"""
        try:
            required_fields = ['input_token', 'output_token', 'amount']
            missing_fields = [field for field in required_fields if field not in trade_params]
            
            if missing_fields:
                return ValidationResult(
                    False, 
                    [f"Missing required fields: {missing_fields}"], 
                    [],
                    security_level=SecurityLevel.HIGH
                )
            
            # Extract and sanitize parameters
            sanitized_params = {}
            for key, value in trade_params.items():
                if isinstance(value, str):
                    sanitized_params[key] = self.sanitizer.sanitize_string(value)
                else:
                    sanitized_params[key] = value
            
            # Validate trade parameters
            return self.trade_validator.validate_trade_params(
                input_token=sanitized_params['input_token'],
                output_token=sanitized_params['output_token'],
                amount=sanitized_params['amount'],
                slippage=sanitized_params.get('slippage', 1.0),
                max_price_impact=sanitized_params.get('max_price_impact')
            )
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return ValidationResult(
                False, 
                [f"Validation failed: {str(e)}"], 
                [],
                security_level=SecurityLevel.CRITICAL
            )
    
    def validate_credentials(self, credentials: Dict[str, str]) -> ValidationResult:
        """Validate all credentials for production use"""
        if self.simulation_mode:
            return ValidationResult(True, [], ["Running in simulation mode"], security_level=SecurityLevel.LOW)
        
        errors = []
        warnings = []
        security_level = SecurityLevel.LOW
        
        # Validate private key
        if 'private_key' in credentials:
            key_result = self.security_validator.validate_private_key(credentials['private_key'])
            if not key_result.is_valid:
                errors.extend(key_result.errors)
                return ValidationResult(False, errors, warnings, security_level=SecurityLevel.CRITICAL)
            warnings.extend(key_result.warnings)
        
        # Validate API keys
        api_keys = {
            'helius_api_key': 'Helius',
            'jupiter_api_key': 'Jupiter'
        }
        
        for key_name, service_name in api_keys.items():
            if key_name in credentials:
                api_result = self.security_validator.validate_api_key(credentials[key_name], service_name)
                if not api_result.is_valid:
                    errors.extend(api_result.errors)
                warnings.extend(api_result.warnings)
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, security_level=security_level) 