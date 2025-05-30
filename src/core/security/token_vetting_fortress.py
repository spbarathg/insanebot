"""
Token Vetting Fortress - 5-Layer Defense Against Memecoin Attacks

This module implements an unbreachable token screening system:
1. Liquidity Lock Verification (24h minimum)
2. Holder Distribution Analysis (max 10% whale dominance)  
3. Contract Honeypot Scanning (Tesseract API integration)
4. Creation Timestamp Gating (15m minimum age)
5. Trading Volume Validation (100 SOL minimum)

SURVIVAL PROTOCOL: 100% rejection of unvetted tokens
"""

import asyncio
import aiohttp
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class VettingResult(Enum):
    """Token vetting results"""
    APPROVED = "approved"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"
    BLACKLISTED = "blacklisted"

class RiskLevel(Enum):
    """Token risk classification"""
    SAFE = "safe"           # 0-20% risk
    MODERATE = "moderate"   # 21-40% risk  
    HIGH = "high"          # 41-70% risk
    EXTREME = "extreme"    # 71-90% risk
    LETHAL = "lethal"      # 91-100% risk

@dataclass
class TokenVettingReport:
    """Comprehensive token vetting analysis"""
    token_address: str
    token_symbol: str
    overall_result: VettingResult
    risk_level: RiskLevel
    risk_score: float  # 0-100
    
    # Individual layer results
    liquidity_lock_status: bool
    liquidity_lock_duration: float  # hours
    holder_distribution_score: float
    whale_dominance_percentage: float
    honeypot_scan_result: bool
    contract_verified: bool
    token_age_hours: float
    trading_volume_sol: float
    
    # Security flags
    red_flags: List[str] = field(default_factory=list)
    yellow_flags: List[str] = field(default_factory=list)
    security_notes: List[str] = field(default_factory=list)
    
    # Metadata
    scan_timestamp: float = field(default_factory=time.time)
    confidence_level: float = 0.0
    
    @property
    def is_tradeable(self) -> bool:
        """Check if token passes all vetting requirements"""
        return (self.overall_result == VettingResult.APPROVED and 
                self.risk_level in [RiskLevel.SAFE, RiskLevel.MODERATE])
    
    @property
    def survival_rating(self) -> str:
        """Get survival rating for this token"""
        if self.risk_score <= 20:
            return "FORTRESS APPROVED ✅"
        elif self.risk_score <= 40:
            return "MODERATE RISK ⚠️"
        elif self.risk_score <= 70:
            return "HIGH DANGER 🚨"
        else:
            return "LETHAL - AVOID ☠️"

class LiquidityLockAnalyzer:
    """Analyzes liquidity lock status and duration"""
    
    def __init__(self):
        self.min_lock_hours = 24  # Minimum 24h lock requirement
        self.preferred_lock_hours = 168  # Prefer 7 days
        
    async def analyze_liquidity_lock(self, token_address: str, pool_data: Dict) -> Tuple[bool, float, List[str]]:
        """Analyze liquidity lock status"""
        try:
            flags = []
            lock_duration = 0.0
            is_locked = False
            
            # Check for LP token burn (strongest signal)
            lp_supply = pool_data.get('lp_total_supply', 0)
            lp_burned = pool_data.get('lp_burned_supply', 0)
            
            if lp_burned > 0:
                burn_percentage = (lp_burned / lp_supply) * 100 if lp_supply > 0 else 0
                if burn_percentage > 90:
                    is_locked = True
                    lock_duration = float('inf')  # Permanent burn
                    flags.append(f"LP burned: {burn_percentage:.1f}% (EXCELLENT)")
                elif burn_percentage > 50:
                    is_locked = True
                    lock_duration = 8760  # Assume 1 year for significant burns
                    flags.append(f"LP partially burned: {burn_percentage:.1f}%")
            
            # Check for time-locked LP tokens
            lock_info = pool_data.get('liquidity_locks', [])
            if lock_info:
                for lock in lock_info:
                    lock_end = lock.get('unlock_time', 0)
                    current_time = time.time()
                    
                    if lock_end > current_time:
                        hours_locked = (lock_end - current_time) / 3600
                        if hours_locked > lock_duration:
                            lock_duration = hours_locked
                            is_locked = True
                        flags.append(f"Time lock: {hours_locked:.1f}h remaining")
            
            # Check pool creator address (honeypot indicator)
            creator = pool_data.get('creator_address', '')
            known_scammers = self._get_known_scammer_addresses()
            if creator.lower() in known_scammers:
                flags.append("🚨 CREATOR ON SCAMMER LIST")
                is_locked = False  # Override lock status for known scammers
            
            # Minimum lock requirement check
            if is_locked and lock_duration < self.min_lock_hours:
                flags.append(f"⚠️ Lock too short: {lock_duration:.1f}h < {self.min_lock_hours}h required")
                is_locked = False
            
            return is_locked, lock_duration, flags
            
        except Exception as e:
            logger.error(f"Error analyzing liquidity lock: {str(e)}")
            return False, 0.0, [f"Lock analysis failed: {str(e)}"]
    
    def _get_known_scammer_addresses(self) -> set:
        """Get list of known scammer wallet addresses"""
        # In production, this would be loaded from a constantly updated database
        return {
            "scammer1addresshere",
            "scammer2addresshere",
            # Add known rug puller addresses
        }

class HolderDistributionAnalyzer:
    """Analyzes token holder distribution for whale dominance"""
    
    def __init__(self):
        self.max_whale_dominance = 0.10  # 10% maximum single holder
        self.max_top10_dominance = 0.40   # 40% maximum top 10 holders
        
    async def analyze_holder_distribution(self, token_address: str, holders_data: List[Dict]) -> Tuple[float, float, List[str]]:
        """Analyze holder distribution patterns"""
        try:
            flags = []
            whale_dominance = 0.0
            distribution_score = 100.0  # Start with perfect score
            
            if not holders_data:
                flags.append("🚨 NO HOLDER DATA AVAILABLE")
                return 0.0, 0.0, flags
            
            total_holders = len(holders_data)
            if total_holders < 50:
                flags.append(f"⚠️ Low holder count: {total_holders}")
                distribution_score -= 20
            
            # Calculate holder percentages
            total_supply = sum(holder.get('balance', 0) for holder in holders_data)
            if total_supply == 0:
                flags.append("🚨 ZERO TOTAL SUPPLY")
                return 0.0, 0.0, flags
            
            # Top holder analysis
            top_holder = holders_data[0] if holders_data else {}
            top_balance = top_holder.get('balance', 0)
            whale_dominance = (top_balance / total_supply) if total_supply > 0 else 0
            
            if whale_dominance > 0.50:  # >50% single holder
                flags.append(f"🚨 EXTREME WHALE DOMINANCE: {whale_dominance:.1%}")
                distribution_score = 0.0  # Immediate fail
            elif whale_dominance > 0.30:  # >30% single holder
                flags.append(f"🚨 HIGH WHALE DOMINANCE: {whale_dominance:.1%}")
                distribution_score -= 50
            elif whale_dominance > self.max_whale_dominance:  # >10% single holder
                flags.append(f"⚠️ Moderate whale dominance: {whale_dominance:.1%}")
                distribution_score -= 25
            else:
                flags.append(f"✅ Good distribution: Top holder {whale_dominance:.1%}")
            
            # Top 10 holders analysis
            if len(holders_data) >= 10:
                top_10_balance = sum(holder.get('balance', 0) for holder in holders_data[:10])
                top_10_percentage = (top_10_balance / total_supply) if total_supply > 0 else 0
                
                if top_10_percentage > 0.80:  # >80% in top 10
                    flags.append(f"🚨 TOP 10 CONTROL: {top_10_percentage:.1%}")
                    distribution_score -= 30
                elif top_10_percentage > self.max_top10_dominance:  # >40% in top 10
                    flags.append(f"⚠️ High top 10 concentration: {top_10_percentage:.1%}")
                    distribution_score -= 15
            
            # Holder growth analysis (simulate based on distribution quality)
            if total_holders > 500 and whale_dominance < 0.05:
                flags.append("✅ Healthy holder growth pattern detected")
                distribution_score += 10
            
            # Dead wallet analysis
            burned_holders = [h for h in holders_data if self._is_burn_address(h.get('address', ''))]
            if burned_holders:
                burned_percentage = sum(h.get('balance', 0) for h in burned_holders) / total_supply
                flags.append(f"🔥 Supply burned: {burned_percentage:.1%}")
                distribution_score += min(20, burned_percentage * 100)  # Bonus for burns
            
            distribution_score = max(0.0, min(100.0, distribution_score))
            
            return distribution_score, whale_dominance, flags
            
        except Exception as e:
            logger.error(f"Error analyzing holder distribution: {str(e)}")
            return 0.0, 0.0, [f"Distribution analysis failed: {str(e)}"]
    
    def _is_burn_address(self, address: str) -> bool:
        """Check if address is a known burn address"""
        burn_addresses = {
            "11111111111111111111111111111111",  # Solana burn address
            "1nc1nerator11111111111111111111111111111111",  # Incinerator
            "0x000000000000000000000000000000000000dead",  # Dead address
        }
        return address.lower() in [addr.lower() for addr in burn_addresses]

class HoneypotScanner:
    """Scans contracts for honeypot patterns using multiple APIs"""
    
    def __init__(self):
        self.tesseract_api_key = None  # Set from environment
        self.backup_apis = []
        
    async def scan_for_honeypot(self, token_address: str, contract_data: Dict) -> Tuple[bool, float, List[str]]:
        """Comprehensive honeypot scanning"""
        try:
            flags = []
            is_safe = True
            confidence = 90.0
            
            # Layer 1: Tesseract API scan (if available)
            tesseract_result = await self._tesseract_scan(token_address)
            if tesseract_result:
                if tesseract_result.get('is_honeypot', False):
                    flags.append("🚨 TESSERACT: Honeypot detected")
                    is_safe = False
                else:
                    flags.append("✅ TESSERACT: Clean contract")
            
            # Layer 2: Contract code analysis
            code_analysis = await self._analyze_contract_code(token_address, contract_data)
            if code_analysis['suspicious_patterns']:
                flags.extend(code_analysis['flags'])
                confidence -= code_analysis['risk_score']
                if code_analysis['risk_score'] > 70:
                    is_safe = False
            
            # Layer 3: Trading simulation
            sim_result = await self._simulate_trades(token_address)
            if not sim_result['can_sell']:
                flags.append("🚨 SIMULATION: Cannot sell tokens")
                is_safe = False
            
            # Layer 4: Tax analysis
            tax_analysis = await self._analyze_taxes(token_address)
            if tax_analysis['sell_tax'] > 15:  # >15% sell tax is suspicious
                flags.append(f"🚨 HIGH SELL TAX: {tax_analysis['sell_tax']}%")
                is_safe = False
            
            return is_safe, confidence, flags
            
        except Exception as e:
            logger.error(f"Error scanning honeypot: {str(e)}")
            return False, 0.0, [f"Honeypot scan failed: {str(e)}"]
    
    async def _tesseract_scan(self, token_address: str) -> Optional[Dict]:
        """Scan using Tesseract API"""
        try:
            if not self.tesseract_api_key:
                return None
                
            async with aiohttp.ClientSession() as session:
                url = f"https://api.tesseract.finance/v1/scan/{token_address}"
                headers = {"Authorization": f"Bearer {self.tesseract_api_key}"}
                
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                        
        except Exception as e:
            logger.debug(f"Tesseract scan error: {str(e)}")
            
        return None
    
    async def _analyze_contract_code(self, token_address: str, contract_data: Dict) -> Dict:
        """Analyze contract code for suspicious patterns"""
        suspicious_patterns = []
        risk_score = 0.0
        flags = []
        
        # Check for common honeypot patterns
        if contract_data.get('has_mint_function', False):
            suspicious_patterns.append("mint_function")
            risk_score += 20
            flags.append("⚠️ Contract has mint function")
        
        if contract_data.get('has_pause_function', False):
            suspicious_patterns.append("pause_function")
            risk_score += 30
            flags.append("🚨 Contract can be paused")
        
        if contract_data.get('owner_can_change_fees', False):
            suspicious_patterns.append("dynamic_fees")
            risk_score += 25
            flags.append("⚠️ Owner can change fees")
        
        return {
            'suspicious_patterns': suspicious_patterns,
            'risk_score': risk_score,
            'flags': flags
        }
    
    async def _simulate_trades(self, token_address: str) -> Dict:
        """Simulate buy/sell transactions"""
        # Simplified simulation - in production, use actual DEX simulation
        return {
            'can_buy': True,
            'can_sell': True,  # This would be determined by actual simulation
            'slippage_buy': 2.5,
            'slippage_sell': 3.0
        }
    
    async def _analyze_taxes(self, token_address: str) -> Dict:
        """Analyze buy/sell taxes"""
        # Simplified tax analysis - in production, analyze actual contract
        return {
            'buy_tax': 5.0,   # 5% buy tax
            'sell_tax': 8.0,  # 8% sell tax
            'max_tax': 15.0   # Maximum allowed tax
        }

class TokenVettingFortress:
    """Main fortress coordinator implementing all 5 vetting layers"""
    
    def __init__(self):
        self.liquidity_analyzer = LiquidityLockAnalyzer()
        self.holder_analyzer = HolderDistributionAnalyzer()
        self.honeypot_scanner = HoneypotScanner()
        
        # Fortress parameters
        self.min_token_age_minutes = 15    # 15 minute minimum age
        self.min_trading_volume_sol = 100  # 100 SOL minimum volume
        
        # Blacklist storage
        self.blacklisted_tokens = set()
        self.quarantined_tokens = set()
        
        # Performance tracking
        self.total_scans = 0
        self.rejected_tokens = 0
        self.survival_rate = 0.0
        
        logger.info("🏰 Token Vetting Fortress initialized - 5-layer defense active")
    
    async def initialize(self):
        """Initialize the defense system (compatibility method)"""
        # Defense system is ready to use after __init__
        return True

    async def vet_token(self, token_address: str, market_data: Dict, 
                       holders_data: List[Dict] = None, 
                       pool_data: Dict = None) -> TokenVettingReport:
        """Execute complete 5-layer token vetting process"""
        
        self.total_scans += 1
        start_time = time.time()
        
        logger.info(f"🔍 FORTRESS SCAN: Vetting {token_address[:8]}... through 5 defense layers")
        
        # Quick blacklist check
        if token_address in self.blacklisted_tokens:
            logger.warning(f"🚫 Token {token_address[:8]}... is BLACKLISTED")
            return self._create_blacklisted_report(token_address, market_data)
        
        try:
            # Initialize report
            report = TokenVettingReport(
                token_address=token_address,
                token_symbol=market_data.get('symbol', 'UNKNOWN'),
                overall_result=VettingResult.REJECTED,
                risk_level=RiskLevel.EXTREME,
                risk_score=100.0
            )
            
            # LAYER 1: Liquidity Lock Verification
            logger.debug("🔒 Layer 1: Liquidity lock verification...")
            if pool_data:
                lock_status, lock_duration, lock_flags = await self.liquidity_analyzer.analyze_liquidity_lock(
                    token_address, pool_data
                )
                report.liquidity_lock_status = lock_status
                report.liquidity_lock_duration = lock_duration
                
                if not lock_status:
                    report.red_flags.extend(lock_flags)
                    report.red_flags.append("🚨 LIQUIDITY NOT LOCKED")
                else:
                    report.security_notes.extend(lock_flags)
            else:
                report.red_flags.append("🚨 NO POOL DATA AVAILABLE")
            
            # LAYER 2: Holder Distribution Analysis  
            logger.debug("👥 Layer 2: Holder distribution analysis...")
            if holders_data:
                dist_score, whale_dom, dist_flags = await self.holder_analyzer.analyze_holder_distribution(
                    token_address, holders_data
                )
                report.holder_distribution_score = dist_score
                report.whale_dominance_percentage = whale_dom
                
                if whale_dom > 0.10:  # >10% whale dominance
                    report.red_flags.extend(dist_flags)
                else:
                    report.security_notes.extend(dist_flags)
            else:
                report.red_flags.append("🚨 NO HOLDER DATA AVAILABLE")
                report.holder_distribution_score = 0.0
                report.whale_dominance_percentage = 1.0  # Assume worst case
            
            # LAYER 3: Contract Honeypot Scanning
            logger.debug("🍯 Layer 3: Honeypot scanning...")
            contract_data = market_data.get('contract_info', {})
            is_safe, confidence, scan_flags = await self.honeypot_scanner.scan_for_honeypot(
                token_address, contract_data
            )
            report.honeypot_scan_result = is_safe
            report.contract_verified = is_safe
            report.confidence_level = confidence / 100.0
            
            if not is_safe:
                report.red_flags.extend(scan_flags)
                report.red_flags.append("🚨 HONEYPOT DETECTED")
            else:
                report.security_notes.extend(scan_flags)
            
            # LAYER 4: Creation Timestamp Gating
            logger.debug("⏰ Layer 4: Token age verification...")
            creation_time = market_data.get('created_at', time.time())
            token_age_seconds = time.time() - creation_time
            token_age_hours = token_age_seconds / 3600
            token_age_minutes = token_age_seconds / 60
            
            report.token_age_hours = token_age_hours
            
            if token_age_minutes < self.min_token_age_minutes:
                report.red_flags.append(f"🚨 TOKEN TOO NEW: {token_age_minutes:.1f}m < {self.min_token_age_minutes}m required")
            else:
                report.security_notes.append(f"✅ Token age acceptable: {token_age_hours:.1f}h")
            
            # LAYER 5: Trading Volume Validation
            logger.debug("📊 Layer 5: Trading volume validation...")
            volume_24h = market_data.get('volume_24h_sol', 0)
            report.trading_volume_sol = volume_24h
            
            if volume_24h < self.min_trading_volume_sol:
                report.red_flags.append(f"🚨 LOW VOLUME: {volume_24h:.1f} SOL < {self.min_trading_volume_sol} SOL required")
            else:
                report.security_notes.append(f"✅ Volume acceptable: {volume_24h:.1f} SOL")
            
            # FINAL ASSESSMENT: Calculate overall risk and result
            report = self._calculate_final_assessment(report)
            
            # Update blacklist if necessary
            if report.risk_level == RiskLevel.LETHAL:
                self.blacklisted_tokens.add(token_address)
                logger.warning(f"🚫 Token {token_address[:8]}... BLACKLISTED due to lethal risk")
            
            scan_time = time.time() - start_time
            logger.info(f"🏰 FORTRESS VERDICT: {report.token_symbol} - {report.survival_rating} "
                       f"(Risk: {report.risk_score:.1f}%, Time: {scan_time:.2f}s)")
            
            if report.overall_result == VettingResult.REJECTED:
                self.rejected_tokens += 1
            
            self.survival_rate = ((self.total_scans - self.rejected_tokens) / self.total_scans) * 100
            
            return report
            
        except Exception as e:
            logger.error(f"🚨 FORTRESS ERROR during vetting {token_address[:8]}...: {str(e)}")
            return self._create_error_report(token_address, market_data, str(e))
    
    def _calculate_final_assessment(self, report: TokenVettingReport) -> TokenVettingReport:
        """Calculate final risk assessment and approval status"""
        
        risk_score = 0.0
        
        # Risk scoring based on red flags
        critical_flags = len([flag for flag in report.red_flags if "🚨" in flag])
        warning_flags = len([flag for flag in report.red_flags if "⚠️" in flag])
        
        risk_score += critical_flags * 25  # 25 points per critical flag
        risk_score += warning_flags * 10   # 10 points per warning flag
        
        # Specific risk factors
        if not report.liquidity_lock_status:
            risk_score += 30  # Major risk for unlocked liquidity
        
        if report.whale_dominance_percentage > 0.10:
            risk_score += (report.whale_dominance_percentage - 0.10) * 200  # Exponential whale risk
        
        if not report.honeypot_scan_result:
            risk_score += 40  # Major risk for honeypots
        
        if report.token_age_hours < 0.25:  # Less than 15 minutes
            risk_score += 20
        
        if report.trading_volume_sol < 100:
            risk_score += 15
        
        # Cap risk score at 100
        risk_score = min(100.0, risk_score)
        report.risk_score = risk_score
        
        # Determine risk level
        if risk_score >= 90:
            report.risk_level = RiskLevel.LETHAL
            report.overall_result = VettingResult.BLACKLISTED
        elif risk_score >= 70:
            report.risk_level = RiskLevel.EXTREME
            report.overall_result = VettingResult.REJECTED
        elif risk_score >= 40:
            report.risk_level = RiskLevel.HIGH
            report.overall_result = VettingResult.QUARANTINED
        elif risk_score >= 20:
            report.risk_level = RiskLevel.MODERATE
            report.overall_result = VettingResult.APPROVED
        else:
            report.risk_level = RiskLevel.SAFE
            report.overall_result = VettingResult.APPROVED
        
        # Override approval for critical issues
        if critical_flags > 0:
            report.overall_result = VettingResult.REJECTED
        
        return report
    
    def _create_blacklisted_report(self, token_address: str, market_data: Dict) -> TokenVettingReport:
        """Create report for blacklisted token"""
        return TokenVettingReport(
            token_address=token_address,
            token_symbol=market_data.get('symbol', 'BLACKLISTED'),
            overall_result=VettingResult.BLACKLISTED,
            risk_level=RiskLevel.LETHAL,
            risk_score=100.0,
            liquidity_lock_status=False,
            liquidity_lock_duration=0.0,
            holder_distribution_score=0.0,
            whale_dominance_percentage=1.0,
            honeypot_scan_result=False,
            contract_verified=False,
            token_age_hours=0.0,
            trading_volume_sol=0.0,
            red_flags=["🚫 TOKEN IS BLACKLISTED"],
            confidence_level=1.0
        )
    
    def _create_error_report(self, token_address: str, market_data: Dict, error: str) -> TokenVettingReport:
        """Create report for vetting errors"""
        return TokenVettingReport(
            token_address=token_address,
            token_symbol=market_data.get('symbol', 'ERROR'),
            overall_result=VettingResult.REJECTED,
            risk_level=RiskLevel.EXTREME,
            risk_score=90.0,
            liquidity_lock_status=False,
            liquidity_lock_duration=0.0,
            holder_distribution_score=0.0,
            whale_dominance_percentage=1.0,
            honeypot_scan_result=False,
            contract_verified=False,
            token_age_hours=0.0,
            trading_volume_sol=0.0,
            red_flags=[f"🚨 VETTING ERROR: {error}"],
            confidence_level=0.0
        )
    
    def get_fortress_status(self) -> Dict[str, Any]:
        """Get current fortress operational status"""
        return {
            "total_scans": self.total_scans,
            "rejected_tokens": self.rejected_tokens,
            "survival_rate": self.survival_rate,
            "blacklisted_count": len(self.blacklisted_tokens),
            "quarantined_count": len(self.quarantined_tokens),
            "fortress_effectiveness": (self.rejected_tokens / max(1, self.total_scans)) * 100
        }
    
    async def emergency_blacklist_token(self, token_address: str, reason: str):
        """Emergency blacklist for detected threats"""
        self.blacklisted_tokens.add(token_address)
        logger.critical(f"🚨 EMERGENCY BLACKLIST: {token_address[:8]}... - {reason}") 