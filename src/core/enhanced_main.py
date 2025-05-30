"""
Enhanced Main Coordinator for Ant Bot System with Titan Shield Integration

PRODUCTION-READY ENTRY POINT:
- Ant Hierarchy System (Founding Queen -> Queens -> Princesses)
- Enhanced AI Coordination with defense-integrated learning
- Titan Shield Coordinator with 7-layer defense systems
- Battle-hardened execution with survival prioritization

This is the main entry point for the Enhanced Ant Bot Ultimate Survival System.
"""

import asyncio
import logging
import time
import signal
import sys
import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Core AI and hierarchy components
from src.core.ai.ant_hierarchy import FoundingAntQueen, AntRole
from src.core.ai.enhanced_ai_coordinator import AICoordinator

# CRITICAL: Titan Shield Defense System
from src.core.titan_shield_coordinator import TitanShieldCoordinator, DefenseMode

# Core services and managers
from src.core.system_replicator import SystemReplicator
from src.services.wallet_manager import WalletManager
from src.core.data_ingestion import DataIngestion
from src.core.portfolio_manager import PortfolioManager

# External API services
from src.services.helius_service import HeliusService
from src.services.jupiter_service import JupiterService
from src.services.quicknode_service import QuickNodeService

# Portfolio risk management (optional)
try:
    from src.core.portfolio_risk_manager import PortfolioRiskManager
except ImportError:
    PortfolioRiskManager = None
    logging.warning("Portfolio risk manager not available - using basic risk controls")

logger = logging.getLogger(__name__)

class AntBotSystem:
    """
    Enhanced Ant Bot System with Integrated Titan Shield Defense
    
    PRODUCTION-READY TRADING SYSTEM:
    - Founding Ant Queen: Supreme system coordinator with defense mode control
    - Ant Queens: Capital managers with integrated shield propagation  
    - Ant Princesses: Protected trading agents with survival-first parameters
    - AI Coordinator: Defense-integrated intelligence (Grok + Local LLM)
    - Titan Shield: 7-layer defense system for maximum survival (>95% score)
    - System Replicator: Autonomous scaling with defense state propagation
    """
    
    # Class constants for resource management
    DEFAULT_CAPITAL = 20.0
    DEFENSE_CHECK_INTERVAL = 30  # seconds
    MAX_UPTIME_HOURS = 24
    CRITICAL_MEMORY_THRESHOLD = 1024  # MB
    
    def __init__(self, initial_capital: float = DEFAULT_CAPITAL):
        """Initialize Enhanced Ant Bot System with Titan Shield protection"""
        self.initial_capital = initial_capital
        self.start_time = time.time()
        self.shutdown_requested = False
        
        # Core Components (initialized to None for proper cleanup)
        self.founding_queen: Optional[FoundingAntQueen] = None
        self.ai_coordinator: Optional[AICoordinator] = None
        self.system_replicator: Optional[SystemReplicator] = None
        self.wallet_manager: Optional[WalletManager] = None
        self.portfolio_risk_manager: Optional[PortfolioRiskManager] = None
        self.data_ingestion: Optional[DataIngestion] = None
        
        # CRITICAL: Initialize Titan Shield Coordinator FIRST for defense integration
        logger.info("🛡️ Initializing Titan Shield Coordinator...")
        self.titan_shield: Optional[TitanShieldCoordinator] = TitanShieldCoordinator()
        if not self.titan_shield:
            logger.critical("❌ CRITICAL: Failed to initialize TitanShieldCoordinator!")
            raise RuntimeError("Cannot initialize system without defense protection")
        
        # External Services
        self.helius_service: Optional[HeliusService] = None
        self.jupiter_service: Optional[JupiterService] = None
        self.quicknode_service: Optional[QuickNodeService] = None
        
        # System State Management
        self.system_metrics = {
            "total_trades_executed": 0,
            "total_profit": 0.0,
            "system_uptime": 0.0,
            "active_ants": 0,
            "replication_count": 0,
            "survival_score": 0.0,
            "defense_activations": 0,
            "defense_rejections": 0
        }
        
        # Defense Integration State
        self.defense_integrated = False
        self.current_defense_mode = DefenseMode.NORMAL
        self.active_positions: Dict[str, Dict] = {}
        self.main_loop_running = False
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        logger.info(f"🛡️ Enhanced Ant Bot System pre-initialized with TitanShieldCoordinator")
        
    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        def signal_handler(signum, frame):
            logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_requested = True
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    async def initialize(self) -> bool:
        """Initialize the complete Ant Bot system with integrated defense systems"""
        try:
            logger.info("🐜 Initializing Enhanced Ant Bot System with Titan Shield...")
            
            # Initialize external services
            if not await self._initialize_external_services():
                raise Exception("External services initialization failed")
            
            # Initialize core infrastructure
            if not await self._initialize_core_infrastructure():
                raise Exception("Core infrastructure initialization failed")
            
            # CRITICAL: Initialize Titan Shield Defense Systems
            logger.info("🛡️ Initializing Titan Shield Defense Systems...")
            if not await self._initialize_defense_systems():
                raise Exception("Defense systems initialization failed")
            
            # Initialize AI Coordinator with defense integration
            logger.info("🧠 Initializing AI Coordinator...")
            self.ai_coordinator = AICoordinator()
            if not await self.ai_coordinator.initialize():
                raise Exception("AI Coordinator initialization failed")
            
            # Initialize Founding Ant Queen with Titan Shield
            logger.info("👑 Initializing Founding Ant Queen...")
            self.founding_queen = FoundingAntQueen(
                ant_id="founding_queen_0", 
                initial_capital=self.initial_capital,
                titan_shield=self.titan_shield  # CRITICAL: Pass TitanShield
            )
            
            if not await self.founding_queen.initialize():
                raise Exception("Founding Queen initialization failed")
            
            # Initialize System Replicator
            self.system_replicator = SystemReplicator(self.founding_queen)
            
            # Initialize data ingestion with threat monitoring
            logger.info("📊 Initializing Data Ingestion...")
            self.data_ingestion = DataIngestion(
                quicknode_service=self.quicknode_service,
                helius_service=self.helius_service,
                jupiter_service=self.jupiter_service
            )
            await self.data_ingestion.start()
            
            # Complete defense integration
            await self._complete_defense_integration()
            
            logger.info("✅ Enhanced Ant Bot System initialization complete!")
            self._print_initialization_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {str(e)}")
            await self._cleanup_failed_initialization()
            return False
    
    async def _initialize_external_services(self) -> bool:
        """Initialize external services (Helius, Jupiter, QuickNode)"""
        try:
            # Initialize QuickNode service (PRIORITY - most reliable)
            self.quicknode_service = QuickNodeService()
            
            # Initialize Helius service (backup)
            self.helius_service = HeliusService()
            
            # Initialize Jupiter service
            self.jupiter_service = JupiterService()
            
            logger.info("✅ External services initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ External services initialization failed: {str(e)}")
            return False
    
    async def _initialize_core_infrastructure(self) -> bool:
        """Initialize core infrastructure components"""
        try:
            # Initialize wallet manager
            self.wallet_manager = WalletManager()
            if not await self.wallet_manager.initialize():
                raise Exception("Wallet manager initialization failed")
            
            # Initialize portfolio risk manager if available
            if PortfolioRiskManager:
                self.portfolio_risk_manager = PortfolioRiskManager(None)
                if not await self.portfolio_risk_manager.initialize():
                    logger.warning("Portfolio risk manager initialization failed - using basic controls")
            
            logger.info("✅ Core infrastructure initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Core infrastructure initialization failed: {str(e)}")
            return False
    
    async def _cleanup_failed_initialization(self):
        """Cleanup resources after failed initialization"""
        try:
            logger.info("🧹 Cleaning up failed initialization...")
            
            # Cleanup services in reverse order
            if self.data_ingestion:
                await self.data_ingestion.stop()
                
            if self.wallet_manager:
                await self.wallet_manager.close()
                
            # Cleanup external services
            for service in [self.quicknode_service, self.helius_service, self.jupiter_service]:
                if service and hasattr(service, 'cleanup'):
                    await service.cleanup()
                    
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _print_initialization_summary(self):
        """Print system initialization summary"""
        defense_status = "✅ INTEGRATED" if self.defense_integrated else "❌ NOT INTEGRATED"
        
        summary = f"""
🐜 ========================================
   ANT BOT SYSTEM - INITIALIZATION COMPLETE
========================================

👑 Founding Queen ID: {self.founding_queen.ant_id}
💰 Initial Capital: {self.initial_capital} SOL
🧠 AI Models: Grok AI + Local LLM
🔄 Self-Replication: Enabled
📊 Data Ingestion: Active

🛡️ TITAN SHIELD DEFENSE SYSTEMS: {defense_status}
   1. Token Vetting Fortress (5-layer screening)
   2. Volatility-Adaptive Armor (Dynamic parameters)
   3. AI Deception Shield (Manipulation detection)
   4. Transaction Warfare System (Network resistance)
   5. Capital Forcefields (Risk containment)
   6. Adversarial Learning Core (AI protection)
   7. Counter-Attack Profit Engines (Exploit monetization)

🏗️ Architecture:
   └── Founding Ant Queen
       ├── AI Coordinator (Grok + Local LLM)
       ├── Titan Shield Coordinator (7-layer defense)
       ├── System Replicator (Auto-scaling)
       └── Ant Queens
           └── Worker Ants (Princesses) [PROTECTED]

⚙️ System Ready for Operations with Maximum Protection!
========================================
        """
        logger.info(summary)
    
    async def start(self):
        """Start the main Ant Bot system operations"""
        try:
            logger.info("🚀 Starting Ant Bot System operations...")
            self.main_loop_running = True
            
            # Start the main operation loop
            await self._main_operation_loop()
            
        except KeyboardInterrupt:
            logger.info("👋 Shutdown requested by user")
            self.shutdown_requested = True
        except Exception as e:
            logger.error(f"💥 Critical error in main operations: {str(e)}")
        finally:
            await self.shutdown()
    
    async def _main_operation_loop(self):
        """Main system operation loop with integrated defense monitoring"""
        logger.info("🔄 Starting main operation loop with Titan Shield protection...")
        
        loop_count = 0
        
        while self.main_loop_running and not self.shutdown_requested:
            try:
                loop_start_time = time.time()
                loop_count += 1
                
                logger.info(f"🔄 Operation Loop #{loop_count} starting with defense monitoring...")
                
                # Step 1: Check defense system status and auto-responses
                await self._monitor_defense_systems()
                
                # Step 2: Get market opportunities with threat screening
                market_opportunities = await self._scan_market_opportunities_with_defense()
                
                # Step 3: Update capital tracking for forcefields
                await self._update_capital_tracking()
                
                # Step 4: Coordinate Ant system for opportunities (with defense approval)
                coordination_results = await self.founding_queen.coordinate_system(market_opportunities)
                
                # Step 5: Execute trading decisions with full protection
                execution_results = await self._execute_trading_decisions_protected(coordination_results["decisions"])
                
                # Step 6: Process learning feedback to defense systems
                await self._process_learning_feedback_with_defense(execution_results)
                
                # Step 7: Check replication conditions with defense constraints
                await self._check_replication_conditions_with_defense()
                
                # Step 8: Update system metrics including defense metrics
                await self._update_system_metrics_with_defense()
                
                # Step 9: Log comprehensive system status
                await self._log_system_status_with_defense(loop_count)
                
                # Wait before next loop (30 seconds)
                loop_duration = time.time() - loop_start_time
                sleep_time = max(0, 30 - loop_duration)
                
                if sleep_time > 0:
                    logger.debug(f"💤 Sleeping for {sleep_time:.1f} seconds until next loop...")
                    await asyncio.sleep(sleep_time)
                    
            except Exception as e:
                logger.error(f"❌ Main operation loop error: {str(e)}")
                
                # Check if error requires defense mode escalation
                if "critical" in str(e).lower() or "emergency" in str(e).lower():
                    await self._update_defense_mode(DefenseMode.HIGH_ALERT)
                
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _monitor_defense_systems(self):
        """Monitor defense systems and handle auto-responses"""
        try:
            if not self.defense_integrated:
                logger.warning("⚠️ Defense systems not integrated - skipping monitoring")
                return
            
            # Get current defense status
            shield_status = self.titan_shield.get_titan_shield_status()
            
            # Check for defense mode changes
            current_shield_mode = self.titan_shield.defense_mode
            if current_shield_mode != self.current_defense_mode:
                await self._update_defense_mode(current_shield_mode)
            
            # Monitor survival score
            survival_score = shield_status.get('overall_survival_score', 0)
            if survival_score < 50:
                logger.critical(f"🚨 CRITICAL SURVIVAL SCORE: {survival_score:.1f}%")
                await self._update_defense_mode(DefenseMode.CRITICAL)
            elif survival_score < 70:
                logger.warning(f"⚠️ LOW SURVIVAL SCORE: {survival_score:.1f}%")
                if self.current_defense_mode == DefenseMode.NORMAL:
                    await self._update_defense_mode(DefenseMode.ELEVATED)
            
            logger.debug(f"🛡️ Defense monitoring: {shield_status['defense_mode']} mode, "
                        f"Survival: {survival_score:.1f}%")
                        
        except Exception as e:
            logger.error(f"❌ Defense monitoring error: {str(e)}")
    
    async def _scan_market_opportunities_with_defense(self) -> List[Dict]:
        """Scan market opportunities with preliminary threat assessment"""
        try:
            # Get basic market data
            raw_opportunities = await self._scan_market_opportunities()
            
            if not self.defense_integrated:
                logger.warning("⚠️ Defense not integrated - returning unfiltered opportunities")
                return raw_opportunities
            
            # Pre-filter opportunities through basic threat detection
            screened_opportunities = []
            
            for opportunity in raw_opportunities:
                token_address = opportunity.get('token_address')
                if not token_address:
                    continue
                
                # Quick threat check (full analysis done during execution)
                kill_signal, kill_reason = self.titan_shield.deception_shield.check_kill_signals(token_address)
                
                if kill_signal:
                    logger.warning(f"🚫 PRE-FILTERED THREAT: {token_address[:8]}... - {kill_reason}")
                    continue
                
                screened_opportunities.append(opportunity)
            
            logger.info(f"🛡️ Pre-screening: {len(raw_opportunities)} → {len(screened_opportunities)} opportunities")
            return screened_opportunities
            
        except Exception as e:
            logger.error(f"❌ Market screening error: {str(e)}")
            return await self._scan_market_opportunities()  # Fallback to unscreened
    
    async def _update_capital_tracking(self):
        """Update capital tracking across all systems"""
        try:
            if not self.defense_integrated:
                return
            
            # Calculate total capital across all agents
            total_capital = 0.0
            self.active_positions.clear()
            
            if self.founding_queen:
                for queen_id, queen in self.founding_queen.queens.items():
                    total_capital += queen.capital.current_balance
                    
                    for princess_id, princess in queen.princesses.items():
                        total_capital += princess.capital.current_balance
                        
                        # Track active positions
                        for position_id, position in princess.active_positions.items():
                            self.active_positions[f"{princess_id}_{position_id}"] = position
            
            # Update Titan Shield capital tracking
            position_update_success = self.titan_shield.capital_forcefields.update_capital_status(
                total_capital, self.active_positions
            )
            
            if not position_update_success:
                logger.warning("⚠️ Capital exposure limits exceeded - defense mode escalation")
                await self._update_defense_mode(DefenseMode.HIGH_ALERT)
            
            logger.debug(f"💰 Capital tracking updated: {total_capital:.2f} SOL, "
                        f"{len(self.active_positions)} active positions")
                        
        except Exception as e:
            logger.error(f"❌ Capital tracking error: {str(e)}")
    
    async def _execute_trading_decisions_protected(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute trading decisions with full Titan Shield protection"""
        execution_results = []
        
        try:
            for decision_data in decisions:
                princess_id = decision_data["princess_id"]
                decision = decision_data["decision"]
                
                if decision["action"] in ["buy", "sell"]:
                    # Find the princess and execute trade with protection
                    princess = self._find_princess_by_id(princess_id)
                    if princess:
                        # CRITICAL: Execute trade with Titan Shield protection
                        result = await princess.execute_trade_protected(decision, self.wallet_manager)
                        
                        if result.get("success"):
                            execution_results.append({
                                "princess_id": princess_id,
                                "trade_id": result["trade_record"]["trade_id"],
                                "decision": decision,
                                "result": result,
                                "timestamp": time.time(),
                                "defense_approved": result.get("defense_approved", False)
                            })
                            
                            # Update system metrics
                            self.system_metrics["total_trades_executed"] += 1
                            self.system_metrics["total_profit"] += result["trade_record"]["profit"]
                        else:
                            # Log defense rejection
                            rejection_reason = result.get("rejection_reason", "Unknown")
                            logger.warning(f"🛡️ TRADE REJECTED by defense systems: {rejection_reason}")
            
            logger.info(f"⚡ Executed {len(execution_results)} protected trades")
            return execution_results
            
        except Exception as e:
            logger.error(f"❌ Protected trading execution error: {str(e)}")
            return execution_results
    
    def _find_princess_by_id(self, princess_id: str):
        """Find a princess by ID across all queens"""
        try:
            for queen in self.founding_queen.queens.values():
                if princess_id in queen.princesses:
                    return queen.princesses[princess_id]
            return None
        except Exception as e:
            logger.error(f"Princess lookup error: {str(e)}")
            return None
    
    async def _process_learning_feedback_with_defense(self, execution_results: List[Dict[str, Any]]):
        """Process learning feedback including defense system learning"""
        try:
            for result in execution_results:
                trade_outcome = {
                    "profit": result["result"]["trade_record"]["profit"],
                    "success": result["result"]["trade_record"]["success"],
                    "market_conditions": {
                        "timestamp": result["timestamp"]
                    },
                    "defense_approved": result.get("defense_approved", False)
                }
                
                # Send feedback to AI coordinator
                await self.ai_coordinator.learn_from_outcome(
                    result["trade_id"],
                    result["decision"],
                    trade_outcome
                )
                
                # CRITICAL: Send feedback to defense learning core
                if self.defense_integrated:
                    token_address = result["decision"].get("token_address")
                    is_meme_coin = True  # Assume meme coin trading
                    
                    self.titan_shield.learning_core.process_trade_outcome(
                        token_address, trade_outcome, is_meme_coin
                    )
            
            if execution_results:
                logger.info(f"🧠 Processed learning feedback for {len(execution_results)} trades "
                           f"(including defense system learning)")
                
        except Exception as e:
            logger.error(f"❌ Learning feedback error: {str(e)}")
    
    async def _check_replication_conditions_with_defense(self):
        """Check replication conditions with defense constraints"""
        try:
            if not self.defense_integrated:
                await self._check_replication_conditions()
                return
            
            # Check if defense mode allows replication
            if self.current_defense_mode in [DefenseMode.CRITICAL, DefenseMode.LOCKDOWN]:
                logger.warning(f"🛡️ Replication blocked by defense mode: {self.current_defense_mode.value}")
                return
            
            # Get survival score
            shield_status = self.titan_shield.get_titan_shield_status()
            survival_score = shield_status.get('overall_survival_score', 0)
            
            # Only allow replication with good survival score
            if survival_score >= 85:
                await self._check_replication_conditions()
            else:
                logger.info(f"🛡️ Replication deferred - survival score too low: {survival_score:.1f}%")
                
        except Exception as e:
            logger.error(f"❌ Replication check error: {str(e)}")
    
    async def _update_system_metrics_with_defense(self):
        """Update system metrics including defense system metrics"""
        try:
            # Update standard metrics
            await self._update_system_metrics()
            
            if self.defense_integrated:
                # Add defense-specific metrics
                shield_status = self.titan_shield.get_titan_shield_status()
                
                self.system_metrics.update({
                    "defense_mode": self.current_defense_mode.value,
                    "survival_score": shield_status.get('overall_survival_score', 0),
                    "threats_detected": shield_status.get('system_performance', {}).get('threats_neutralized', 0),
                    "defense_actions": shield_status.get('system_performance', {}).get('auto_responses_taken', 0),
                    "vetting_effectiveness": shield_status.get('current_metrics', {}).get('vetting_effectiveness', 0),
                    "armor_efficiency": shield_status.get('current_metrics', {}).get('armor_efficiency', 0)
                })
                
        except Exception as e:
            logger.error(f"❌ Metrics update error: {str(e)}")
    
    async def _log_system_status_with_defense(self, loop_count: int):
        """Log comprehensive system status including defense metrics"""
        try:
            # Log standard status
            await self._log_system_status(loop_count)
            
            if self.defense_integrated:
                # Log defense-specific status
                shield_status = self.titan_shield.get_titan_shield_status()
                
                logger.info(
                    f"🛡️ DEFENSE STATUS - Mode: {self.current_defense_mode.value}, "
                    f"Survival: {shield_status.get('overall_survival_score', 0):.1f}%, "
                    f"Threats: {shield_status.get('system_performance', {}).get('threats_neutralized', 0)}, "
                    f"Actions: {shield_status.get('system_performance', {}).get('auto_responses_taken', 0)}"
                )
                
        except Exception as e:
            logger.error(f"❌ Status logging error: {str(e)}")
    
    async def shutdown(self):
        """Graceful system shutdown"""
        try:
            logger.info("🛑 Initiating system shutdown...")
            self.main_loop_running = False
            
            # Stop data ingestion
            if self.data_ingestion:
                await self.data_ingestion.close()
                logger.info("✅ Data ingestion stopped")
            
            # Terminate replicated instances
            if self.system_replicator:
                for instance_id in list(self.system_replicator.instances.keys()):
                    await self.system_replicator.terminate_instance(instance_id)
                logger.info("✅ Replicated instances terminated")
            
            # Close wallet manager
            if self.wallet_manager:
                await self.wallet_manager.close()
                logger.info("✅ Wallet manager closed")
            
            # Close portfolio risk manager
            if self.portfolio_risk_manager:
                await self.portfolio_risk_manager.close()
                logger.info("✅ Portfolio risk manager closed")
            
            # Close external services
            if self.helius_service:
                await self.helius_service.close()
            if self.jupiter_service:
                await self.jupiter_service.close()
            if self.quicknode_service:
                await self.quicknode_service.close()
            logger.info("✅ External services closed")
            
            # Save final system state
            await self._save_system_state()
            
            logger.info("🏁 Ant Bot System shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {str(e)}")
    
    async def _save_system_state(self):
        """Save final system state for recovery"""
        try:
            final_state = {
                "shutdown_time": time.time(),
                "system_metrics": self.system_metrics,
                "founding_queen_status": self.founding_queen.get_system_status() if self.founding_queen else {},
                "ai_performance": self.ai_coordinator.get_performance_summary() if self.ai_coordinator else {},
                "replication_history": self.system_replicator.replication_history if self.system_replicator else []
            }
            
            os.makedirs("data/system_state", exist_ok=True)
            with open("data/system_state/final_state.json", "w") as f:
                json.dump(final_state, f, indent=2)
            
            logger.info("💾 System state saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Save state error: {str(e)}")
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview"""
        try:
            overview = {
                "system_info": {
                    "start_time": self.start_time,
                    "uptime": time.time() - self.start_time,
                    "status": "running" if self.main_loop_running else "stopped"
                },
                "metrics": self.system_metrics,
                "components": {
                    "founding_queen": self.founding_queen.get_system_status() if self.founding_queen else None,
                    "ai_coordinator": self.ai_coordinator.get_performance_summary() if self.ai_coordinator else None,
                    "replication": self.system_replicator.get_replication_status() if self.system_replicator else None
                }
            }
            return overview
            
        except Exception as e:
            logger.error(f"System overview error: {str(e)}")
            return {"error": str(e)}
    
    async def _initialize_defense_systems(self) -> bool:
        """Initialize all defense systems with proper validation"""
        try:
            if not self.titan_shield:
                logger.critical("❌ TitanShieldCoordinator not available for initialization")
                return False
            
            logger.info("🛡️ Initializing defense systems...")
            
            # Initialize Token Vetting Fortress
            logger.info("🛡️ Layer 1: Initializing Token Vetting Fortress...")
            vetting_success = await self.titan_shield.initialize_token_vetting()
            if not vetting_success:
                logger.error("❌ Token Vetting Fortress initialization failed")
                return False
            
            # Initialize Volatility Adaptive Armor
            logger.info("🛡️ Layer 2: Initializing Volatility Adaptive Armor...")
            armor_success = await self.titan_shield.initialize_volatility_armor()
            if not armor_success:
                logger.error("❌ Volatility Adaptive Armor initialization failed")
                return False
            
            # Initialize AI Deception Shield
            logger.info("🛡️ Layer 3: Initializing AI Deception Shield...")
            shield_success = await self.titan_shield.initialize_deception_shield()
            if not shield_success:
                logger.error("❌ AI Deception Shield initialization failed")
                return False
            
            # Initialize Transaction Warfare System
            logger.info("🛡️ Layer 4: Initializing Transaction Warfare System...")
            warfare_success = await self.titan_shield.initialize_transaction_warfare()
            if not warfare_success:
                logger.error("❌ Transaction Warfare System initialization failed")
                return False
            
            # Initialize remaining layers (Capital Forcefields, Learning Core, Profit Engines)
            logger.info("🛡️ Layers 5-7: Initializing remaining defense systems...")
            remaining_success = await self.titan_shield.initialize_remaining_systems()
            if not remaining_success:
                logger.error("❌ Remaining defense systems initialization failed")
                return False
            
            # Set defense mode to normal operations
            await self.titan_shield.set_defense_mode(DefenseMode.NORMAL)
            self.current_defense_mode = DefenseMode.NORMAL
            
            logger.info("✅ All 7-layer defense systems initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Defense systems initialization critical failure: {str(e)}")
            return False
    
    async def _complete_defense_integration(self):
        """Complete defense integration with all system components"""
        try:
            if not self.titan_shield:
                raise Exception("TitanShieldCoordinator not initialized")
            
            logger.info("🛡️ Completing system-wide defense integration...")
            
            # Validate all defense systems are operational
            shield_status = self.titan_shield.get_titan_shield_status()
            if not shield_status.get('all_systems_operational', False):
                raise Exception("Not all defense systems are operational")
            
            # Mark defense integration as complete
            self.defense_integrated = True
            
            # Initialize survival metrics tracking
            survival_metrics = self.titan_shield.get_survival_metrics()
            self.system_metrics["survival_score"] = survival_metrics.overall_score
            
            logger.info(f"✅ Defense integration complete - System survival score: {survival_metrics.overall_score:.1f}%")
            logger.info(f"🛡️ Defense mode: {survival_metrics.defense_mode.value}")
            logger.info(f"🛡️ All trading agents will operate under Titan Shield protection")
            
        except Exception as e:
            logger.error(f"❌ Defense integration completion failed: {str(e)}")
            self.defense_integrated = False
            raise Exception(f"Critical: Defense integration failure - {str(e)}")
    
    async def _propagate_defense_to_agents(self):
        """Propagate Titan Shield to all trading agents"""
        try:
            if self.founding_queen:
                # Pass Titan Shield reference to Founding Queen (already done in init)
                
                # Propagate to all existing Queens
                for queen_id, queen in self.founding_queen.queens.items():
                    queen.titan_shield = self.titan_shield
                    
                    # Propagate to all Princesses under each Queen
                    for princess_id, princess in queen.princesses.items():
                        princess.titan_shield = self.titan_shield
                        logger.debug(f"🛡️ Defense systems integrated with Princess {princess_id}")
                
                logger.info("🛡️ Titan Shield propagated to all trading agents")
                
        except Exception as e:
            logger.error(f"❌ Defense propagation failed: {str(e)}")
    
    async def _update_defense_mode(self, new_mode: DefenseMode):
        """Update defense mode across all systems"""
        try:
            old_mode = self.current_defense_mode
            self.current_defense_mode = new_mode
            
            # Update Titan Shield defense mode
            self.titan_shield.defense_mode = new_mode
            
            # Propagate to all agents
            await self._propagate_defense_mode_to_agents(new_mode)
            
            logger.critical(f"🛡️ DEFENSE MODE CHANGE: {old_mode.value} → {new_mode.value}")
            
        except Exception as e:
            logger.error(f"❌ Defense mode update failed: {str(e)}")
    
    async def _propagate_defense_mode_to_agents(self, defense_mode: DefenseMode):
        """Propagate defense mode changes to all trading agents"""
        try:
            if self.founding_queen:
                for queen_id, queen in self.founding_queen.queens.items():
                    for princess_id, princess in queen.princesses.items():
                        # Update Princess defense parameters based on mode
                        await self._update_princess_defense_parameters(princess, defense_mode)
                
        except Exception as e:
            logger.error(f"❌ Defense mode propagation failed: {str(e)}")
    
    async def _update_princess_defense_parameters(self, princess, defense_mode: DefenseMode):
        """Update Princess trading parameters based on defense mode"""
        try:
            # Adjust trading parameters based on defense mode
            if defense_mode == DefenseMode.LOCKDOWN:
                princess.max_position_multiplier = 0.1  # 10% of normal position size
                princess.trading_enabled = False
            elif defense_mode == DefenseMode.CRITICAL:
                princess.max_position_multiplier = 0.3  # 30% of normal position size
                princess.trading_enabled = True
            elif defense_mode == DefenseMode.HIGH_ALERT:
                princess.max_position_multiplier = 0.5  # 50% of normal position size
                princess.trading_enabled = True
            elif defense_mode == DefenseMode.ELEVATED:
                princess.max_position_multiplier = 0.7  # 70% of normal position size
                princess.trading_enabled = True
            else:  # NORMAL
                princess.max_position_multiplier = 1.0  # Full position size
                princess.trading_enabled = True
                
            logger.debug(f"🛡️ Updated Princess {princess.ant_id} for {defense_mode.value} mode")
            
        except Exception as e:
            logger.debug(f"❌ Princess parameter update failed: {str(e)}")

    async def _scan_market_opportunities(self) -> List[Dict[str, Any]]:
        """Scan for market opportunities using data ingestion"""
        try:
            # Get market data from data ingestion
            market_data = await self.data_ingestion.get_market_data()
            
            # Filter and enhance opportunities
            opportunities = []
            for token_data in market_data[:10]:  # Limit to top 10 opportunities
                # Enhance with additional metrics
                enhanced_data = {
                    **token_data,
                    "timestamp": time.time(),
                    "source": "market_scanner"
                }
                opportunities.append(enhanced_data)
            
            logger.info(f"📊 Found {len(opportunities)} market opportunities")
            return opportunities
            
        except Exception as e:
            logger.error(f"❌ Market scanning error: {str(e)}")
            return []
    
    async def _check_replication_conditions(self):
        """Check and handle system replication"""
        try:
            # Check if replication conditions are met
            should_replicate = await self.system_replicator.monitor_replication_conditions()
            
            if should_replicate:
                logger.info("🔄 Replication conditions met, attempting to replicate system...")
                new_instance_id = await self.system_replicator.replicate_system()
                
                if new_instance_id:
                    self.system_metrics["replication_count"] += 1
                    logger.info(f"✅ Successfully replicated system as instance: {new_instance_id}")
                else:
                    logger.warning("⚠️ System replication attempt failed")
            
            # Manage existing instances
            await self.system_replicator.manage_instances()
                
        except Exception as e:
            logger.error(f"❌ Replication check error: {str(e)}")
    
    async def _update_system_metrics(self):
        """Update system-wide metrics"""
        try:
            # Update uptime
            self.system_metrics["system_uptime"] = time.time() - self.start_time
            
            # Count active ants
            total_ants = 1  # Founding Queen
            total_ants += len(self.founding_queen.queens)  # Queens
            
            for queen in self.founding_queen.queens.values():
                total_ants += len(queen.princesses)  # Princesses
            
            self.system_metrics["active_ants"] = total_ants
                
        except Exception as e:
            logger.error(f"❌ Metrics update error: {str(e)}")
    
    async def _log_system_status(self, loop_count: int):
        """Log comprehensive system status"""
        try:
            # Get system status from all components
            founding_queen_status = self.founding_queen.get_system_status()
            ai_performance = self.ai_coordinator.get_performance_summary()
            replication_status = self.system_replicator.get_replication_status()
            
            # Create status summary
            status_summary = {
                "loop_count": loop_count,
                "system_metrics": self.system_metrics,
                "founding_queen": {
                    "active_queens": founding_queen_status.get("active_queens", 0),
                    "total_capital": founding_queen_status.get("system_metrics", {}).get("total_capital", 0),
                    "total_trades": founding_queen_status.get("system_metrics", {}).get("total_trades", 0),
                    "system_profit": founding_queen_status.get("system_metrics", {}).get("system_profit", 0)
                },
                "ai_performance": ai_performance,
                "replication": {
                    "active_instances": replication_status.get("active_instances", 0),
                    "total_instances": replication_status.get("total_instances", 0)
                }
            }
            
            # Log detailed status every 10 loops (5 minutes)
            if loop_count % 10 == 0:
                logger.info(f"📊 SYSTEM STATUS REPORT (Loop #{loop_count}):")
                logger.info(f"   🐜 Total Ants: {self.system_metrics['active_ants']}")
                logger.info(f"   💰 Total Capital: {status_summary['founding_queen']['total_capital']:.4f} SOL")
                logger.info(f"   📈 Total Profit: {status_summary['founding_queen']['system_profit']:.4f} SOL")
                logger.info(f"   ⚡ Total Trades: {status_summary['founding_queen']['total_trades']}")
                logger.info(f"   🔄 Replicated Instances: {status_summary['replication']['active_instances']}")
                logger.info(f"   ⏱️ Uptime: {self.system_metrics['system_uptime']/3600:.1f} hours")
            else:
                # Brief status for other loops
                logger.info(f"📊 Loop #{loop_count}: {self.system_metrics['active_ants']} ants, {status_summary['founding_queen']['system_profit']:.4f} SOL profit")
                
        except Exception as e:
            logger.error(f"❌ Status logging error: {str(e)}")

# Signal handlers for graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"📡 Received signal {signum}, initiating graceful shutdown...")
    # Set global shutdown flag (this would be handled by the main function)
    sys.exit(0)

async def main():
    """Main entry point for the Enhanced Ant Bot System"""
    logger.info("🐜 Starting Enhanced Ant Bot System...")
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Get initial capital from environment or use default
    initial_capital = float(os.getenv("INITIAL_CAPITAL", "20.0"))
    
    # Create and start the system
    system = AntBotSystem(initial_capital)
    
    if await system.initialize():
        await system.start()
    else:
        logger.error("💀 Failed to initialize Ant Bot System")
        sys.exit(1)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the system
    asyncio.run(main()) 