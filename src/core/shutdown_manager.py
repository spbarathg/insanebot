"""
Graceful Shutdown Manager - System shutdown and cleanup coordination

Handles signal processing, graceful component shutdown, connection cleanup,
and proper resource disposal for safe system termination.
"""

import asyncio
import logging
import signal
import time
import sys
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass
from enum import Enum
import functools
import threading

logger = logging.getLogger(__name__)

class ShutdownPhase(Enum):
    """Shutdown phases"""
    RUNNING = "running"
    SIGNAL_RECEIVED = "signal_received"
    STOPPING_SERVICES = "stopping_services"
    CLEANING_UP = "cleaning_up"
    SHUTDOWN_COMPLETE = "shutdown_complete"

@dataclass
class ShutdownHandler:
    """Shutdown handler definition"""
    name: str
    handler: Callable
    priority: int = 100  # Lower number = higher priority
    timeout: float = 30.0
    critical: bool = False  # If true, shutdown fails if this handler fails

class ShutdownManager:
    """
    Graceful shutdown manager
    
    Features:
    - Signal handling (SIGTERM, SIGINT, SIGQUIT)
    - Ordered shutdown sequence by priority
    - Timeout handling for each shutdown phase
    - Cleanup coordination across components
    - Status reporting during shutdown
    - Force shutdown fallback
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Shutdown state
        self.phase = ShutdownPhase.RUNNING
        self.shutdown_requested = False
        self.shutdown_completed = False
        self.shutdown_start_time: Optional[float] = None
        
        # Handlers and components
        self.shutdown_handlers: List[ShutdownHandler] = []
        self.active_components: Set[str] = set()
        self.shutdown_callbacks: List[Callable] = []
        
        # Monitoring
        self.shutdown_status: Dict[str, Any] = {}
        self.failed_handlers: List[str] = []
        
        # Signal handling
        self.original_handlers: Dict[int, Any] = {}
        self.signals_registered = False
        
        logger.info("ShutdownManager initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default shutdown configuration"""
        return {
            'total_shutdown_timeout': 300.0,  # 5 minutes total
            'default_handler_timeout': 30.0,
            'force_shutdown_after': 600.0,   # 10 minutes absolute maximum
            'status_update_interval': 5.0,
            'enable_signal_handling': True,
            'graceful_signals': [signal.SIGTERM, signal.SIGINT],
            'immediate_signals': [signal.SIGQUIT] if hasattr(signal, 'SIGQUIT') else []
        }
    
    def register_signal_handlers(self) -> bool:
        """Register signal handlers for graceful shutdown"""
        try:
            if not self.config.get('enable_signal_handling', True):
                return True
            
            # Store original handlers
            graceful_signals = self.config.get('graceful_signals', [signal.SIGTERM, signal.SIGINT])
            immediate_signals = self.config.get('immediate_signals', [])
            
            for sig in graceful_signals:
                if hasattr(signal, sig.name if hasattr(sig, 'name') else str(sig)):
                    self.original_handlers[sig] = signal.signal(sig, self._graceful_signal_handler)
            
            for sig in immediate_signals:
                if hasattr(signal, sig.name if hasattr(sig, 'name') else str(sig)):
                    self.original_handlers[sig] = signal.signal(sig, self._immediate_signal_handler)
            
            self.signals_registered = True
            logger.info("Signal handlers registered for graceful shutdown")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register signal handlers: {e}")
            return False
    
    def _graceful_signal_handler(self, signum: int, frame):
        """Handle graceful shutdown signals"""
        signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        logger.info(f"Received graceful shutdown signal: {signal_name}")
        
        if not self.shutdown_requested:
            # Start graceful shutdown in a separate thread to avoid blocking the signal handler
            threading.Thread(target=self._start_graceful_shutdown, daemon=True).start()
    
    def _immediate_signal_handler(self, signum: int, frame):
        """Handle immediate shutdown signals"""
        signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        logger.warning(f"Received immediate shutdown signal: {signal_name}")
        
        # Force immediate shutdown
        self._force_shutdown()
    
    def _start_graceful_shutdown(self):
        """Start graceful shutdown process"""
        try:
            # Use asyncio.run if there's no running loop, otherwise create a task
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(self.initiate_shutdown())
            except RuntimeError:
                # No running loop, create one
                asyncio.run(self.initiate_shutdown())
        except Exception as e:
            logger.error(f"Error starting graceful shutdown: {e}")
            self._force_shutdown()
    
    def register_shutdown_handler(self, name: str, handler: Callable, 
                                 priority: int = 100, timeout: float = None, 
                                 critical: bool = False) -> bool:
        """Register a shutdown handler"""
        try:
            timeout = timeout or self.config.get('default_handler_timeout', 30.0)
            
            shutdown_handler = ShutdownHandler(
                name=name,
                handler=handler,
                priority=priority,
                timeout=timeout,
                critical=critical
            )
            
            self.shutdown_handlers.append(shutdown_handler)
            
            # Sort by priority (lower number = higher priority)
            self.shutdown_handlers.sort(key=lambda h: h.priority)
            
            logger.info(f"Registered shutdown handler: {name} (priority: {priority})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register shutdown handler {name}: {e}")
            return False
    
    def register_component(self, component_name: str):
        """Register an active component"""
        self.active_components.add(component_name)
        logger.debug(f"Registered active component: {component_name}")
    
    def unregister_component(self, component_name: str):
        """Unregister a component (mark as stopped)"""
        self.active_components.discard(component_name)
        logger.debug(f"Unregistered component: {component_name}")
    
    def add_shutdown_callback(self, callback: Callable):
        """Add a callback to be called during shutdown"""
        self.shutdown_callbacks.append(callback)
    
    async def initiate_shutdown(self, reason: str = "Manual shutdown request") -> bool:
        """Initiate graceful shutdown process"""
        if self.shutdown_requested:
            logger.warning("Shutdown already in progress")
            return False
        
        try:
            self.shutdown_requested = True
            self.shutdown_start_time = time.time()
            self.phase = ShutdownPhase.SIGNAL_RECEIVED
            
            logger.info(f"Initiating graceful shutdown: {reason}")
            
            # Set up force shutdown timer
            force_timeout = self.config.get('force_shutdown_after', 600.0)
            asyncio.create_task(self._force_shutdown_timer(force_timeout))
            
            # Start shutdown status monitoring
            asyncio.create_task(self._shutdown_monitoring())
            
            # Execute shutdown sequence
            success = await self._execute_shutdown_sequence()
            
            if success:
                self.phase = ShutdownPhase.SHUTDOWN_COMPLETE
                self.shutdown_completed = True
                logger.info("Graceful shutdown completed successfully")
            else:
                logger.error("Graceful shutdown completed with errors")
            
            return success
            
        except Exception as e:
            logger.error(f"Error during shutdown initiation: {e}")
            self._force_shutdown()
            return False
    
    async def _execute_shutdown_sequence(self) -> bool:
        """Execute the shutdown sequence"""
        try:
            success = True
            
            # Phase 1: Stop accepting new requests/tasks
            self.phase = ShutdownPhase.STOPPING_SERVICES
            logger.info("Phase 1: Stopping services...")
            
            # Call shutdown callbacks
            for callback in self.shutdown_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                except Exception as e:
                    logger.error(f"Error in shutdown callback: {e}")
            
            # Phase 2: Execute shutdown handlers in priority order
            self.phase = ShutdownPhase.CLEANING_UP
            logger.info("Phase 2: Executing shutdown handlers...")
            
            for handler in self.shutdown_handlers:
                handler_success = await self._execute_handler(handler)
                if not handler_success:
                    self.failed_handlers.append(handler.name)
                    if handler.critical:
                        logger.error(f"Critical shutdown handler {handler.name} failed")
                        success = False
                    else:
                        logger.warning(f"Non-critical shutdown handler {handler.name} failed")
            
            # Phase 3: Final cleanup
            await self._final_cleanup()
            
            return success
            
        except Exception as e:
            logger.error(f"Error in shutdown sequence: {e}")
            return False
    
    async def _execute_handler(self, handler: ShutdownHandler) -> bool:
        """Execute a single shutdown handler"""
        try:
            logger.info(f"Executing shutdown handler: {handler.name}")
            start_time = time.time()
            
            # Execute handler with timeout
            if asyncio.iscoroutinefunction(handler.handler):
                await asyncio.wait_for(handler.handler(), timeout=handler.timeout)
            else:
                # Run sync handler in thread pool
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, handler.handler),
                    timeout=handler.timeout
                )
            
            execution_time = time.time() - start_time
            logger.info(f"Shutdown handler {handler.name} completed in {execution_time:.2f}s")
            
            # Update status
            self.shutdown_status[handler.name] = {
                'status': 'completed',
                'execution_time': execution_time,
                'success': True
            }
            
            return True
            
        except asyncio.TimeoutError:
            logger.error(f"Shutdown handler {handler.name} timed out after {handler.timeout}s")
            self.shutdown_status[handler.name] = {
                'status': 'timeout',
                'execution_time': handler.timeout,
                'success': False
            }
            return False
            
        except Exception as e:
            logger.error(f"Shutdown handler {handler.name} failed: {e}")
            self.shutdown_status[handler.name] = {
                'status': 'error',
                'error': str(e),
                'success': False
            }
            return False
    
    async def _final_cleanup(self):
        """Perform final cleanup operations"""
        try:
            logger.info("Performing final cleanup...")
            
            # Restore original signal handlers
            if self.signals_registered:
                for sig, original_handler in self.original_handlers.items():
                    try:
                        signal.signal(sig, original_handler)
                    except Exception as e:
                        logger.error(f"Error restoring signal handler for {sig}: {e}")
            
            # Log final status
            active_components = len(self.active_components)
            if active_components > 0:
                logger.warning(f"Shutdown completed with {active_components} components still active: {self.active_components}")
            
            total_time = time.time() - self.shutdown_start_time if self.shutdown_start_time else 0
            logger.info(f"Total shutdown time: {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in final cleanup: {e}")
    
    async def _shutdown_monitoring(self):
        """Monitor shutdown progress and provide status updates"""
        try:
            interval = self.config.get('status_update_interval', 5.0)
            
            while not self.shutdown_completed and self.shutdown_requested:
                elapsed = time.time() - self.shutdown_start_time if self.shutdown_start_time else 0
                
                logger.info(f"Shutdown progress - Phase: {self.phase.value}, Elapsed: {elapsed:.1f}s, "
                          f"Active components: {len(self.active_components)}, "
                          f"Completed handlers: {len([h for h in self.shutdown_status.values() if h.get('success')])}")
                
                await asyncio.sleep(interval)
                
        except Exception as e:
            logger.error(f"Error in shutdown monitoring: {e}")
    
    async def _force_shutdown_timer(self, timeout: float):
        """Force shutdown after timeout"""
        try:
            await asyncio.sleep(timeout)
            
            if not self.shutdown_completed:
                logger.error(f"Force shutdown triggered after {timeout}s")
                self._force_shutdown()
                
        except asyncio.CancelledError:
            # Timer was cancelled, normal shutdown completed
            pass
        except Exception as e:
            logger.error(f"Error in force shutdown timer: {e}")
    
    def _force_shutdown(self):
        """Force immediate shutdown"""
        try:
            logger.critical("FORCE SHUTDOWN - Terminating immediately")
            
            # Try to log final status
            if self.shutdown_start_time:
                elapsed = time.time() - self.shutdown_start_time
                logger.critical(f"Force shutdown after {elapsed:.1f}s in phase: {self.phase.value}")
            
            # Force exit
            sys.exit(1)
            
        except Exception as e:
            # Last resort
            import os
            os._exit(1)
    
    def get_shutdown_status(self) -> Dict[str, Any]:
        """Get current shutdown status"""
        try:
            elapsed = time.time() - self.shutdown_start_time if self.shutdown_start_time else 0
            
            return {
                'shutdown_requested': self.shutdown_requested,
                'shutdown_completed': self.shutdown_completed,
                'current_phase': self.phase.value,
                'elapsed_time': elapsed,
                'active_components': list(self.active_components),
                'total_handlers': len(self.shutdown_handlers),
                'completed_handlers': len([h for h in self.shutdown_status.values() if h.get('success')]),
                'failed_handlers': self.failed_handlers,
                'handler_status': self.shutdown_status
            }
            
        except Exception as e:
            return {'error': f'Failed to get shutdown status: {str(e)}'}
    
    def is_shutting_down(self) -> bool:
        """Check if system is shutting down"""
        return self.shutdown_requested
    
    def wait_for_shutdown(self, timeout: float = None) -> bool:
        """Wait for shutdown to complete"""
        try:
            start_time = time.time()
            
            while not self.shutdown_completed:
                if timeout and (time.time() - start_time) > timeout:
                    return False
                
                time.sleep(0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Error waiting for shutdown: {e}")
            return False

# Global shutdown manager instance
_shutdown_manager: Optional[ShutdownManager] = None

def get_shutdown_manager() -> ShutdownManager:
    """Get global shutdown manager instance"""
    global _shutdown_manager
    if _shutdown_manager is None:
        _shutdown_manager = ShutdownManager()
    return _shutdown_manager

def register_shutdown_handler(name: str, handler: Callable, 
                            priority: int = 100, timeout: float = None, 
                            critical: bool = False) -> bool:
    """Register a shutdown handler with the global manager"""
    return get_shutdown_manager().register_shutdown_handler(
        name, handler, priority, timeout, critical
    )

def register_component(component_name: str):
    """Register an active component with the global manager"""
    get_shutdown_manager().register_component(component_name)

def unregister_component(component_name: str):
    """Unregister a component from the global manager"""
    get_shutdown_manager().unregister_component(component_name)

async def initiate_shutdown(reason: str = "Manual shutdown request") -> bool:
    """Initiate graceful shutdown"""
    return await get_shutdown_manager().initiate_shutdown(reason) 