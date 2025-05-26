"""
Plugin System - Extensibility framework

Provides a plugin architecture for extending the Ant Bot system with:
- External tool integrations
- Custom trading strategies
- Additional data sources
- Monitoring and analytics extensions
- User-defined automations
"""

import logging
import importlib
import inspect
import asyncio
from typing import Dict, List, Optional, Any, Callable, Type
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class PluginInfo:
    """Information about a registered plugin"""
    name: str
    version: str
    description: str
    plugin_type: str
    enabled: bool = True
    priority: int = 0
    dependencies: List[str] = None

class AntBotPlugin(ABC):
    """Base class for all Ant Bot plugins"""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the plugin"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """Cleanup plugin resources"""
        pass
    
    @abstractmethod
    def get_plugin_info(self) -> PluginInfo:
        """Get plugin information"""
        pass

class PluginSystem:
    """
    Plugin System for Ant Bot extensibility
    
    Manages loading, initialization, and execution of plugins that extend
    the core Ant Bot functionality.
    """
    
    def __init__(self):
        self.system_id = "plugin_system"
        self.initialized = False
        
        # Plugin registry
        self.registered_plugins: Dict[str, AntBotPlugin] = {}
        self.plugin_info: Dict[str, PluginInfo] = {}
        
        # Plugin hooks
        self.hooks: Dict[str, List[Callable]] = {}
        
        # Plugin categories
        self.plugin_categories = {
            "trading": [],      # Trading strategy plugins
            "data": [],         # Data source plugins
            "analysis": [],     # Analysis and indicator plugins
            "notification": [], # Notification plugins
            "automation": [],   # Automation plugins
            "integration": []   # External integration plugins
        }
        
        logger.info(f"PluginSystem {self.system_id} created")
    
    async def initialize(self) -> bool:
        """Initialize the plugin system"""
        try:
            logger.info(f"Initializing PluginSystem {self.system_id}...")
            
            # Initialize core hooks
            await self._initialize_core_hooks()
            
            # Auto-discover plugins
            await self._auto_discover_plugins()
            
            # Initialize enabled plugins
            await self._initialize_plugins()
            
            self.initialized = True
            logger.info(f"PluginSystem {self.system_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing PluginSystem {self.system_id}: {e}")
            return False
    
    async def _initialize_core_hooks(self):
        """Initialize core system hooks"""
        core_hooks = [
            "before_trade",
            "after_trade", 
            "before_analysis",
            "after_analysis",
            "on_error",
            "on_system_start",
            "on_system_stop",
            "on_data_received",
            "on_pattern_discovered",
            "on_queen_split",
            "on_worker_created"
        ]
        
        for hook in core_hooks:
            self.hooks[hook] = []
    
    async def _auto_discover_plugins(self):
        """Auto-discover plugins in the plugins directory"""
        try:
            # In a real implementation, this would scan a plugins directory
            # For now, we'll register some example plugins
            await self._register_example_plugins()
            
        except Exception as e:
            logger.error(f"Error auto-discovering plugins: {e}")
    
    async def _register_example_plugins(self):
        """Register example plugins for demonstration"""
        # Example trading strategy plugin
        class ExampleTradingPlugin(AntBotPlugin):
            async def initialize(self) -> bool:
                return True
            
            async def cleanup(self) -> bool:
                return True
            
            def get_plugin_info(self) -> PluginInfo:
                return PluginInfo(
                    name="example_trading",
                    version="1.0.0",
                    description="Example trading strategy plugin",
                    plugin_type="trading",
                    enabled=False  # Disabled by default
                )
        
        # Example notification plugin
        class ExampleNotificationPlugin(AntBotPlugin):
            async def initialize(self) -> bool:
                return True
            
            async def cleanup(self) -> bool:
                return True
            
            def get_plugin_info(self) -> PluginInfo:
                return PluginInfo(
                    name="example_notification",
                    version="1.0.0", 
                    description="Example notification plugin",
                    plugin_type="notification",
                    enabled=False
                )
        
        # Register example plugins
        await self.register_plugin(ExampleTradingPlugin())
        await self.register_plugin(ExampleNotificationPlugin())
    
    async def _initialize_plugins(self):
        """Initialize all enabled plugins"""
        for plugin_name, plugin in self.registered_plugins.items():
            plugin_info = self.plugin_info[plugin_name]
            
            if plugin_info.enabled:
                try:
                    success = await plugin.initialize()
                    if success:
                        logger.info(f"Plugin {plugin_name} initialized successfully")
                    else:
                        logger.warning(f"Plugin {plugin_name} failed to initialize")
                        plugin_info.enabled = False
                        
                except Exception as e:
                    logger.error(f"Error initializing plugin {plugin_name}: {e}")
                    plugin_info.enabled = False
    
    async def register_plugin(self, plugin: AntBotPlugin) -> bool:
        """Register a new plugin"""
        try:
            plugin_info = plugin.get_plugin_info()
            plugin_name = plugin_info.name
            
            # Check if plugin is already registered
            if plugin_name in self.registered_plugins:
                logger.warning(f"Plugin {plugin_name} is already registered")
                return False
            
            # Validate plugin
            if not await self._validate_plugin(plugin):
                logger.error(f"Plugin {plugin_name} failed validation")
                return False
            
            # Register plugin
            self.registered_plugins[plugin_name] = plugin
            self.plugin_info[plugin_name] = plugin_info
            
            # Add to category
            if plugin_info.plugin_type in self.plugin_categories:
                self.plugin_categories[plugin_info.plugin_type].append(plugin_name)
            
            logger.info(f"Plugin {plugin_name} registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error registering plugin: {e}")
            return False
    
    async def _validate_plugin(self, plugin: AntBotPlugin) -> bool:
        """Validate plugin meets requirements"""
        try:
            # Check if plugin implements required methods
            required_methods = ["initialize", "cleanup", "get_plugin_info"]
            
            for method in required_methods:
                if not hasattr(plugin, method):
                    logger.error(f"Plugin missing required method: {method}")
                    return False
            
            # Check plugin info
            plugin_info = plugin.get_plugin_info()
            if not plugin_info.name or not plugin_info.plugin_type:
                logger.error("Plugin missing required info (name or type)")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating plugin: {e}")
            return False
    
    async def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin"""
        try:
            if plugin_name not in self.registered_plugins:
                logger.error(f"Plugin {plugin_name} not found")
                return False
            
            plugin_info = self.plugin_info[plugin_name]
            
            if plugin_info.enabled:
                logger.info(f"Plugin {plugin_name} is already enabled")
                return True
            
            # Initialize plugin
            plugin = self.registered_plugins[plugin_name]
            success = await plugin.initialize()
            
            if success:
                plugin_info.enabled = True
                logger.info(f"Plugin {plugin_name} enabled successfully")
                return True
            else:
                logger.error(f"Failed to enable plugin {plugin_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error enabling plugin {plugin_name}: {e}")
            return False
    
    async def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin"""
        try:
            if plugin_name not in self.registered_plugins:
                logger.error(f"Plugin {plugin_name} not found")
                return False
            
            plugin_info = self.plugin_info[plugin_name]
            
            if not plugin_info.enabled:
                logger.info(f"Plugin {plugin_name} is already disabled")
                return True
            
            # Cleanup plugin
            plugin = self.registered_plugins[plugin_name]
            success = await plugin.cleanup()
            
            plugin_info.enabled = False
            
            if success:
                logger.info(f"Plugin {plugin_name} disabled successfully")
            else:
                logger.warning(f"Plugin {plugin_name} disabled but cleanup failed")
            
            return True
            
        except Exception as e:
            logger.error(f"Error disabling plugin {plugin_name}: {e}")
            return False
    
    def register_hook(self, hook_name: str, callback: Callable) -> bool:
        """Register a callback for a hook"""
        try:
            if hook_name not in self.hooks:
                self.hooks[hook_name] = []
            
            self.hooks[hook_name].append(callback)
            logger.debug(f"Callback registered for hook {hook_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering hook {hook_name}: {e}")
            return False
    
    async def trigger_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Trigger all callbacks for a hook"""
        results = []
        
        try:
            if hook_name not in self.hooks:
                return results
            
            for callback in self.hooks[hook_name]:
                try:
                    if inspect.iscoroutinefunction(callback):
                        result = await callback(*args, **kwargs)
                    else:
                        result = callback(*args, **kwargs)
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error in hook {hook_name} callback: {e}")
                    results.append(None)
            
            return results
            
        except Exception as e:
            logger.error(f"Error triggering hook {hook_name}: {e}")
            return results
    
    async def execute_plugin_method(self, plugin_name: str, method_name: str, *args, **kwargs) -> Any:
        """Execute a method on a specific plugin"""
        try:
            if plugin_name not in self.registered_plugins:
                logger.error(f"Plugin {plugin_name} not found")
                return None
            
            plugin_info = self.plugin_info[plugin_name]
            if not plugin_info.enabled:
                logger.error(f"Plugin {plugin_name} is not enabled")
                return None
            
            plugin = self.registered_plugins[plugin_name]
            
            if not hasattr(plugin, method_name):
                logger.error(f"Plugin {plugin_name} does not have method {method_name}")
                return None
            
            method = getattr(plugin, method_name)
            
            if inspect.iscoroutinefunction(method):
                return await method(*args, **kwargs)
            else:
                return method(*args, **kwargs)
                
        except Exception as e:
            logger.error(f"Error executing plugin method {plugin_name}.{method_name}: {e}")
            return None
    
    def get_enabled_plugins(self) -> List[str]:
        """Get list of enabled plugin names"""
        return [
            name for name, info in self.plugin_info.items()
            if info.enabled
        ]
    
    def get_plugins_by_category(self, category: str) -> List[str]:
        """Get plugins by category"""
        return self.plugin_categories.get(category, [])
    
    def get_plugin_info_dict(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get plugin information as dictionary"""
        if plugin_name not in self.plugin_info:
            return None
        
        info = self.plugin_info[plugin_name]
        return {
            "name": info.name,
            "version": info.version,
            "description": info.description,
            "plugin_type": info.plugin_type,
            "enabled": info.enabled,
            "priority": info.priority,
            "dependencies": info.dependencies or []
        }
    
    async def cleanup_all_plugins(self):
        """Cleanup all plugins"""
        for plugin_name, plugin in self.registered_plugins.items():
            plugin_info = self.plugin_info[plugin_name]
            
            if plugin_info.enabled:
                try:
                    await plugin.cleanup()
                    logger.info(f"Plugin {plugin_name} cleaned up")
                except Exception as e:
                    logger.error(f"Error cleaning up plugin {plugin_name}: {e}")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get plugin system metrics"""
        total_plugins = len(self.registered_plugins)
        enabled_plugins = len(self.get_enabled_plugins())
        
        category_counts = {
            category: len(plugins) 
            for category, plugins in self.plugin_categories.items()
        }
        
        return {
            "system_id": self.system_id,
            "initialized": self.initialized,
            "total_plugins": total_plugins,
            "enabled_plugins": enabled_plugins,
            "disabled_plugins": total_plugins - enabled_plugins,
            "category_counts": category_counts,
            "total_hooks": len(self.hooks),
            "hook_callbacks": sum(len(callbacks) for callbacks in self.hooks.values())
        } 