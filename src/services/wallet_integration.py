"""
Wallet Integration Service

Integrates Smart Money Tracker with existing Helius/QuickNode services
and provides easy wallet list management for copy trading.

Features:
- Easy wallet list import/export
- Integration with Helius/QuickNode services
- Wallet validation and verification
- Performance-based auto-filtering
- Backup and restore functionality
"""

import json
import csv
import logging
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import asyncio

from .smart_money_tracker import SmartMoneyTracker, WalletTransaction

logger = logging.getLogger(__name__)

class WalletIntegration:
    """
    Wallet Integration Service
    
    Provides easy management of smart money wallets and integration
    with blockchain services.
    """
    
    def __init__(self, helius_service=None, quicknode_service=None):
        self.helius_service = helius_service
        self.quicknode_service = quicknode_service
        self.smart_money_tracker = SmartMoneyTracker(helius_service, quicknode_service)
        
        # Configuration
        self.config = {
            'wallet_list_path': 'config/smart_wallets.json',
            'backup_path': 'config/backups/',
            'auto_backup_interval': 3600,  # 1 hour
            'performance_filter_enabled': True,
            'min_performance_score': 60,
            'max_wallets': 100,
            'validation_enabled': True
        }
        
        logger.info("Wallet Integration initialized")
    
    async def load_wallet_list(self, file_path: str = None) -> bool:
        """Load wallet list from file"""
        try:
            if not file_path:
                file_path = self.config['wallet_list_path']
            
            if not Path(file_path).exists():
                logger.warning(f"Wallet list file not found: {file_path}")
                return False
            
            with open(file_path, 'r') as f:
                wallet_data = json.load(f)
            
            wallets = wallet_data.get('wallets', [])
            logger.info(f"Loading {len(wallets)} wallets from {file_path}")
            
            successful_loads = await self.smart_money_tracker.add_multiple_wallets(wallets)
            
            logger.info(f"✅ Successfully loaded {successful_loads}/{len(wallets)} wallets")
            return successful_loads > 0
            
        except Exception as e:
            logger.error(f"Error loading wallet list: {e}")
            return False
    
    async def save_wallet_list(self, file_path: str = None, include_performance: bool = True) -> bool:
        """Save current wallet list to file"""
        try:
            if not file_path:
                file_path = self.config['wallet_list_path']
            
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare wallet data
            wallet_data = {
                'metadata': {
                    'created_timestamp': time.time(),
                    'total_wallets': len(self.smart_money_tracker.tracked_wallets),
                    'performance_included': include_performance
                },
                'wallets': []
            }
            
            # Add wallet information
            for address, wallet_info in self.smart_money_tracker.tracked_wallets.items():
                wallet_entry = {
                    'address': address,
                    'name': wallet_info.get('name', ''),
                    'notes': wallet_info.get('notes', ''),
                    'added_timestamp': wallet_info.get('added_timestamp', time.time())
                }
                
                # Add performance data if requested
                if include_performance:
                    performance = self.smart_money_tracker.get_wallet_performance(address)
                    if performance:
                        wallet_entry['performance'] = {
                            'score': performance.score,
                            'win_rate': performance.win_rate,
                            'total_profit_sol': performance.total_profit_sol,
                            'total_trades': performance.total_trades,
                            'last_updated': performance.last_updated
                        }
                
                wallet_data['wallets'].append(wallet_entry)
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(wallet_data, f, indent=2)
            
            logger.info(f"✅ Saved {len(wallet_data['wallets'])} wallets to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving wallet list: {e}")
            return False
    
    async def import_wallets_from_csv(self, csv_path: str) -> int:
        """Import wallets from CSV file"""
        try:
            wallets_to_add = []
            
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    address = row.get('address', '').strip()
                    name = row.get('name', '').strip()
                    notes = row.get('notes', '').strip()
                    
                    if address:
                        wallets_to_add.append({
                            'address': address,
                            'name': name,
                            'notes': notes
                        })
            
            if wallets_to_add:
                successful_adds = await self.smart_money_tracker.add_multiple_wallets(wallets_to_add)
                logger.info(f"✅ Imported {successful_adds}/{len(wallets_to_add)} wallets from CSV")
                return successful_adds
            else:
                logger.warning("No valid wallets found in CSV file")
                return 0
                
        except Exception as e:
            logger.error(f"Error importing wallets from CSV: {e}")
            return 0
    
    async def export_wallets_to_csv(self, csv_path: str, include_performance: bool = True) -> bool:
        """Export wallets to CSV file"""
        try:
            fieldnames = ['address', 'name', 'notes', 'added_timestamp']
            
            if include_performance:
                fieldnames.extend([
                    'score', 'win_rate', 'total_profit_sol', 
                    'total_trades', 'avg_profit_per_trade'
                ])
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for address, wallet_info in self.smart_money_tracker.tracked_wallets.items():
                    row = {
                        'address': address,
                        'name': wallet_info.get('name', ''),
                        'notes': wallet_info.get('notes', ''),
                        'added_timestamp': wallet_info.get('added_timestamp', time.time())
                    }
                    
                    if include_performance:
                        performance = self.smart_money_tracker.get_wallet_performance(address)
                        if performance:
                            row.update({
                                'score': performance.score,
                                'win_rate': performance.win_rate,
                                'total_profit_sol': performance.total_profit_sol,
                                'total_trades': performance.total_trades,
                                'avg_profit_per_trade': performance.avg_profit_per_trade
                            })
                        else:
                            row.update({
                                'score': 0,
                                'win_rate': 0,
                                'total_profit_sol': 0,
                                'total_trades': 0,
                                'avg_profit_per_trade': 0
                            })
                    
                    writer.writerow(row)
            
            logger.info(f"✅ Exported wallets to {csv_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting wallets to CSV: {e}")
            return False
    
    async def add_wallet_batch(self, wallet_addresses: List[str], 
                              auto_name: bool = True) -> int:
        """Add multiple wallet addresses quickly"""
        try:
            wallets_to_add = []
            
            for i, address in enumerate(wallet_addresses):
                name = f"Smart Wallet {i+1}" if auto_name else ""
                wallets_to_add.append({
                    'address': address.strip(),
                    'name': name,
                    'notes': 'Batch imported'
                })
            
            successful_adds = await self.smart_money_tracker.add_multiple_wallets(wallets_to_add)
            logger.info(f"✅ Added {successful_adds}/{len(wallet_addresses)} wallets in batch")
            return successful_adds
            
        except Exception as e:
            logger.error(f"Error adding wallet batch: {e}")
            return 0
    
    async def validate_and_filter_wallets(self) -> Dict[str, Any]:
        """Validate all tracked wallets and filter by performance"""
        try:
            validation_results = {
                'total_wallets': len(self.smart_money_tracker.tracked_wallets),
                'valid_wallets': 0,
                'invalid_wallets': 0,
                'high_performance_wallets': 0,
                'low_performance_wallets': 0,
                'removed_wallets': []
            }
            
            wallets_to_remove = []
            
            for address in list(self.smart_money_tracker.tracked_wallets.keys()):
                # Validate wallet address
                if not self.smart_money_tracker._is_valid_solana_address(address):
                    validation_results['invalid_wallets'] += 1
                    wallets_to_remove.append(address)
                    continue
                
                validation_results['valid_wallets'] += 1
                
                # Check performance if filtering enabled
                if self.config['performance_filter_enabled']:
                    performance = self.smart_money_tracker.get_wallet_performance(address)
                    
                    if performance and performance.score < self.config['min_performance_score']:
                        validation_results['low_performance_wallets'] += 1
                        wallets_to_remove.append(address)
                    else:
                        validation_results['high_performance_wallets'] += 1
            
            # Remove invalid/low-performance wallets
            for address in wallets_to_remove:
                if address in self.smart_money_tracker.tracked_wallets:
                    wallet_info = self.smart_money_tracker.tracked_wallets[address]
                    validation_results['removed_wallets'].append({
                        'address': address,
                        'name': wallet_info.get('name', ''),
                        'reason': 'invalid' if not self.smart_money_tracker._is_valid_solana_address(address) else 'low_performance'
                    })
                    del self.smart_money_tracker.tracked_wallets[address]
                    
                    if address in self.smart_money_tracker.wallet_performances:
                        del self.smart_money_tracker.wallet_performances[address]
            
            logger.info(f"✅ Validation complete: {validation_results['valid_wallets']} valid, "
                       f"{validation_results['invalid_wallets']} invalid, "
                       f"{len(wallets_to_remove)} removed")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating wallets: {e}")
            return {'error': str(e)}
    
    async def get_wallet_recommendations(self, min_score: float = 80) -> List[Dict[str, Any]]:
        """Get recommended wallets based on performance"""
        try:
            recommendations = []
            
            top_wallets = self.smart_money_tracker.get_top_performing_wallets(limit=20)
            
            for performance in top_wallets:
                if performance.score >= min_score:
                    wallet_info = self.smart_money_tracker.tracked_wallets.get(performance.wallet_address, {})
                    
                    recommendations.append({
                        'address': performance.wallet_address,
                        'name': wallet_info.get('name', ''),
                        'score': performance.score,
                        'win_rate': performance.win_rate,
                        'total_profit_sol': performance.total_profit_sol,
                        'total_trades': performance.total_trades,
                        'recommendation_reason': self._get_recommendation_reason(performance)
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting wallet recommendations: {e}")
            return []
    
    def _get_recommendation_reason(self, performance) -> str:
        """Generate recommendation reason for a wallet"""
        reasons = []
        
        if performance.score >= 90:
            reasons.append("Exceptional performance")
        elif performance.score >= 80:
            reasons.append("Strong performance")
        
        if performance.win_rate >= 0.8:
            reasons.append("High win rate")
        elif performance.win_rate >= 0.7:
            reasons.append("Good win rate")
        
        if performance.total_profit_sol >= 100:
            reasons.append("High profits")
        elif performance.total_profit_sol >= 50:
            reasons.append("Solid profits")
        
        if performance.total_trades >= 50:
            reasons.append("Experienced trader")
        
        return " | ".join(reasons) if reasons else "Recommended"
    
    async def start_monitoring_with_integration(self) -> bool:
        """Start monitoring with full integration"""
        try:
            # Load existing wallet list
            await self.load_wallet_list()
            
            # Validate and filter wallets
            await self.validate_and_filter_wallets()
            
            # Start monitoring
            success = await self.smart_money_tracker.start_monitoring()
            
            if success:
                logger.info("🚀 Smart money monitoring started with integration")
                
                # Schedule auto-backup
                asyncio.create_task(self._auto_backup_loop())
            
            return success
            
        except Exception as e:
            logger.error(f"Error starting monitoring with integration: {e}")
            return False
    
    async def _auto_backup_loop(self):
        """Auto-backup wallet list periodically"""
        try:
            while self.smart_money_tracker.monitoring_active:
                try:
                    await asyncio.sleep(self.config['auto_backup_interval'])
                except asyncio.CancelledError:
                    break
                
                if not self.smart_money_tracker.monitoring_active:
                    break
                
                # Create backup
                backup_path = f"{self.config['backup_path']}wallets_backup_{int(time.time())}.json"
                await self.save_wallet_list(backup_path)
                
                logger.debug(f"Auto-backup created: {backup_path}")
                
        except asyncio.CancelledError:
            logger.info("Auto-backup loop cancelled")
        except Exception as e:
            if self.smart_money_tracker.monitoring_active:
                logger.error(f"Auto-backup loop error: {e}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""
        try:
            monitoring_status = self.smart_money_tracker.get_monitoring_status()
            
            return {
                'integration_active': True,
                'services': {
                    'helius_connected': self.helius_service is not None,
                    'quicknode_connected': self.quicknode_service is not None
                },
                'monitoring': monitoring_status,
                'config': self.config,
                'wallet_recommendations_available': len(self.smart_money_tracker.wallet_performances) > 0
            }
            
        except Exception as e:
            logger.error(f"Error getting integration status: {e}")
            return {'error': str(e)}

# Integration with existing services
async def setup_smart_money_integration(helius_service=None, quicknode_service=None, 
                                       wallet_list_path: str = None) -> WalletIntegration:
    """
    Set up complete smart money wallet integration
    
    Args:
        helius_service: Existing Helius service instance
        quicknode_service: Existing QuickNode service instance
        wallet_list_path: Path to wallet list file
    
    Returns:
        Configured WalletIntegration instance
    """
    try:
        # Create integration instance
        integration = WalletIntegration(helius_service, quicknode_service)
        
        # Configure paths
        if wallet_list_path:
            integration.config['wallet_list_path'] = wallet_list_path
        
        logger.info("✅ Smart money integration setup complete")
        return integration
        
    except Exception as e:
        logger.error(f"Error setting up smart money integration: {e}")
        raise

# Convenience functions for easy wallet management
async def quick_add_wallets(integration: WalletIntegration, addresses: List[str]) -> int:
    """Quickly add multiple wallet addresses"""
    return await integration.add_wallet_batch(addresses)

async def quick_load_from_file(integration: WalletIntegration, file_path: str) -> bool:
    """Quickly load wallets from file"""
    if file_path.endswith('.csv'):
        count = await integration.import_wallets_from_csv(file_path)
        return count > 0
    else:
        return await integration.load_wallet_list(file_path)

# Example wallet list for testing
EXAMPLE_WALLETS = [
    {
        'address': '7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU',
        'name': 'Example Wallet 1',
        'notes': 'High-performance memecoin trader'
    },
    {
        'address': '5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1',
        'name': 'Example Wallet 2', 
        'notes': 'Consistent profitable trades'
    },
    {
        'address': 'DCA265Vj8a9CEuX1eb1LWRnDT7uK6q1xMipnNyatn23M',
        'name': 'Example Wallet 3',
        'notes': 'Early memecoin detector'
    }
] 