"""
Token manager for handling token metadata and caching.
"""
import os
import json
import time
import logging
from typing import Dict, List, Optional, Any
import aiofiles
from pathlib import Path

logger = logging.getLogger(__name__)

class TokenManager:
    """
    Manages token metadata and caching.
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize token manager."""
        self.known_tokens = {}
        self.data_dir = data_dir
        self.tokens_file = os.path.join(self.data_dir, "known_tokens.json")
        self._ensure_data_dir()
        self._load_known_tokens()
        
    def _ensure_data_dir(self) -> None:
        """Ensure data directory exists."""
        os.makedirs(self.data_dir, exist_ok=True)
        
    def _load_known_tokens(self) -> None:
        """Load known tokens from file."""
        try:
            if not os.path.exists(self.tokens_file):
                logger.warning("Known tokens file not found, creating new file")
                # Initialize with empty dict and save
                self._save_known_tokens()
                return
                
            with open(self.tokens_file, 'r') as f:
                self.known_tokens = json.load(f)
                logger.info(f"Loaded {len(self.known_tokens)} known tokens")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load known tokens: {str(e)}")
            # Initialize with empty dict
            self.known_tokens = {}
            # Try to create a new file
            self._save_known_tokens()
        except Exception as e:
            logger.error(f"Error loading known tokens: {str(e)}")
            # Initialize with empty dict
            self.known_tokens = {}
            
    def _save_known_tokens(self) -> None:
        """Save known tokens to file."""
        try:
            # Ensure directory exists
            self._ensure_data_dir()
            
            # Create temp file path
            temp_file = f"{self.tokens_file}.tmp"
            
            # First write to a temp file
            with open(temp_file, 'w') as f:
                json.dump(self.known_tokens, f, indent=2)
                
            # Ensure all data is written to disk
            os.fsync(f.fileno())
            
            # Then rename to final file (atomic operation on most filesystems)
            os.replace(temp_file, self.tokens_file)
            
            # Set permissions to ensure it's readable and writable
            os.chmod(self.tokens_file, 0o666)
            
            logger.debug(f"Saved {len(self.known_tokens)} known tokens")
        except (IOError, PermissionError) as e:
            logger.error(f"Failed to save known tokens (permission error): {str(e)}")
        except Exception as e:
            logger.error(f"Error saving known tokens: {str(e)}")
            
    async def async_save_known_tokens(self) -> None:
        """Save known tokens asynchronously."""
        try:
            # Ensure directory exists
            self._ensure_data_dir()
            
            # Create temp file path
            temp_file = f"{self.tokens_file}.tmp"
            
            # First write to a temp file
            async with aiofiles.open(temp_file, 'w') as f:
                await f.write(json.dumps(self.known_tokens, indent=2))
                
            # Then rename to final file (atomic operation on most filesystems)
            os.replace(temp_file, self.tokens_file)
            
            # Set permissions to ensure it's readable and writable
            os.chmod(self.tokens_file, 0o666)
            
            logger.debug(f"Saved {len(self.known_tokens)} known tokens asynchronously")
        except Exception as e:
            logger.error(f"Error saving known tokens asynchronously: {str(e)}")
            
    def get_token(self, token_address: str) -> Optional[Dict]:
        """Get token metadata by address."""
        return self.known_tokens.get(token_address)
        
    def add_token(self, token_address: str, metadata: Dict) -> None:
        """Add token metadata."""
        self.known_tokens[token_address] = {
            **metadata,
            "updated_at": time.time()
        }
        
        # Save to file
        self._save_known_tokens()
        
    def update_token(self, token_address: str, updates: Dict) -> None:
        """Update token metadata."""
        if token_address in self.known_tokens:
            self.known_tokens[token_address] = {
                **self.known_tokens[token_address],
                **updates,
                "updated_at": time.time()
            }
            
            # Save to file
            self._save_known_tokens()
            
    def list_tokens(self) -> List[Dict]:
        """List all known tokens."""
        return [
            {"address": address, **metadata}
            for address, metadata in self.known_tokens.items()
        ]
        
    def clear_cache(self) -> None:
        """Clear token cache."""
        self.known_tokens = {}
        self._save_known_tokens()
        logger.info("Token cache cleared")
        
    def get_tokens_updated_since(self, timestamp: float) -> List[Dict]:
        """Get tokens updated since timestamp."""
        return [
            {"address": address, **metadata}
            for address, metadata in self.known_tokens.items()
            if metadata.get("updated_at", 0) > timestamp
        ]
        
    def close(self) -> None:
        """Close token manager and save data."""
        self._save_known_tokens()
        logger.info("Token manager closed") 