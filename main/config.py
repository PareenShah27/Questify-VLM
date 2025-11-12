"""
Configuration module for Questify search engine.
Manages application settings and configuration options.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


class QuestifyConfig:
    """Manages configuration settings for Questify search engine."""
    
    DEFAULT_CONFIG = {
        # Text preprocessing settings
        'text_preprocessing': {
            'remove_stopwords': True,
            'min_token_length': 3,
        },
        
        # Search settings
        'search': {
            'max_results': 10,
            'min_similarity_score': 0.01,
        },
        
        # Storage settings
        'storage': {
            'documents_path': 'documents',
            'enable_persistence': True,
        },
        
        # Performance settings
        'performance': {
            'enable_caching': True,
            'cache_size_limit': 1000,
        },
        
        # UI settings
        'ui': {
            'page_title': 'Questify - Text Search Engine',
            'results_per_page': 10,
            'enable_file_upload': True,
            'max_file_size_mb': 10,
        }
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.config_file = Path(config_file) if config_file else Path('config.json')
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Load configuration from file if it exists
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                
                # Deep merge with default config
                self.config = self._deep_merge(self.DEFAULT_CONFIG, file_config)
        except Exception as e:
            print(f"Error loading config from {self.config_file}: {e}")
            print("Using default configuration")
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving config to {self.config_file}: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated key path (e.g., 'search.max_results')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated key path (e.g., 'search.max_results')
            value: Value to set
        """
        keys = key_path.split('.')
        config_ref = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        # Set the value
        config_ref[keys[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name (e.g., 'search')
            
        Returns:
            Configuration section dictionary
        """
        return self.config.get(section, {}).copy()
    
    def update_section(self, section: str, updates: Dict[str, Any]) -> None:
        """
        Update configuration section.
        
        Args:
            section: Section name
            updates: Dictionary of updates to apply
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section].update(updates)
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        self.config = self.DEFAULT_CONFIG.copy()
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """
        Recursively merge two dictionaries.
        
        Args:
            base: Base dictionary
            update: Updates to apply
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return json.dumps(self.config, indent=2, ensure_ascii=False)


# Global configuration instance
config = QuestifyConfig()