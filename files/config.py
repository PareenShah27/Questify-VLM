import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class QuestifyConfig:
    """Fixed configuration manager for Questify VLM search engine."""
    
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
            'search_mode': 'text',  # 'text', 'vlm', 'hybrid', or 'auto'
        },
        
        # Storage settings
        'storage': {
            'documents_path': 'documents',
            'images_path': 'images',
            'vector_store_path': 'vector_store',
            'enable_persistence': True,
        },
        
        # Performance settings
        'performance': {
            'enable_caching': True,
            'cache_size_limit': 1000,
        },
        
        # UI settings
        'ui': {
            'page_title': 'Questify VLM - Multimodal Search Engine',
            'results_per_page': 10,
            'enable_file_upload': True,
            'max_file_size_mb': 50,
        },
        
        # VLM settings
        'vlm': {
            'enabled': False,  # False by default, enable when dependencies available
            'model_name': 'vidore/colqwen2-v1.0',
            'device': 'auto',  # 'auto', 'cuda', 'cpu', 'mps'
            'batch_size': 4,
        },
        
        # Hybrid search settings
        'hybrid': {
            'text_weight': 0.5,
            'vlm_weight': 0.5,
        }
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.config_file = Path(config_file) if config_file else Path('config.json')
        self.config = self._deep_copy_dict(self.DEFAULT_CONFIG)
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file if it exists."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                self.config = self._deep_merge(self.DEFAULT_CONFIG, file_config)
                print(f"✓ Configuration loaded from {self.config_file}")
            else:
                print(f"ℹ No config file found at {self.config_file}, using defaults")
        except Exception as e:
            print(f"⚠ Error loading config: {e}, using defaults")
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            print(f"✓ Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"✗ Error saving config: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path (e.g., 'search.max_results')
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
            key_path: Dot-separated path (e.g., 'search.max_results')
            value: Value to set
        """
        keys = key_path.split('.')
        config_ref = self.config
        
        # Navigate to parent
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        config_ref[keys[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self.config.get(section, {}).copy()
    
    def update_section(self, section: str, updates: Dict[str, Any]) -> None:
        """Update configuration section."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section].update(updates)
    
    def is_vlm_enabled(self) -> bool:
        """Check if VLM is enabled."""
        return self.get('vlm.enabled', False)
    
    def get_vlm_model(self) -> str:
        """Get configured VLM model."""
        return self.get('vlm.model_name', 'vidore/colqwen2-v1.0')
    
    def get_search_mode(self) -> str:
        """Get current search mode."""
        return self.get('search.search_mode', 'text')
    
    def set_search_mode(self, mode: str) -> None:
        """
        Set search mode.
        
        Args:
            mode: 'text', 'vlm', 'hybrid', or 'auto'
        """
        valid_modes = ['text', 'vlm', 'hybrid', 'auto']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")
        self.set('search.search_mode', mode)
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Recursively merge two dictionaries."""
        result = self._deep_copy_dict(base)
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    @staticmethod
    def _deep_copy_dict(d: Dict) -> Dict:
        """Create a deep copy of a dictionary."""
        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                result[key] = QuestifyConfig._deep_copy_dict(value)
            else:
                result[key] = value
        return result
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        self.config = self._deep_copy_dict(self.DEFAULT_CONFIG)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return json.dumps(self.config, indent=2)
    
    def __repr__(self) -> str:
        """Representation of configuration."""
        return f"QuestifyConfig(file={self.config_file})"


# FIXED: Global instance for easy import
config = QuestifyConfig()