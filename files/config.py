"""
files/config.py - Questify VLM Unified Configuration Management

Comprehensive configuration module supporting:
- Text-based TF-IDF search (original functionality)
- Vision-Language Model (VLM) search (new)
- Hybrid search combining both modes

Features dot-notation access, validation, persistence, and runtime updates.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union


class QuestifyConfig:
    """
    Unified configuration manager for Questify VLM search engine.
    
    Manages settings for text search, VLM search, hybrid search, storage,
    UI, performance, and logging. Provides dot-notation access to settings.
    
    Example:
        >>> config = QuestifyConfig()
        >>> config.get('search.max_results')
        10
        >>> config.set('vlm.enabled', True)
        >>> config.validate()
        True
    """
    
    # Default configuration settings
    DEFAULT_CONFIG = {
        # ========== TEXT PREPROCESSING (Original) ==========
        'text_preprocessing': {
            'remove_stopwords': True,
            'min_token_length': 3,
            'lowercase': True,
            'remove_punctuation': True,
        },
        
        # ========== SEARCH CONFIGURATION ==========
        'search': {
            'max_results': 10,
            'min_similarity_score': 0.01,
            'search_mode': 'auto',  # 'text', 'vlm', 'hybrid', 'auto'
            'default_mode': 'text',  # fallback if 'auto' fails
            'enable_caching': True,
            'cache_size_limit': 1000,
            'timeout_seconds': 30,
        },
        
        # ========== VLM CONFIGURATION (New) ==========
        'vlm': {
            'enabled': True,
            'model_name': 'vidore/colqwen2-v1.0',
            'model_type': 'colqwen2',  # colqwen2, dino
            'device': 'auto',  # 'auto', 'cuda', 'cpu'
            'batch_size': 1,
            'flash_attention': False,
            'load_in_8bit': False,
            'chunk_size': 512,
            'stride': 256,
            'max_tokens': 1024,
            'temperature': 0.7,
        },
        
        # ========== HYBRID SEARCH CONFIGURATION (New) ==========
        'hybrid': {
            'text_weight': 0.5,  # Weight for text search scores
            'vlm_weight': 0.5,   # Weight for VLM search scores
            'ensemble_method': 'rrf',  # 'rrf', 'average', 'max', 'product'
            'diversity_penalty': 0.0,  # 0.0 to 1.0
            'reranking_enabled': False,
            'reranking_top_k': 20,
        },
        
        # ========== STORAGE CONFIGURATION ==========
        'storage': {
            'documents_path': 'documents',
            'images_path': 'images',
            'vector_store_path': 'vector_store',
            'enable_persistence': True,
            'persistence_format': 'pickle',  # 'pickle', 'json'
            'auto_save_interval': 100,  # Save after N operations
            'compression_enabled': False,
        },
        
        # ========== UI CONFIGURATION ==========
        'ui': {
            'page_title': 'Questify VLM - Multimodal Search Engine',
            'theme': 'light',  # 'light', 'dark'
            'layout': 'wide',  # 'wide', 'centered'
            'show_stats': True,
            'show_config': True,
            'show_debug_info': False,
            'results_per_page': 10,
            'max_file_size_mb': 50,
        },
        
        # ========== LOGGING CONFIGURATION ==========
        'logging': {
            'level': 'INFO',  # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
            'enable_file_logging': False,
            'log_file': 'questify.log',
            'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'console_output': True,
        },
        
        # ========== QUALITY ATTRIBUTES (New) ==========
        'quality_attributes': {
            'track_search_time': True,
            'track_relevance': True,
            'track_diversity': True,
            'track_precision_recall': True,
        },
        
        # ========== PERFORMANCE CONFIGURATION ==========
        'performance': {
            'enable_caching': True,
            'cache_ttl_seconds': 3600,
            'index_refresh_interval': 100,
            'lazy_loading': True,
            'parallel_processing': False,
            'num_workers': 4,
        },
    }
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Optional path to JSON config file to load
        """
        # Start with default configuration
        self._config = self._deep_copy(self.DEFAULT_CONFIG)
        self._logger = self._setup_logging()
        
        # Load from file if provided
        if config_file:
            self.load_json(config_file)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key using dot notation (e.g., 'vlm.enabled')
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        
        Example:
            >>> config.get('search.max_results')
            10
            >>> config.get('nonexistent.key', 'default_value')
            'default_value'
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key using dot notation
            value: Value to set
        
        Example:
            >>> config.set('vlm.enabled', True)
            >>> config.set('search.max_results', 20)
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to parent
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        self._logger.info(f"Configuration updated: {key} = {value}")
    
    def validate(self) -> bool:
        """
        Validate configuration for consistency and correctness.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate search mode
            valid_modes = ['text', 'vlm', 'hybrid', 'auto']
            mode = self.get('search.search_mode')
            if mode not in valid_modes:
                self._logger.error(f"Invalid search_mode: {mode}")
                return False
            
            # Validate weights sum to 1.0
            text_w = self.get('hybrid.text_weight', 0)
            vlm_w = self.get('hybrid.vlm_weight', 0)
            if abs((text_w + vlm_w) - 1.0) > 0.01:
                self._logger.warning(f"Hybrid weights don't sum to 1.0: {text_w} + {vlm_w}")
            
            # Validate ensemble method
            valid_ensemble = ['rrf', 'average', 'max', 'product']
            ensemble = self.get('hybrid.ensemble_method')
            if ensemble not in valid_ensemble:
                self._logger.error(f"Invalid ensemble_method: {ensemble}")
                return False
            
            # Validate device
            valid_devices = ['auto', 'cuda', 'cpu']
            device = self.get('vlm.device')
            if device not in valid_devices:
                self._logger.error(f"Invalid VLM device: {device}")
                return False
            
            # Validate positive numbers
            if self.get('search.max_results', 1) <= 0:
                self._logger.error("search.max_results must be positive")
                return False
            
            if self.get('vlm.batch_size', 1) <= 0:
                self._logger.error("vlm.batch_size must be positive")
                return False
            
            self._logger.info("Configuration validation passed")
            return True
        
        except Exception as e:
            self._logger.error(f"Configuration validation error: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export configuration as dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return self._deep_copy(self._config)
    
    def from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Import configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        """
        self._config = self._deep_copy(config_dict)
        self._logger.info("Configuration imported from dictionary")
    
    def load_json(self, config_file: Union[str, Path]) -> None:
        """
        Load configuration from JSON file.
        
        Args:
            config_file: Path to JSON configuration file
        """
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                self._logger.warning(f"Config file not found: {config_file}")
                return
            
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            
            # Deep merge with existing config
            self._deep_merge(self._config, loaded_config)
            self._logger.info(f"Configuration loaded from: {config_file}")
        
        except Exception as e:
            self._logger.error(f"Error loading config file: {e}")
    
    def save_json(self, config_file: Union[str, Path]) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            config_file: Path to save JSON configuration
        """
        try:
            config_path = Path(config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(self._config, f, indent=4)
            
            self._logger.info(f"Configuration saved to: {config_file}")
        
        except Exception as e:
            self._logger.error(f"Error saving config file: {e}")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name (e.g., 'vlm', 'search', 'storage')
        
        Returns:
            Dictionary with section configuration
        """
        return self._deep_copy(self._config.get(section, {}))
    
    def set_section(self, section: str, config_dict: Dict[str, Any]) -> None:
        """
        Set entire configuration section.
        
        Args:
            section: Section name
            config_dict: Configuration dictionary for section
        """
        self._config[section] = self._deep_copy(config_dict)
        self._logger.info(f"Configuration section updated: {section}")
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self._config = self._deep_copy(self.DEFAULT_CONFIG)
        self._logger.info("Configuration reset to defaults")
    
    def summary(self) -> str:
        """
        Get human-readable configuration summary.
        
        Returns:
            Formatted configuration summary string
        """
        summary_lines = [
            "=" * 60,
            "QUESTIFY VLM CONFIGURATION SUMMARY",
            "=" * 60,
        ]
        
        # Text search
        summary_lines.append("\n📝 TEXT SEARCH:")
        summary_lines.append(f"  • Preprocessing: Stopwords={self.get('text_preprocessing.remove_stopwords')}")
        
        # Search
        summary_lines.append("\n🔍 SEARCH SETTINGS:")
        summary_lines.append(f"  • Mode: {self.get('search.search_mode')}")
        summary_lines.append(f"  • Max Results: {self.get('search.max_results')}")
        summary_lines.append(f"  • Caching: {'Enabled' if self.get('search.enable_caching') else 'Disabled'}")
        
        # VLM
        summary_lines.append("\n🖼️  VLM SETTINGS:")
        summary_lines.append(f"  • Enabled: {'Yes' if self.get('vlm.enabled') else 'No'}")
        if self.get('vlm.enabled'):
            summary_lines.append(f"  • Model: {self.get('vlm.model_name')}")
            summary_lines.append(f"  • Device: {self.get('vlm.device')}")
            summary_lines.append(f"  • Batch Size: {self.get('vlm.batch_size')}")
            summary_lines.append(f"  • Flash Attention: {'Enabled' if self.get('vlm.flash_attention') else 'Disabled'}")
        
        # Hybrid
        summary_lines.append("\n⚡ HYBRID SEARCH:")
        summary_lines.append(f"  • Text Weight: {self.get('hybrid.text_weight')}")
        summary_lines.append(f"  • VLM Weight: {self.get('hybrid.vlm_weight')}")
        summary_lines.append(f"  • Ensemble: {self.get('hybrid.ensemble_method')}")
        
        # Storage
        summary_lines.append("\n💾 STORAGE:")
        summary_lines.append(f"  • Documents Path: {self.get('storage.documents_path')}")
        summary_lines.append(f"  • Images Path: {self.get('storage.images_path')}")
        summary_lines.append(f"  • Persistence: {'Enabled' if self.get('storage.enable_persistence') else 'Disabled'}")
        
        summary_lines.append("\n" + "=" * 60)
        
        return "\n".join(summary_lines)
    
    # ==================== PRIVATE METHODS ====================
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for configuration manager."""
        logger = logging.getLogger('QuestifyConfig')
        
        if not logger.handlers:
            level = getattr(logging, self.get('logging.level', 'INFO'))
            logger.setLevel(level)
            
            # Console handler
            if self.get('logging.console_output', True):
                console_handler = logging.StreamHandler()
                console_handler.setLevel(level)
                formatter = logging.Formatter(self.get('logging.log_format', '%(message)s'))
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
        
        return logger
    
    @staticmethod
    def _deep_copy(obj: Any) -> Any:
        """Deep copy a configuration object."""
        if isinstance(obj, dict):
            return {k: QuestifyConfig._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [QuestifyConfig._deep_copy(item) for item in obj]
        else:
            return obj
    
    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> None:
        """Deep merge override dictionary into base dictionary."""
        for key, value in override.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                QuestifyConfig._deep_merge(base[key], value)
            else:
                base[key] = value
    
    # ==================== PROPERTIES ====================
    
    @property
    def vlm_enabled(self) -> bool:
        """Shortcut for checking if VLM is enabled."""
        return self.get('vlm.enabled', False)
    
    @property
    def search_mode(self) -> str:
        """Shortcut for getting search mode."""
        return self.get('search.search_mode', 'auto')
    
    @property
    def max_results(self) -> int:
        """Shortcut for getting max results."""
        return self.get('search.max_results', 10)
    
    @property
    def vlm_model(self) -> str:
        """Shortcut for getting VLM model name."""
        return self.get('vlm.model_name', 'vidore/colqwen2-v1.0')


# Global config instance for convenience
config = QuestifyConfig()


# ==================== CONVENIENCE FUNCTIONS ====================

def get_config() -> QuestifyConfig:
    """Get global configuration instance."""
    global config
    return config


def reset_config() -> None:
    """Reset global configuration to defaults."""
    global config
    config = QuestifyConfig()


if __name__ == '__main__':
    # Example usage
    cfg = QuestifyConfig()
    
    # Print summary
    print(cfg.summary())
    
    # Test access patterns
    print(f"\nAccessing values:")
    print(f"  Search max results: {cfg.get('search.max_results')}")
    print(f"  VLM enabled: {cfg.vlm_enabled}")
    print(f"  VLM model: {cfg.vlm_model}")
    
    # Test validation
    print(f"\nValidation: {'✓ Passed' if cfg.validate() else '✗ Failed'}")
    
    # Test export
    config_dict = cfg.to_dict()
    print(f"\nConfiguration exported: {len(config_dict)} sections")