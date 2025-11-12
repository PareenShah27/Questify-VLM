"""
files/vlm_config.py - Configuration for VLM Components

Central configuration for Vision Language Model settings, model selection,
and quality attributes.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VLMModelConfig:
    """Configuration for VLM model selection and parameters."""
    
    # Model selection
    model_name: str = "vidore/colqwen2-v1.0"  # Default: best performing model
    model_type: str = "colqwen2"  # 'colqwen2', 'colpali', 'colsmol'
    
    # Device settings
    device: str = "auto"  # 'auto', 'cuda', 'cpu', 'mps'
    use_flash_attention: bool = True
    
    # Model precision
    torch_dtype: str = "bfloat16"  # 'bfloat16', 'float16', 'float32'
    
    # Batch processing
    batch_size: int = 4
    max_batch_size: int = 16
    
    # Model-specific parameters
    embedding_dim: int = 128  # ColPali default
    max_patches: int = 1024


@dataclass
class LateInteractionConfig:
    """Configuration for late interaction mechanism."""
    
    # Similarity computation
    similarity_metric: str = "cosine"  # 'cosine' or 'dot_product'
    
    # Token pooling (for compression)
    use_token_pooling: bool = False
    pool_factor: int = 2  # Reduces vectors by this factor
    
    # Top-k patches for interpretability
    top_k_patches: int = 20
    
    # MaxSim operation
    normalize_scores: bool = True


@dataclass
class RetrievalConfig:
    """Configuration for retrieval and ranking."""
    
    # Search parameters
    top_k: int = 10
    similarity_threshold: float = 0.0
    
    # Retrieval mode
    use_late_interaction: bool = True
    use_hybrid_search: bool = False  # Combine with text search
    
    # Re-ranking
    enable_reranking: bool = False
    rerank_top_k: int = 50


@dataclass
class ImageProcessingConfig:
    """Configuration for image preprocessing."""
    
    # Image loading
    supported_formats: Optional[list] = None
    max_image_size: Optional[tuple] = None  # (width, height)
    
    # PDF conversion
    pdf_dpi: int = 200
    convert_pdfs: bool = True
    
    # Image validation
    min_image_size: tuple = (32, 32)
    max_file_size_mb: float = 50.0
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [
                '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'
            ]


@dataclass
class StorageConfig:
    """Configuration for vector storage."""
    
    # Storage paths
    vector_store_dir: str = "vector_store"
    image_store_dir: str = "images"
    models_cache_dir: str = "models"
    
    # Vector indexing
    use_faiss: bool = False
    faiss_index_type: str = "IVF"  # 'Flat', 'IVF', 'HNSW'
    
    # Persistence
    auto_save: bool = True
    save_interval: int = 100  # Save every N documents


@dataclass
class QualityAttributesConfig:
    """
    Configuration for Quality Attributes focus areas:
    - Accuracy / Retrieval Effectiveness
    - Robustness / Generalizability  
    - Performance
    """
    
    # Accuracy / Retrieval Effectiveness
    enable_accuracy_metrics: bool = True
    track_mrr: bool = True  # Mean Reciprocal Rank
    track_ndcg: bool = True  # Normalized Discounted Cumulative Gain
    track_precision_at_k: bool = True
    
    # Robustness / Generalizability
    enable_cross_domain_validation: bool = False
    test_on_degraded_images: bool = False
    multilingual_support: bool = False
    
    # Performance
    enable_performance_metrics: bool = True
    track_indexing_time: bool = True
    track_search_time: bool = True
    track_memory_usage: bool = True
    
    # Benchmarking
    benchmark_mode: bool = False
    log_metrics: bool = True


class VLMConfig:
    """
    Master configuration class combining all VLM settings.
    """
    
    def __init__(
        self,
        model_config: Optional[VLMModelConfig] = None,
        late_interaction_config: Optional[LateInteractionConfig] = None,
        retrieval_config: Optional[RetrievalConfig] = None,
        image_config: Optional[ImageProcessingConfig] = None,
        storage_config: Optional[StorageConfig] = None,
        quality_config: Optional[QualityAttributesConfig] = None
    ):
        """
        Initialize master VLM configuration.
        
        Args:
            model_config: VLM model settings
            late_interaction_config: Late interaction settings
            retrieval_config: Retrieval and ranking settings
            image_config: Image processing settings
            storage_config: Storage settings
            quality_config: Quality attributes settings
        """
        self.model = model_config or VLMModelConfig()
        self.late_interaction = late_interaction_config or LateInteractionConfig()
        self.retrieval = retrieval_config or RetrievalConfig()
        self.image = image_config or ImageProcessingConfig()
        self.storage = storage_config or StorageConfig()
        self.quality = quality_config or QualityAttributesConfig()
    
    def get_model_info(self) -> dict:
        """Get model configuration as dictionary."""
        return {
            'model_name': self.model.model_name,
            'model_type': self.model.model_type,
            'device': self.model.device,
            'batch_size': self.model.batch_size,
            'embedding_dim': self.model.embedding_dim
        }
    
    def get_quality_focus_areas(self) -> dict:
        """Get quality attributes configuration."""
        return {
            'accuracy_tracking': self.quality.enable_accuracy_metrics,
            'robustness_testing': self.quality.enable_cross_domain_validation,
            'performance_monitoring': self.quality.enable_performance_metrics
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create configuration from dictionary."""
        return cls(
            model_config=VLMModelConfig(**config_dict.get('model', {})),
            late_interaction_config=LateInteractionConfig(**config_dict.get('late_interaction', {})),
            retrieval_config=RetrievalConfig(**config_dict.get('retrieval', {})),
            image_config=ImageProcessingConfig(**config_dict.get('image', {})),
            storage_config=StorageConfig(**config_dict.get('storage', {})),
            quality_config=QualityAttributesConfig(**config_dict.get('quality', {}))
        )
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'model': self.model.__dict__,
            'late_interaction': self.late_interaction.__dict__,
            'retrieval': self.retrieval.__dict__,
            'image': self.image.__dict__,
            'storage': self.storage.__dict__,
            'quality': self.quality.__dict__
        }


# Predefined configurations for different use cases

def get_high_accuracy_config() -> VLMConfig:
    """Configuration optimized for maximum retrieval accuracy."""
    return VLMConfig(
        model_config=VLMModelConfig(
            model_name="vidore/colqwen2-v1.0",  # Best accuracy
            batch_size=2  # Smaller batches for precision
        ),
        late_interaction_config=LateInteractionConfig(
            use_token_pooling=False,  # No compression
            similarity_metric="cosine"
        ),
        retrieval_config=RetrievalConfig(
            top_k=20,
            use_late_interaction=True,
            enable_reranking=True
        )
    )


def get_high_performance_config() -> VLMConfig:
    """Configuration optimized for speed and efficiency."""
    return VLMConfig(
        model_config=VLMModelConfig(
            model_name="vidore/colSmol-500M",  # Smaller, faster model
            batch_size=16  # Larger batches
        ),
        late_interaction_config=LateInteractionConfig(
            use_token_pooling=True,  # Compression enabled
            pool_factor=3
        ),
        retrieval_config=RetrievalConfig(
            top_k=10,
            use_late_interaction=True
        ),
        storage_config=StorageConfig(
            use_faiss=True,
            faiss_index_type="IVF"
        )
    )


def get_balanced_config() -> VLMConfig:
    """Balanced configuration for general use."""
    return VLMConfig(
        model_config=VLMModelConfig(
            model_name="vidore/colpali-v1.3",
            batch_size=4
        ),
        late_interaction_config=LateInteractionConfig(
            use_token_pooling=False,
            pool_factor=2
        ),
        retrieval_config=RetrievalConfig(
            top_k=10,
            use_late_interaction=True
        )
    )


def get_development_config() -> VLMConfig:
    """Fast configuration for development and testing."""
    return VLMConfig(
        model_config=VLMModelConfig(
            model_name="vidore/colSmol-256M",  # Smallest model
            batch_size=8,
            device="cpu"  # Can run on CPU
        ),
        quality_config=QualityAttributesConfig(
            benchmark_mode=True,
            log_metrics=True
        )
    )