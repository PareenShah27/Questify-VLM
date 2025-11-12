"""
core/vlm_embedding.py - Vision Language Model Embedding Generation

This module handles the embedding generation for documents using Vision Language Models.
Based on ColPali architecture using PaliGemma/ColQwen2 as the backbone VLM.
"""

import torch
from PIL import Image
from typing import List, Union, Optional
import numpy as np
from transformers import AutoProcessor
from transformers.utils.import_utils import is_flash_attn_2_available

try:
    from colpali_engine.models import ColQwen2, ColQwen2Processor, ColPali, ColPaliProcessor
    COLPALI_AVAILABLE = True
except ImportError:
    COLPALI_AVAILABLE = False
    print("Warning: colpali-engine not installed. VLM features will not be available.")
    print("Install with: pip install colpali-engine")


class VLMEmbedder:
    """
    Vision Language Model Embedder for generating multi-vector document embeddings.
    
    This class wraps ColPali/ColQwen2 models to generate embeddings for both
    document images and text queries, following the late interaction paradigm.
    """
    
    def __init__(
        self,
        model_name: str = "vidore/colqwen2-v1.0",
        device: str = "auto",
        torch_dtype = torch.bfloat16,
        use_flash_attention: bool = True
    ):
        """
        Initialize the VLM Embedder.
        
        Args:
            model_name: HuggingFace model identifier (default: colqwen2-v1.0)
            device: Device to run the model on ('cuda', 'cpu', 'mps', or 'auto')
            torch_dtype: Data type for model weights
            use_flash_attention: Whether to use flash attention 2 if available
        """
        if not COLPALI_AVAILABLE:
            raise ImportError(
                "colpali-engine is required for VLM functionality. "
                "Install with: pip install colpali-engine"
            )
        
        self.model_name = model_name
        self.device = self._get_device(device)
        self.torch_dtype = torch_dtype
        
        # Determine model class based on model name
        if "colqwen" in model_name.lower():
            self.model_class = ColQwen2
            self.processor_class = ColQwen2Processor
        elif "colpali" in model_name.lower():
            self.model_class = ColPali
            self.processor_class = ColPaliProcessor
        else:
            # Default to ColQwen2
            self.model_class = ColQwen2
            self.processor_class = ColQwen2Processor
        
        # Load model and processor
        attn_implementation = None
        if use_flash_attention and is_flash_attn_2_available():
            attn_implementation = "flash_attention_2"
        
        print(f"Loading VLM model: {model_name}")
        self.model = self.model_class.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            attn_implementation=attn_implementation,
        ).eval()
        
        proc = self.processor_class.from_pretrained(model_name)
        # Some processor.from_pretrained implementations return (processor, extra_info)
        if isinstance(proc, tuple):
            self.processor = proc[0]
        else:
            self.processor = proc
        print(f"VLM model loaded successfully on {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Determine the appropriate device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda:0"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def embed_images(
        self,
        images: List[Union[Image.Image, np.ndarray, str]],
        batch_size: int = 4
    ) -> List[torch.Tensor]:
        """
        Generate multi-vector embeddings for document images.
        
        Args:
            images: List of PIL Images, numpy arrays, or image paths
            batch_size: Batch size for processing
        
        Returns:
            List of embedding tensors, one per image
        """
        # Load images if paths are provided
        processed_images = []
        for img in images:
            if isinstance(img, str):
                processed_images.append(Image.open(img).convert('RGB'))
            elif isinstance(img, np.ndarray):
                processed_images.append(Image.fromarray(img))
            else:
                processed_images.append(img)
        
        # Generate embeddings in batches
        all_embeddings = []
        
        for i in range(0, len(processed_images), batch_size):
            batch_images = processed_images[i:i + batch_size]
            
            # Process images
            batch_inputs = self.processor.process_images(batch_images).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                batch_embeddings = self.model(**batch_inputs)
            
            # Move to CPU and store
            all_embeddings.extend(list(torch.unbind(batch_embeddings.to("cpu"))))
        
        return all_embeddings
    
    def embed_queries(
        self,
        queries: List[str]
    ) -> torch.Tensor:
        """
        Generate multi-vector embeddings for text queries.
        
        Args:
            queries: List of query strings
        
        Returns:
            Tensor of query embeddings
        """
        # Process queries
        batch_queries = self.processor.process_queries(queries).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            query_embeddings = self.model(**batch_queries)
        
        return query_embeddings.to("cpu")
    
    def compute_similarity(
        self,
        query_embeddings: torch.Tensor,
        image_embeddings: Union[List[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute similarity scores between queries and images using the processor's scoring.
        
        Args:
            query_embeddings: Query embedding tensor
            image_embeddings: List of image embedding tensors
        
        Returns:
            Similarity score matrix (num_queries x num_images)
        """
        # Convert list to tensor if needed
        if isinstance(image_embeddings, list):
            # Pad embeddings to same length
            max_len = max(emb.shape[0] for emb in image_embeddings)
            padded_embeddings = []
            for emb in image_embeddings:
                if emb.shape[0] < max_len:
                    padding = torch.zeros(
                        (max_len - emb.shape[0], emb.shape[1]),
                        dtype=emb.dtype
                    )
                    emb = torch.cat([emb, padding], dim=0)
                padded_embeddings.append(emb)
            image_embeddings = torch.stack(padded_embeddings)
        
        # Use processor's scoring method
        scores = self.processor.score_multi_vector(query_embeddings, image_embeddings)
        
        return scores
    
    def get_model_info(self) -> dict:
        """Return information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "dtype": str(self.torch_dtype),
            "model_class": self.model_class.__name__,
            "processor_class": self.processor_class.__name__,
        }


class VLMConfig:
    """Configuration class for VLM embedder."""
    
    DEFAULT_MODELS = {
        "colqwen2-v1.0": "vidore/colqwen2-v1.0",  # Best performance (89.3 on ViDoRe)
        "colqwen2-v0.1": "vidore/colqwen2-v0.1",
        "colpali-v1.3": "vidore/colpali-v1.3",    # Gemma-based
        "colpali-v1.2": "vidore/colpali-v1.2",
        "colsmol-500m": "vidore/colSmol-500M",    # Lightweight option
    }
    
    @classmethod
    def get_model_name(cls, key: str) -> str:
        """Get full model name from key."""
        return cls.DEFAULT_MODELS.get(key, key)