"""
core/vlm_embedding.py - VLM Embedding Generation for Questify

Supports ColPali/ColQwen2 models for visual document embeddings.
"""

import logging
import numpy as np
from typing import Optional, List, Union
from pathlib import Path
import warnings


class VLMEmbedder:
    """Generate embeddings using Vision-Language Models."""
    
    def __init__(self, config):
        """Initialize VLM embedder."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.model = None
        self.processor = None
        self.device = config.get('vlm.device', 'auto')
        self.model_name = config.get('vlm.model_name', 'vidore/colqwen2-v1.0')
        self.batch_size = config.get('vlm.batch_size', 1)
        
        self.logger.info(f"VLMEmbedder initialized with model: {self.model_name}")
        
        # Try to initialize model
        if config.get('vlm.enabled', True):
            self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize VLM model and processor."""
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            
            self.logger.info(f"Loading model from {self.model_name}...")
            
            # Determine device
            if self.device == 'auto':
                try:
                    import torch
                    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                except:
                    self.device = 'cpu'
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Load model
            model_kwargs = {}
            if self.config.get('vlm.load_in_8bit', False):
                model_kwargs['load_in_8bit'] = True
            
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"Model loaded on {self.device}")
        except ImportError:
            self.logger.warning("Transformers library not available. VLM disabled.")
            self.model = None
        except Exception as e:
            self.logger.error(f"Error loading VLM model: {e}")
            self.model = None
    
    def encode_text(self, text: str) -> Optional[np.ndarray]:
        """
        Encode text to embedding.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector or None
        """
        if not self.model or not self.processor:
            return None
        
        try:
            # Process text
            inputs = self.processor.tokenizer(text, return_tensors='pt')
            
            # Get embeddings (simplified - actual implementation would vary)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                import torch
                with torch.no_grad():
                    outputs = self.model(**inputs.to(self.device))
                
                # Extract embeddings (last hidden state, mean pooling)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                return embeddings.cpu().numpy()[0]
        except Exception as e:
            self.logger.error(f"Error encoding text: {e}")
            return None
    
    def encode_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Encode image to embedding.
        
        Args:
            image_path: Path to image or PDF
            
        Returns:
            Embedding vector or None
        """
        if not self.model or not self.processor:
            return None
        
        try:
            from PIL import Image
            import torch
            
            image_path = Path(image_path)
            
            # Handle different image formats
            if image_path.suffix.lower() in ['.pdf']:
                # Convert PDF to images first
                images = self._pdf_to_images(image_path)
            else:
                # Load image
                image = Image.open(image_path).convert('RGB')
                images = [image]
            
            # Encode images (simplified)
            embeddings_list = []
            for img in images:
                inputs = self.processor(images=img, return_tensors='pt')
                
                with torch.no_grad():
                    outputs = self.model(**inputs.to(self.device))
                
                # Extract embeddings
                emb = outputs.last_hidden_state.mean(dim=1)
                embeddings_list.append(emb.cpu().numpy()[0])
            
            # Average embeddings if multiple pages
            if embeddings_list:
                return np.mean(embeddings_list, axis=0)
            return None
        except Exception as e:
            self.logger.error(f"Error encoding image: {e}")
            return None
    
    def encode_batch(self, items: List[str]) -> Optional[np.ndarray]:
        """Encode multiple items in batch."""
        try:
            embeddings = []
            for item in items:
                emb = self.encode_text(item)
                if emb is not None:
                    embeddings.append(emb)
            
            if embeddings:
                return np.array(embeddings)
            return None
        except Exception as e:
            self.logger.error(f"Error encoding batch: {e}")
            return None
    
    def _pdf_to_images(self, pdf_path: Path) -> List:
        """Convert PDF to images."""
        try:
            import pdf2image
            images = pdf2image.convert_from_path(str(pdf_path))
            return images
        except ImportError:
            self.logger.warning("pdf2image not available. Cannot convert PDF.")
            return []
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        try:
            dummy_emb = self.encode_text("test")
            if dummy_emb is not None:
                return len(dummy_emb)
            return 768  # Default dimension
        except:
            return 768