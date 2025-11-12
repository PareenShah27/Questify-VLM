"""
utils/image_preprocessor.py - Image and PDF Processing Utilities

Handles image normalization, PDF conversion, and format handling.
Used by VLMEmbedder from core/vlm_embedding.py
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
from PIL import Image
import io


class ImagePreprocessor:
    """Preprocess images for VLM models."""
    
    def __init__(self, config):
        """Initialize image preprocessor."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}
        self.chunk_size = config.get('vlm.chunk_size', 512)
        self.stride = config.get('vlm.stride', 256)
    
    def load_image(self, image_path: Union[str, Path]) -> Optional[Image.Image]:
        """
        Load image from file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image or None
        """
        try:
            image_path = Path(image_path)
            
            if not image_path.exists():
                self.logger.warning(f"Image not found: {image_path}")
                return None
            
            if image_path.suffix.lower() not in self.supported_formats:
                self.logger.warning(f"Unsupported format: {image_path.suffix}")
                return None
            
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            self.logger.error(f"Error loading image: {e}")
            return None
    
    def preprocess_image(self, image: Image.Image,
                        target_size: Optional[Tuple[int, int]] = None) -> Optional[Image.Image]:
        """
        Preprocess image for model input.
        
        Args:
            image: PIL Image
            target_size: Target size (width, height)
            
        Returns:
            Preprocessed image or None
        """
        try:
            if image is None:
                return None
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if target size provided
            if target_size:
                image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            return image
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {e}")
            return None
    
    def normalize_image(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        Normalize image to numpy array [0, 1].
        
        Args:
            image: PIL Image
            
        Returns:
            Normalized numpy array or None
        """
        try:
            if image is None:
                return None
            
            # Convert to numpy
            arr = np.array(image, dtype=np.float32)
            
            # Normalize to [0, 1]
            arr = arr / 255.0
            
            return arr
        except Exception as e:
            self.logger.error(f"Error normalizing image: {e}")
            return None
    
    def batch_load_images(self, image_paths: List[str]) -> List[Image.Image]:
        """Load multiple images."""
        images = []
        for path in image_paths:
            img = self.load_image(path)
            if img is not None:
                images.append(img)
        return images
    
    def get_image_statistics(self, image: Image.Image) -> Dict[str, Any]:
        """Get statistics about image."""
        try:
            if image is None:
                return {}
            
            arr = np.array(image)
            return {
                'width': image.width,
                'height': image.height,
                'format': image.format,
                'mode': image.mode,
                'size_bytes': len(np.asarray(image).tobytes()),
                'mean_pixel_value': float(np.mean(arr)),
                'std_pixel_value': float(np.std(arr)),
            }
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}


class PDFProcessor:
    """Process PDF files for VLM indexing."""
    
    def __init__(self, config):
        """Initialize PDF processor."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def pdf_to_images(self, pdf_path: Union[str, Path], dpi: int = 150) -> List[Image.Image]:
        """
        Convert PDF to images (one per page).
        
        Args:
            pdf_path: Path to PDF file
            dpi: DPI for conversion
            
        Returns:
            List of PIL Images
        """
        try:
            pdf_path = Path(pdf_path)
            
            if not pdf_path.exists():
                self.logger.warning(f"PDF not found: {pdf_path}")
                return []
            
            # Try to import pdf2image
            try:
                import pdf2image
                images = pdf2image.convert_from_path(str(pdf_path), dpi=dpi)
                self.logger.info(f"Converted PDF to {len(images)} images")
                return images
            except ImportError:
                self.logger.error("pdf2image not installed. Cannot convert PDF.")
                return []
        except Exception as e:
            self.logger.error(f"Error converting PDF: {e}")
            return []
    
    def extract_pdf_pages(self, pdf_path: str) -> Dict[int, Image.Image]:
        """
        Extract PDF pages as images.
        
        Args:
            pdf_path: Path to PDF
            
        Returns:
            Dict mapping page_number -> Image
        """
        try:
            images = self.pdf_to_images(pdf_path)
            return {i: img for i, img in enumerate(images)}
        except Exception as e:
            self.logger.error(f"Error extracting pages: {e}")
            return {}
    
    def get_pdf_info(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """Get PDF information."""
        try:
            pdf_path = Path(pdf_path)
            
            if not pdf_path.exists():
                return {}
            
            images = self.pdf_to_images(pdf_path)
            
            return {
                'file_name': pdf_path.name,
                'file_size_bytes': pdf_path.stat().st_size,
                'num_pages': len(images),
                'page_dimensions': [(img.width, img.height) for img in images] if images else [],
            }
        except Exception as e:
            self.logger.error(f"Error getting PDF info: {e}")
            return {}


class DocumentTypeDetector:
    """Detect document type and handle accordingly."""
    
    def __init__(self, config):
        """Initialize document type detector."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}
        self.pdf_extensions = {'.pdf'}
        self.text_extensions = {'.txt', '.md', '.docx', '.doc'}
    
    def detect_document_type(self, file_path: Union[str, Path]) -> str:
        """
        Detect document type.
        
        Args:
            file_path: Path to file
            
        Returns:
            Document type: 'image', 'pdf', 'text', or 'unknown'
        """
        try:
            file_path = Path(file_path)
            extension = file_path.suffix.lower()
            
            if extension in self.image_extensions:
                return 'image'
            elif extension in self.pdf_extensions:
                return 'pdf'
            elif extension in self.text_extensions:
                return 'text'
            else:
                return 'unknown'
        except Exception as e:
            self.logger.error(f"Error detecting type: {e}")
            return 'unknown'
    
    def is_visual_document(self, file_path: str) -> bool:
        """Check if file is visual document."""
        doc_type = self.detect_document_type(file_path)
        return doc_type in ['image', 'pdf']
    
    def is_text_document(self, file_path: str) -> bool:
        """Check if file is text document."""
        doc_type = self.detect_document_type(file_path)
        return doc_type == 'text'
    
    def batch_detect_types(self, file_paths: List[str]) -> Dict[str, str]:
        """Detect types for multiple files."""
        return {
            file_path: self.detect_document_type(file_path)
            for file_path in file_paths
        }