"""
utils/image_preprocessor.py - Image Preprocessing for VLM Pipeline

Handles image loading, conversion, and preprocessing for visual document processing.
"""

import os
from PIL import Image
from typing import List, Union, Tuple, Optional
import numpy as np
from pathlib import Path


class ImagePreprocessor:
    """
    Image preprocessing utilities for VLM document processing.
    """
    
    def __init__(
        self,
        supported_formats: Optional[List[str]] = None,
        max_image_size: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize Image Preprocessor.
        
        Args:
            supported_formats: List of supported image formats
            max_image_size: Maximum image dimensions (width, height)
        """
        self.supported_formats = supported_formats or [
            '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'
        ]
        self.max_image_size = max_image_size
    
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load an image from file path.
        
        Args:
            image_path: Path to image file
        
        Returns:
            PIL Image object
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if image_path.suffix.lower() not in self.supported_formats:
            raise ValueError(
                f"Unsupported format: {image_path.suffix}. "
                f"Supported: {self.supported_formats}"
            )
        
        image = Image.open(image_path).convert('RGB')
        
        # Resize if needed
        if self.max_image_size:
            image = self.resize_image(image, self.max_image_size)
        
        return image
    
    def load_images_from_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = False
    ) -> List[Tuple[str, Image.Image]]:
        """
        Load all images from a directory.
        
        Args:
            directory: Directory path
            recursive: Whether to search recursively
        
        Returns:
            List of (filename, image) tuples
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        images = []
        
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for file_path in directory.glob(pattern):
            if file_path.suffix.lower() in self.supported_formats:
                try:
                    image = self.load_image(file_path)
                    images.append((str(file_path), image))
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")
        
        return images
    
    @staticmethod
    def resize_image(
        image: Image.Image,
        max_size: Tuple[int, int],
        maintain_aspect_ratio: bool = True
    ) -> Image.Image:
        """
        Resize image to maximum dimensions.
        
        Args:
            image: PIL Image
            max_size: (max_width, max_height)
            maintain_aspect_ratio: Whether to maintain aspect ratio
        
        Returns:
            Resized PIL Image
        """
        if maintain_aspect_ratio:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        else:
            image = image.resize(max_size, Image.Resampling.LANCZOS)
        
        return image
    
    @staticmethod
    def image_to_array(image: Image.Image) -> np.ndarray:
        """Convert PIL Image to numpy array."""
        return np.array(image)
    
    @staticmethod
    def array_to_image(array: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image."""
        return Image.fromarray(array.astype('uint8'))
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """
        Normalize image array to [0, 1] range.
        
        Args:
            image: Image array
        
        Returns:
            Normalized image array
        """
        return image.astype(np.float32) / 255.0
    
    def validate_image(self, image: Image.Image) -> bool:
        """
        Validate if image meets requirements.
        
        Args:
            image: PIL Image
        
        Returns:
            True if valid, False otherwise
        """
        # Check if image is not empty
        if image.size[0] == 0 or image.size[1] == 0:
            return False
        
        # Check if image is not too large
        if self.max_image_size:
            if image.size[0] > self.max_image_size[0] * 2:
                return False
            if image.size[1] > self.max_image_size[1] * 2:
                return False
        
        return True


class PDFConverter:
    """
    Convert PDF documents to images for VLM processing.
    """
    
    def __init__(self, dpi: int = 200):
        """
        Initialize PDF Converter.
        
        Args:
            dpi: Resolution for PDF to image conversion
        """
        self.dpi = dpi
        
        try:
            from pdf2image import convert_from_path
            self.convert_from_path = convert_from_path
            self.available = True
        except ImportError:
            print("Warning: pdf2image not installed. PDF conversion not available.")
            print("Install with: pip install pdf2image")
            print("Also requires poppler: sudo apt install poppler-utils (Linux)")
            self.available = False
    
    def convert_pdf_to_images(
        self,
        pdf_path: Union[str, Path],
        output_folder: Optional[Union[str, Path]] = None
    ) -> List[Image.Image]:
        """
        Convert PDF to list of images, one per page.
        
        Args:
            pdf_path: Path to PDF file
            output_folder: Optional folder to save images
        
        Returns:
            List of PIL Images, one per page
        """
        if not self.available:
            raise RuntimeError("pdf2image not available. Cannot convert PDFs.")
        
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Convert PDF to images
        images = self.convert_from_path(
            str(pdf_path),
            dpi=self.dpi
        )
        
        # Save images if output folder is specified
        if output_folder:
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
            
            pdf_name = pdf_path.stem
            for i, image in enumerate(images):
                output_path = output_folder / f"{pdf_name}_page_{i+1}.png"
                image.save(output_path, "PNG")
        
        return images
    
    def convert_pdfs_batch(
        self,
        pdf_directory: Union[str, Path],
        output_directory: Union[str, Path]
    ) -> dict:
        """
        Convert all PDFs in a directory to images.
        
        Args:
            pdf_directory: Directory containing PDFs
            output_directory: Directory to save images
        
        Returns:
            Dictionary mapping PDF names to list of image paths
        """
        if not self.available:
            raise RuntimeError("pdf2image not available. Cannot convert PDFs.")
        
        pdf_directory = Path(pdf_directory)
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for pdf_path in pdf_directory.glob("*.pdf"):
            try:
                images = self.convert_pdf_to_images(
                    pdf_path,
                    output_folder=output_directory / pdf_path.stem
                )
                results[pdf_path.stem] = len(images)
                print(f"Converted {pdf_path.name}: {len(images)} pages")
            except Exception as e:
                print(f"Error converting {pdf_path.name}: {e}")
                results[pdf_path.stem] = 0
        
        return results