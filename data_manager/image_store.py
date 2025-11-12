"""
data_manager/image_store.py - Image Document Storage Management

Mirrors document_store.py but handles visual documents (PDFs, images).
Provides consistent interface for document management.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from PIL import Image
import hashlib


class ImageStore:
    """
    Storage and management system for visual documents.
    
    Parallels the text DocumentStore, providing consistent interface
    for loading, storing, and retrieving image-based documents.
    """
    
    def __init__(self, store_dir: Union[str, Path] = "images"):
        """
        Initialize Image Store.
        
        Args:
            store_dir: Directory to store images and metadata
        """
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.store_dir / "metadata.json"
        
        # In-memory storage
        self.documents: Dict[str, dict] = {}  # doc_id -> metadata
        self.document_paths: Dict[str, Path] = {}  # doc_id -> file path
        
        # Load existing metadata
        self.load_metadata()
    
    def add_image(
        self,
        image_path: Union[str, Path],
        doc_id: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Add an image document to the store.
        
        Args:
            image_path: Path to image file
            doc_id: Optional custom document ID (auto-generated if None)
            metadata: Optional metadata dict
        
        Returns:
            Document ID
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Generate doc_id if not provided
        if doc_id is None:
            doc_id = self._generate_doc_id(image_path)
        
        # Store image reference
        self.document_paths[doc_id] = image_path
        
        # Initialize metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'doc_id': doc_id,
            'path': str(image_path),
            'filename': image_path.name,
            'image_format': image_path.suffix.lower(),
            'file_size_bytes': image_path.stat().st_size,
            'hash': self._compute_file_hash(image_path),
            'indexed': False,
            'embedding_id': None
        })
        
        self.documents[doc_id] = metadata
        return doc_id
    
    def add_images_batch(
        self,
        image_directory: Union[str, Path],
        recursive: bool = False,
        supported_formats: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add multiple images from directory.
        
        Args:
            image_directory: Directory containing images
            recursive: Whether to search recursively
            supported_formats: List of supported file extensions
        
        Returns:
            List of added document IDs
        """
        if supported_formats is None:
            supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
        
        image_dir = Path(image_directory)
        doc_ids = []
        
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for file_path in sorted(image_dir.glob(pattern)):
            if file_path.suffix.lower() in supported_formats:
                try:
                    doc_id = self.add_image(file_path)
                    doc_ids.append(doc_id)
                except Exception as e:
                    print(f"Warning: Failed to add {file_path}: {e}")
        
        return doc_ids
    
    def get_document(self, doc_id: str) -> Tuple[Optional[Image.Image], dict]:
        """
        Retrieve a document's image and metadata.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            (PIL Image or None, metadata) tuple
        """
        if doc_id not in self.documents:
            raise KeyError(f"Document not found: {doc_id}")
        
        metadata = self.documents[doc_id]
        image_path = self.document_paths[doc_id]
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = None
        
        return image, metadata
    
    def get_document_path(self, doc_id: str) -> Path:
        """Get file path for a document."""
        if doc_id not in self.document_paths:
            raise KeyError(f"Document not found: {doc_id}")
        return self.document_paths[doc_id]
    
    def get_all_documents(self) -> Dict[str, dict]:
        """Get all document metadata."""
        return self.documents.copy()
    
    def get_all_doc_ids(self) -> List[str]:
        """Get list of all document IDs."""
        return list(self.documents.keys())
    
    def mark_indexed(self, doc_id: str, embedding_id: Optional[str] = None):
        """
        Mark document as indexed by VLM.
        
        Args:
            doc_id: Document ID
            embedding_id: ID of corresponding embedding
        """
        if doc_id in self.documents:
            self.documents[doc_id]['indexed'] = True
            if embedding_id:
                self.documents[doc_id]['embedding_id'] = embedding_id
    
    def get_indexed_documents(self) -> List[str]:
        """Get list of already indexed documents."""
        return [
            doc_id for doc_id, meta in self.documents.items()
            if meta.get('indexed', False)
        ]
    
    def get_unindexed_documents(self) -> List[str]:
        """Get list of documents not yet indexed."""
        return [
            doc_id for doc_id, meta in self.documents.items()
            if not meta.get('indexed', False)
        ]
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove document from store."""
        if doc_id in self.documents:
            del self.documents[doc_id]
        if doc_id in self.document_paths:
            del self.document_paths[doc_id]
        return True
    
    def search_by_metadata(
        self,
        query_key: str,
        query_value: str
    ) -> List[str]:
        """
        Search documents by metadata field.
        
        Args:
            query_key: Metadata field name
            query_value: Value to search for
        
        Returns:
            List of matching document IDs
        """
        results = []
        for doc_id, metadata in self.documents.items():
            if metadata.get(query_key) == query_value:
                results.append(doc_id)
        return results
    
    def get_statistics(self) -> dict:
        """Get statistics about stored documents."""
        indexed_count = len(self.get_indexed_documents())
        total_size = sum(meta.get('file_size_bytes', 0) for meta in self.documents.values())
        
        return {
            'total_documents': len(self.documents),
            'indexed_documents': indexed_count,
            'unindexed_documents': len(self.documents) - indexed_count,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'storage_directory': str(self.store_dir)
        }
    
    def save_metadata(self):
        """Save metadata to JSON file."""
        metadata_export = {}
        for doc_id, meta in self.documents.items():
            # Convert Path to string for JSON serialization
            meta_copy = meta.copy()
            meta_copy['path'] = str(meta_copy['path'])
            metadata_export[doc_id] = meta_copy
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata_export, f, indent=2)
        
        print(f"Saved metadata for {len(self.documents)} documents")
    
    def load_metadata(self):
        """Load metadata from JSON file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata_import = json.load(f)
            
            for doc_id, meta in metadata_import.items():
                self.documents[doc_id] = meta
                self.document_paths[doc_id] = Path(meta['path'])
            
            print(f"Loaded metadata for {len(self.documents)} documents")
    
    @staticmethod
    def _generate_doc_id(file_path: Path) -> str:
        """Generate unique document ID from file."""
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
        return f"{file_path.stem}_{file_hash}"
    
    @staticmethod
    def _compute_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
        """Compute SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def export_catalog(self, output_path: Union[str, Path]):
        """
        Export document catalog to CSV for review.
        
        Args:
            output_path: Path to save CSV file
        """
        import csv
        
        output_path = Path(output_path)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    'doc_id', 'filename', 'image_format',
                    'file_size_mb', 'indexed', 'embedding_id'
                ]
            )
            writer.writeheader()
            
            for doc_id, meta in self.documents.items():
                writer.writerow({
                    'doc_id': doc_id,
                    'filename': meta.get('filename', ''),
                    'image_format': meta.get('image_format', ''),
                    'file_size_mb': round(meta.get('file_size_bytes', 0) / (1024 * 1024), 2),
                    'indexed': meta.get('indexed', False),
                    'embedding_id': meta.get('embedding_id', '')
                })
        
        print(f"Exported catalog to {output_path}")