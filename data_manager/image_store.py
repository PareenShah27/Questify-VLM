"""
data_manager/image_store.py - Visual Document Storage

Manages storage and retrieval of visual documents (images, PDFs).
Integrates with VLMDocumentIndexer from core/indexer.py
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json
from datetime import datetime
import hashlib


class ImageStore:
    """Store and manage visual documents (images, PDFs)."""
    
    def __init__(self, config):
        """Initialize image store."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.documents = {}  # doc_id -> document_data
        self.metadata = {}   # doc_id -> metadata
        self.doc_ids = []
        
        # Setup storage paths
        self.storage_path = Path(config.get('storage.images_path', 'images'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing metadata if persistence enabled
        if config.get('storage.enable_persistence', True):
            self._load_metadata()
        
        self.logger.info(f"ImageStore initialized. Path: {self.storage_path}")
    
    def add_image(self, doc_id: str, image_path: Union[str, Path], metadata: Optional[Dict] = None) -> bool:
        """
        Add an image document.
        
        Args:
            doc_id: Unique document ID
            image_path: Path to image file
            metadata: Optional metadata dictionary
            
        Returns:
            True if successful
        """
        try:
            image_path = Path(image_path)
            
            if not image_path.exists():
                self.logger.warning(f"Image file not found: {image_path}")
                return False
            
            # Store reference to image
            self.documents[doc_id] = {
                'path': str(image_path),
                'format': image_path.suffix.lower(),
                'size_bytes': image_path.stat().st_size,
                'added_at': datetime.now().isoformat(),
            }
            
            # Store metadata
            meta = metadata or {}
            meta['type'] = 'image'
            meta['file_size'] = image_path.stat().st_size
            meta['file_hash'] = self._compute_file_hash(image_path)
            self.metadata[doc_id] = meta
            
            self.doc_ids = list(self.documents.keys())
            
            self.logger.info(f"Image added: {doc_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error adding image: {e}")
            return False
    
    def add_pdf(self, doc_id: str, pdf_path: Union[str, Path], metadata: Optional[Dict] = None) -> bool:
        """
        Add a PDF document.
        
        Args:
            doc_id: Unique document ID
            pdf_path: Path to PDF file
            metadata: Optional metadata dictionary
            
        Returns:
            True if successful
        """
        try:
            pdf_path = Path(pdf_path)
            
            if not pdf_path.exists():
                self.logger.warning(f"PDF file not found: {pdf_path}")
                return False
            
            self.documents[doc_id] = {
                'path': str(pdf_path),
                'format': '.pdf',
                'size_bytes': pdf_path.stat().st_size,
                'added_at': datetime.now().isoformat(),
            }
            
            # Store metadata
            meta = metadata or {}
            meta['type'] = 'pdf'
            meta['file_size'] = pdf_path.stat().st_size
            meta['file_hash'] = self._compute_file_hash(pdf_path)
            self.metadata[doc_id] = meta
            
            self.doc_ids = list(self.documents.keys())
            
            self.logger.info(f"PDF added: {doc_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error adding PDF: {e}")
            return False
    
    def add_visual_documents(self, docs: Dict[str, str]) -> Dict[str, bool]:
        """
        Add multiple visual documents.
        
        Args:
            docs: Dict mapping doc_id -> file_path
            
        Returns:
            Dict mapping doc_id -> success boolean
        """
        results = {}
        for doc_id, file_path in docs.items():
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.pdf':
                results[doc_id] = self.add_pdf(doc_id, str(file_path))
            else:
                results[doc_id] = self.add_image(doc_id, str(file_path))
        
        return results
    
    def get_document_path(self, doc_id: str) -> Optional[str]:
        """Get file path for a visual document."""
        return self.documents.get(doc_id, {}).get('path')
    
    def get_document_info(self, doc_id: str) -> Optional[Dict]:
        """Get document information."""
        if doc_id not in self.documents:
            return None
        
        return {
            'doc_id': doc_id,
            'path': self.documents[doc_id]['path'],
            'format': self.documents[doc_id].get('format'),
            'size_bytes': self.documents[doc_id].get('size_bytes'),
            'metadata': self.metadata.get(doc_id, {}),
            'added_at': self.documents[doc_id]['added_at'],
        }
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove a visual document."""
        try:
            if doc_id in self.documents:
                del self.documents[doc_id]
                if doc_id in self.metadata:
                    del self.metadata[doc_id]
                self.doc_ids.remove(doc_id)
                self.logger.info(f"Document removed: {doc_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error removing document: {e}")
            return False
    
    def remove_documents(self, doc_ids: List[str]) -> Dict[str, bool]:
        """Remove multiple documents."""
        results = {}
        for doc_id in doc_ids:
            results[doc_id] = self.remove_document(doc_id)
        return results
    
    def list_documents(self) -> List[str]:
        """List all document IDs."""
        return self.doc_ids.copy()
    
    def document_exists(self, doc_id: str) -> bool:
        """Check if document exists."""
        return doc_id in self.documents
    
    def get_document_count(self) -> int:
        """Get total number of documents."""
        return len(self.documents)
    
    def get_documents_by_type(self, doc_type: str) -> List[str]:
        """
        Get documents by type (image, pdf).
        
        Args:
            doc_type: Type filter ('image' or 'pdf')
            
        Returns:
            List of matching doc_ids
        """
        matching = []
        for doc_id, meta in self.metadata.items():
            if meta.get('type') == doc_type:
                matching.append(doc_id)
        return matching
    
    def get_document_metadata(self, doc_id: str) -> Optional[Dict]:
        """Get metadata for a document."""
        return self.metadata.get(doc_id)
    
    def update_document_metadata(self, doc_id: str, metadata: Dict) -> bool:
        """Update metadata for a document."""
        try:
            if doc_id not in self.documents:
                return False
            
            if doc_id not in self.metadata:
                self.metadata[doc_id] = {}
            
            self.metadata[doc_id].update(metadata)
            self.logger.info(f"Metadata updated for: {doc_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating metadata: {e}")
            return False
    
    def save_metadata_to_disk(self) -> bool:
        """Save metadata to disk."""
        try:
            format = self.config.get('storage.persistence_format', 'pickle')
            
            if format == 'json':
                file_path = self.storage_path / 'metadata.json'
                with open(file_path, 'w') as f:
                    json.dump({
                        'documents': self.documents,
                        'metadata': self.metadata,
                    }, f)
            else:  # pickle
                file_path = self.storage_path / 'metadata.pkl'
                with open(file_path, 'wb') as f:
                    pickle.dump({
                        'documents': self.documents,
                        'metadata': self.metadata,
                    }, f)
            
            self.logger.info(f"Metadata saved to: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
            return False
    
    def _load_metadata(self) -> bool:
        """Load metadata from disk."""
        try:
            format = self.config.get('storage.persistence_format', 'pickle')
            
            if format == 'json':
                file_path = self.storage_path / 'metadata.json'
            else:
                file_path = self.storage_path / 'metadata.pkl'
            
            if not file_path.exists():
                return False
            
            if format == 'json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
            else:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            
            self.documents = data.get('documents', {})
            self.metadata = data.get('metadata', {})
            self.doc_ids = list(self.documents.keys())
            
            self.logger.info(f"Loaded {len(self.documents)} documents metadata")
            return True
        except Exception as e:
            self.logger.warning(f"Could not load metadata: {e}")
            return False
    
    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Compute MD5 hash of file."""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics."""
        try:
            total_size = sum(
                data.get('size_bytes', 0)
                for data in self.documents.values()
            )
            
            image_count = len(self.get_documents_by_type('image'))
            pdf_count = len(self.get_documents_by_type('pdf'))
            
            return {
                'total_documents': len(self.documents),
                'image_documents': image_count,
                'pdf_documents': pdf_count,
                'total_size_bytes': total_size,
                'average_document_size': total_size / len(self.documents) if self.documents else 0,
                'doc_ids': self.doc_ids,
            }
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}