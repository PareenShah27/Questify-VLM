"""
data_manager/document_store.py - Text Document Storage

Manages storage and retrieval of text documents.
Integrates with TFIDFIndexer from core/indexer.py
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from datetime import datetime


class DocumentStore:
    """Store and manage text documents."""
    
    def __init__(self, config):
        """Initialize document store."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.documents = {}  # doc_id -> document_data
        self.metadata = {}   # doc_id -> metadata
        self.doc_ids = []
        
        # Setup storage paths
        self.storage_path = Path(config.get('storage.documents_path', 'documents'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing documents if persistence enabled
        if config.get('storage.enable_persistence', True):
            self._load_documents()
        
        self.logger.info(f"DocumentStore initialized. Path: {self.storage_path}")
    
    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """
        Add a document to the store.
        
        Args:
            doc_id: Unique document ID
            content: Document text content
            metadata: Optional metadata dictionary
            
        Returns:
            True if successful
        """
        try:
            if not doc_id or not content:
                self.logger.warning("Document ID and content cannot be empty")
                return False
            
            self.documents[doc_id] = {
                'content': content,
                'added_at': datetime.now().isoformat(),
            }
            
            # Store metadata
            meta = metadata or {}
            meta['size'] = len(content)
            meta['word_count'] = len(content.split())
            self.metadata[doc_id] = meta
            
            self.doc_ids = list(self.documents.keys())
            
            self.logger.info(f"Document added: {doc_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error adding document: {e}")
            return False
    
    def add_documents(self, documents: Dict[str, str]) -> Dict[str, bool]:
        """
        Add multiple documents.
        
        Args:
            documents: Dict mapping doc_id -> content
            
        Returns:
            Dict mapping doc_id -> success boolean
        """
        results = {}
        for doc_id, content in documents.items():
            results[doc_id] = self.add_document(doc_id, content)
        return results
    
    def get_document(self, doc_id: str) -> Optional[str]:
        """
        Get document content by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document content or None
        """
        return self.documents.get(doc_id, {}).get('content')
    
    def get_document_with_metadata(self, doc_id: str) -> Optional[Dict]:
        """Get document with metadata."""
        if doc_id not in self.documents:
            return None
        
        return {
            'doc_id': doc_id,
            'content': self.documents[doc_id]['content'],
            'metadata': self.metadata.get(doc_id, {}),
            'added_at': self.documents[doc_id]['added_at'],
        }
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove a document."""
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
    
    def get_all_documents(self) -> Dict[str, str]:
        """
        Get all documents.
        
        Returns:
            Dict mapping doc_id -> content
        """
        return {
            doc_id: data['content']
            for doc_id, data in self.documents.items()
        }
    
    def list_documents(self) -> List[str]:
        """List all document IDs."""
        return self.doc_ids.copy()
    
    def document_exists(self, doc_id: str) -> bool:
        """Check if document exists."""
        return doc_id in self.documents
    
    def get_document_count(self) -> int:
        """Get total number of documents."""
        return len(self.documents)
    
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
    
    def search_documents(self, query: str) -> List[str]:
        """
        Simple text search across documents.
        
        Args:
            query: Search query
            
        Returns:
            List of matching doc_ids
        """
        try:
            query_lower = query.lower()
            matching = []
            
            for doc_id, data in self.documents.items():
                content_lower = data['content'].lower()
                if query_lower in content_lower:
                    matching.append(doc_id)
            
            return matching
        except Exception as e:
            self.logger.error(f"Error searching documents: {e}")
            return []
    
    def save_to_disk(self) -> bool:
        """Save documents to disk."""
        try:
            format = self.config.get('storage.persistence_format', 'pickle')
            
            if format == 'json':
                file_path = self.storage_path / 'documents.json'
                with open(file_path, 'w') as f:
                    json.dump({
                        'documents': self.documents,
                        'metadata': self.metadata,
                    }, f)
            else:  # pickle
                file_path = self.storage_path / 'documents.pkl'
                with open(file_path, 'wb') as f:
                    pickle.dump({
                        'documents': self.documents,
                        'metadata': self.metadata,
                    }, f)
            
            self.logger.info(f"Documents saved to: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving documents: {e}")
            return False
    
    def _load_documents(self) -> bool:
        """Load documents from disk."""
        try:
            format = self.config.get('storage.persistence_format', 'pickle')
            
            if format == 'json':
                file_path = self.storage_path / 'documents.json'
            else:
                file_path = self.storage_path / 'documents.pkl'
            
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
            
            self.logger.info(f"Loaded {len(self.documents)} documents from disk")
            return True
        except Exception as e:
            self.logger.warning(f"Could not load documents: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all documents."""
        try:
            self.documents.clear()
            self.metadata.clear()
            self.doc_ids.clear()
            self.logger.info("DocumentStore cleared")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing store: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics."""
        try:
            total_size = sum(
                len(data['content'])
                for data in self.documents.values()
            )
            
            avg_size = total_size / len(self.documents) if self.documents else 0
            
            return {
                'total_documents': len(self.documents),
                'total_size_bytes': total_size,
                'average_document_size': avg_size,
                'doc_ids': self.doc_ids,
            }
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}