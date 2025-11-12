"""
Document storage module for Questify search engine.
Manages document storage, retrieval, and file operations.
"""

import os
import json
from typing import Dict, List, Optional, Union
from pathlib import Path


class DocumentStore:
    """Manages document storage and retrieval operations."""
    
    def __init__(self, storage_path: str = "documents"):
        """
        Initialize document store.
        
        Args:
            storage_path: Path to document storage directory
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.documents = {}  # doc_id -> document_content
        self.metadata = {}   # doc_id -> metadata (filename, size, etc.)
        self.index_file = self.storage_path / "index.json"
        
        # Load existing documents on initialization
        self._load_index()
    
    def add_document(self, doc_id: str, content: str, 
                     filename: Optional[str] = None, 
                     metadata: Optional[Dict] = None) -> bool:
        """
        Add a document to the store.
        
        Args:
            doc_id: Unique document identifier
            content: Document content
            filename: Original filename (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            True if document was added successfully
        """
        try:
            # Store document content
            self.documents[doc_id] = content
            
            # Store metadata
            doc_metadata = {
                'filename': filename or f"{doc_id}.txt",
                'size': len(content),
                'added_timestamp': self._get_timestamp(),
                **(metadata or {})
            }
            self.metadata[doc_id] = doc_metadata
            
            # Save to file
            doc_file = self.storage_path / f"{doc_id}.txt"
            with open(doc_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Update index
            self._save_index()
            
            return True
            
        except Exception as e:
            print(f"Error adding document {doc_id}: {e}")
            return False
    
    def add_document_from_file(self, file_path: Union[str, Path], 
                             doc_id: Optional[str] = None) -> Optional[str]:
        """
        Add a document from a file.
        
        Args:
            file_path: Path to the file
            doc_id: Optional document ID (will use filename if not provided)
            
        Returns:
            Document ID if successful, None otherwise
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                print(f"File not found: {file_path}")
                return None
            
            # Generate doc_id if not provided
            if not doc_id:
                doc_id = file_path.stem
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add document
            success = self.add_document(
                doc_id=doc_id,
                content=content,
                filename=file_path.name,
                metadata={
                    'original_path': str(file_path),
                    'file_size': file_path.stat().st_size
                }
            )
            
            return doc_id if success else None
            
        except Exception as e:
            print(f"Error adding document from file {file_path}: {e}")
            return None
    
    def get_document_content(self, doc_id: str) -> Optional[str]:
        """
        Get document content by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document content or None if not found
        """
        return self.documents.get(doc_id)
    
    def get_document_metadata(self, doc_id: str) -> Optional[Dict]:
        """
        Get document metadata by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document metadata or None if not found
        """
        return self.metadata.get(doc_id)
    
    def get_all_documents(self) -> Dict[str, str]:
        """
        Get all documents.
        
        Returns:
            Dictionary mapping doc_id to content
        """
        return self.documents.copy()
    
    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the store.
        
        Args:
            doc_id: Document ID to remove
            
        Returns:
            True if document was removed successfully
        """
        try:
            if doc_id not in self.documents:
                return False
            
            # Remove from memory
            del self.documents[doc_id]
            del self.metadata[doc_id]
            
            # Remove file
            doc_file = self.storage_path / f"{doc_id}.txt"
            if doc_file.exists():
                doc_file.unlink()
            
            # Update index
            self._save_index()
            
            return True
            
        except Exception as e:
            print(f"Error removing document {doc_id}: {e}")
            return False
    
    def list_documents(self) -> List[Dict]:
        """
        List all documents with their metadata.
        
        Returns:
            List of document info dictionaries
        """
        documents_info = []
        for doc_id, content in self.documents.items():
            info = {
                'doc_id': doc_id,
                'content_length': len(content),
                **self.metadata.get(doc_id, {})
            }
            documents_info.append(info)
        
        return documents_info
    
    def search_documents_by_name(self, filename_pattern: str) -> List[str]:
        """
        Search documents by filename pattern.
        
        Args:
            filename_pattern: Pattern to search for in filenames
            
        Returns:
            List of matching document IDs
        """
        matches = []
        pattern_lower = filename_pattern.lower()
        
        for doc_id, metadata in self.metadata.items():
            filename = metadata.get('filename', '').lower()
            if pattern_lower in filename:
                matches.append(doc_id)
        
        return matches
    
    def get_storage_stats(self) -> Dict:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        total_size = sum(len(content) for content in self.documents.values())
        
        return {
            'total_documents': len(self.documents),
            'total_size_chars': total_size,
            'storage_path': str(self.storage_path),
            'avg_document_size': total_size / max(1, len(self.documents))
        }
    
    def _load_index(self) -> None:
        """Load document index from file."""
        try:
            if self.index_file.exists():
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                    
                self.metadata = index_data.get('metadata', {})
                
                # Load document content from individual files
                for doc_id in self.metadata.keys():
                    doc_file = self.storage_path / f"{doc_id}.txt"
                    if doc_file.exists():
                        with open(doc_file, 'r', encoding='utf-8') as f:
                            self.documents[doc_id] = f.read()
                            
        except Exception as e:
            print(f"Error loading index: {e}")
            self.documents = {}
            self.metadata = {}
    
    def _save_index(self) -> None:
        """Save document index to file."""
        try:
            index_data = {
                'metadata': self.metadata,
                'last_updated': self._get_timestamp()
            }
            
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().isoformat()