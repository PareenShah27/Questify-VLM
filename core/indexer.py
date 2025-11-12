"""
core/indexer.py - Document Indexing for Questify VLM

Supports:
- TF-IDF based text document indexing
- VLM-based visual document indexing  
- Hybrid index management
"""

import numpy as np
import logging
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import pickle
from pathlib import Path


class TFIDFIndexer:
    """TF-IDF based text document indexing using scikit-learn."""
    
    def __init__(self, config):
        """Initialize TF-IDF indexer with configuration."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=config.get('text_preprocessing.lowercase', True),
            stop_words='english' if config.get('text_preprocessing.remove_stopwords', True) else None,
            min_df=1,
            max_df=0.9,
            sublinear_tf=True,
            use_idf=True,
        )
        
        self.documents = {}  # doc_id -> text
        self.doc_ids = []    # ordered doc ids
        self.tfidf_matrix = None
        self.vocabulary = {}
        
        self.logger.info("TFIDFIndexer initialized")
    
    def add_documents(self, documents: Dict[str, str]) -> None:
        """
        Add text documents to TF-IDF index.
        
        Args:
            documents: Dict mapping doc_id -> text content
        """
        try:
            self.documents.update(documents)
            self.doc_ids = list(self.documents.keys())
            
            # Rebuild TF-IDF matrix
            texts = [self.documents[doc_id] for doc_id in self.doc_ids]
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            self.vocabulary = self.vectorizer.vocabulary_
            
            self.logger.info(f"Added {len(documents)} documents. Total: {len(self.documents)}")
        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
            raise
    
    def remove_documents(self, doc_ids: List[str]) -> None:
        """Remove documents from index."""
        try:
            for doc_id in doc_ids:
                if doc_id in self.documents:
                    del self.documents[doc_id]
            
            self.doc_ids = list(self.documents.keys())
            
            if len(self.documents) > 0:
                texts = [self.documents[doc_id] for doc_id in self.doc_ids]
                self.tfidf_matrix = self.vectorizer.fit_transform(texts)
                self.vocabulary = self.vectorizer.vocabulary_
            
            self.logger.info(f"Removed {len(doc_ids)} documents")
        except Exception as e:
            self.logger.error(f"Error removing documents: {e}")
            raise
    
    def get_document_vector(self, doc_id: str) -> Optional[np.ndarray]:
        """Get TF-IDF vector for a document."""
        try:
            if doc_id not in self.doc_ids or self.tfidf_matrix is None:
                return None
            
            idx = self.doc_ids.index(doc_id)
            return self.tfidf_matrix.getrow(idx).toarray().flatten()
        except Exception as e:
            self.logger.error(f"Error getting document vector: {e}")
            return None
    
    def get_all_vectors(self) -> np.ndarray:
        """Get all document vectors as matrix."""
        if self.tfidf_matrix is None:
            return np.array([])

        return csr_matrix(self.tfidf_matrix).toarray()
    
    def get_vocabulary(self) -> Dict[str, int]:
        """Get word -> ID vocabulary mapping."""
        return self.vocabulary.copy() if self.vocabulary else {}
    
    def index_size(self) -> int:
        """Get number of indexed documents."""
        return len(self.documents)
    
    def get_document_text(self, doc_id: str) -> Optional[str]:
        """Get original text of a document."""
        return self.documents.get(doc_id)
    
    def list_documents(self) -> List[str]:
        """List all document IDs."""
        return self.doc_ids.copy()


class VLMDocumentIndexer:
    """VLM-based visual document indexing."""
    
    def __init__(self, config, vlm_embedder):
        """Initialize VLM document indexer."""
        self.config = config
        self.vlm_embedder = vlm_embedder
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.documents = {}  # doc_id -> metadata
        self.embeddings = {}  # doc_id -> embedding
        self.doc_ids = []
        
        self.logger.info("VLMDocumentIndexer initialized")
    
    def add_visual_documents(self,
                           image_paths: List[str],
                           doc_ids: List[str],
                           batch_size: int = 1) -> Dict[str, Any]:
        """
        Index visual documents (images/PDFs) with VLM.
        
        Args:
            image_paths: List of paths to images/PDFs
            doc_ids: List of document IDs
            batch_size: Batch size for processing
            
        Returns:
            Dict with success stats
        """
        if len(image_paths) != len(doc_ids):
            raise ValueError("image_paths and doc_ids must have same length")
        
        indexed = 0
        failed = 0
        
        try:
            for i, (image_path, doc_id) in enumerate(zip(image_paths, doc_ids)):
                try:
                    # Generate VLM embedding
                    embedding = self.vlm_embedder.encode_image(image_path)
                    
                    self.documents[doc_id] = {
                        'path': image_path,
                        'indexed_at': None,
                    }
                    self.embeddings[doc_id] = embedding
                    indexed += 1
                    
                except Exception as e:
                    self.logger.warning(f"Failed to index {doc_id}: {e}")
                    failed += 1
            
            self.doc_ids = list(self.documents.keys())
            self.logger.info(f"VLM indexing complete: {indexed} success, {failed} failed")
            
            return {
                'success': failed == 0,
                'indexed': indexed,
                'failed': failed,
            }
        except Exception as e:
            self.logger.error(f"Error in add_visual_documents: {e}")
            raise
    
    def remove_visual_document(self, doc_id: str) -> bool:
        """Remove a visual document."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            del self.embeddings[doc_id]
            self.doc_ids.remove(doc_id)
            return True
        return False
    
    def get_document_embedding(self, doc_id: str) -> Optional[np.ndarray]:
        """Get VLM embedding for a document."""
        return self.embeddings.get(doc_id)
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """Get all visual document embeddings."""
        return self.embeddings.copy()
    
    def index_size(self) -> int:
        """Get number of indexed visual documents."""
        return len(self.documents)
    
    def list_documents(self) -> List[str]:
        """List all visual document IDs."""
        return self.doc_ids.copy()


class HybridIndexManager:
    """Manages both text and VLM indexes."""
    
    def __init__(self, config, tfidf_indexer, vlm_indexer):
        """Initialize hybrid index manager."""
        self.config = config
        self.tfidf_indexer = tfidf_indexer
        self.vlm_indexer = vlm_indexer
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about both indexes."""
        return {
            'text_documents': self.tfidf_indexer.index_size(),
            'visual_documents': self.vlm_indexer.index_size(),
            'total_documents': (
                self.tfidf_indexer.index_size() + 
                self.vlm_indexer.index_size()
            ),
            'text_indexed': self.tfidf_indexer.index_size() > 0,
            'visual_indexed': self.vlm_indexer.index_size() > 0,
        }
    
    def validate_indexes(self) -> bool:
        """Validate both indexes are healthy."""
        try:
            # Check text index
            if self.tfidf_indexer.index_size() > 0:
                if self.tfidf_indexer.get_all_vectors().shape[0] == 0:
                    self.logger.error("Text index validation failed")
                    return False
            
            # Check VLM index
            if self.vlm_indexer.index_size() > 0:
                if len(self.vlm_indexer.get_all_embeddings()) == 0:
                    self.logger.error("VLM index validation failed")
                    return False
            
            self.logger.info("Both indexes validated successfully")
            return True
        except Exception as e:
            self.logger.error(f"Index validation error: {e}")
            return False
    
    def get_all_document_ids(self) -> Dict[str, List[str]]:
        """Get all document IDs from both indexes."""
        return {
            'text_documents': self.tfidf_indexer.list_documents(),
            'visual_documents': self.vlm_indexer.list_documents(),
        }