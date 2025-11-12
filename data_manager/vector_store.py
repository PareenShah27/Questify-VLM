"""
data_manager/vector_store.py - VLM Embedding Vector Storage

Manages storage and retrieval of VLM embeddings.
Integrates with VLMEmbedder from core/vlm_embedding.py
"""

import logging
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
from datetime import datetime


class VectorStore:
    """Store and manage VLM embedding vectors."""
    
    def __init__(self, config):
        """Initialize vector store."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.vectors = {}  # doc_id -> embedding
        self.metadata = {}  # doc_id -> metadata
        self.doc_ids = []
        
        # Setup storage paths
        self.storage_path = Path(config.get('storage.vector_store_path', 'vector_store'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing vectors if persistence enabled
        if config.get('storage.enable_persistence', True):
            self._load_vectors()
        
        self.logger.info(f"VectorStore initialized. Path: {self.storage_path}")
    
    def add_vector(self, doc_id: str, embedding: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        """
        Add a document embedding vector.
        
        Args:
            doc_id: Unique document ID
            embedding: Embedding vector (1D numpy array)
            metadata: Optional metadata dictionary
            
        Returns:
            True if successful
        """
        try:
            if embedding is None or len(embedding) == 0:
                self.logger.warning(f"Invalid embedding for {doc_id}")
                return False
            
            # Ensure numpy array
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            # Flatten if needed
            if len(embedding.shape) > 1:
                embedding = embedding.flatten()
            
            self.vectors[doc_id] = embedding
            
            # Store metadata
            meta = metadata or {}
            meta['embedding_dim'] = len(embedding)
            meta['embedding_type'] = 'vlm'
            meta['added_at'] = datetime.now().isoformat()
            self.metadata[doc_id] = meta
            
            self.doc_ids = list(self.vectors.keys())
            
            self.logger.info(f"Vector added: {doc_id} (dim={len(embedding)})")
            return True
        except Exception as e:
            self.logger.error(f"Error adding vector: {e}")
            return False
    
    def add_vectors(self, vectors: Dict[str, np.ndarray]) -> Dict[str, bool]:
        """
        Add multiple embedding vectors.
        
        Args:
            vectors: Dict mapping doc_id -> embedding
            
        Returns:
            Dict mapping doc_id -> success boolean
        """
        results = {}
        for doc_id, embedding in vectors.items():
            results[doc_id] = self.add_vector(doc_id, embedding)
        return results
    
    def get_vector(self, doc_id: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Embedding vector or None
        """
        vector = self.vectors.get(doc_id)
        if vector is not None:
            return vector.copy()
        return None
    
    def get_vectors_batch(self, doc_ids: List[str]) -> np.ndarray:
        """
        Get multiple vectors as matrix.
        
        Args:
            doc_ids: List of document IDs
            
        Returns:
            Matrix of embeddings (NxD) or empty array
        """
        try:
            vectors = []
            for doc_id in doc_ids:
                if doc_id in self.vectors:
                    vectors.append(self.vectors[doc_id])
            
            if vectors:
                return np.array(vectors)
            return np.array([])
        except Exception as e:
            self.logger.error(f"Error getting batch: {e}")
            return np.array([])
    
    def get_all_vectors(self) -> Dict[str, np.ndarray]:
        """Get all vectors."""
        return {
            doc_id: vec.copy()
            for doc_id, vec in self.vectors.items()
        }
    
    def remove_vector(self, doc_id: str) -> bool:
        """Remove a vector."""
        try:
            if doc_id in self.vectors:
                del self.vectors[doc_id]
                if doc_id in self.metadata:
                    del self.metadata[doc_id]
                self.doc_ids.remove(doc_id)
                self.logger.info(f"Vector removed: {doc_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error removing vector: {e}")
            return False
    
    def remove_vectors(self, doc_ids: List[str]) -> Dict[str, bool]:
        """Remove multiple vectors."""
        results = {}
        for doc_id in doc_ids:
            results[doc_id] = self.remove_vector(doc_id)
        return results
    
    def vector_exists(self, doc_id: str) -> bool:
        """Check if vector exists."""
        return doc_id in self.vectors
    
    def list_vectors(self) -> List[str]:
        """List all document IDs with vectors."""
        return self.doc_ids.copy()
    
    def get_vector_count(self) -> int:
        """Get number of stored vectors."""
        return len(self.vectors)
    
    def get_vector_dimension(self, doc_id: str) -> Optional[int]:
        """Get dimension of a vector."""
        if doc_id in self.vectors:
            return len(self.vectors[doc_id])
        return None
    
    def get_vector_metadata(self, doc_id: str) -> Optional[Dict]:
        """Get metadata for a vector."""
        return self.metadata.get(doc_id)
    
    def update_vector_metadata(self, doc_id: str, metadata: Dict) -> bool:
        """Update metadata for a vector."""
        try:
            if doc_id not in self.vectors:
                return False
            
            if doc_id not in self.metadata:
                self.metadata[doc_id] = {}
            
            self.metadata[doc_id].update(metadata)
            return True
        except Exception as e:
            self.logger.error(f"Error updating metadata: {e}")
            return False
    
    def similarity_search(self,
                         query_vector: np.ndarray,
                         top_k: int = 10,
                         threshold: float = 0.0) -> List[Tuple[str, float]]:
        """
        Find most similar vectors to query.
        
        Uses cosine similarity (from core/similarity.py internally).
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (doc_id, similarity_score) tuples
        """
        try:
            if len(self.vectors) == 0:
                return []
            
            results = []
            
            for doc_id, doc_vector in self.vectors.items():
                # Cosine similarity
                dot_product = np.dot(query_vector, doc_vector)
                norm_query = np.linalg.norm(query_vector)
                norm_doc = np.linalg.norm(doc_vector)
                
                if norm_query > 0 and norm_doc > 0:
                    similarity = dot_product / (norm_query * norm_doc)
                else:
                    similarity = 0.0
                
                if similarity >= threshold:
                    results.append((doc_id, float(similarity)))
            
            # Sort by similarity descending
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results[:top_k]
        except Exception as e:
            self.logger.error(f"Error in similarity search: {e}")
            return []
    
    def save_vectors_to_disk(self) -> bool:
        """Save vectors to disk."""
        try:
            format = self.config.get('storage.persistence_format', 'pickle')
            compression = self.config.get('storage.compression_enabled', False)
            
            if format == 'json':
                # Convert numpy arrays to lists for JSON
                vectors_serializable = {
                    doc_id: vec.tolist()
                    for doc_id, vec in self.vectors.items()
                }
                
                file_path = self.storage_path / 'vectors.json'
                with open(file_path, 'w') as f:
                    json.dump({
                        'vectors': vectors_serializable,
                        'metadata': self.metadata,
                    }, f)
            else:  # pickle
                file_path = self.storage_path / 'vectors.pkl'
                with open(file_path, 'wb') as f:
                    pickle.dump({
                        'vectors': self.vectors,
                        'metadata': self.metadata,
                    }, f)
            
            self.logger.info(f"Vectors saved to: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving vectors: {e}")
            return False
    
    def _load_vectors(self) -> bool:
        """Load vectors from disk."""
        try:
            format = self.config.get('storage.persistence_format', 'pickle')
            
            if format == 'json':
                file_path = self.storage_path / 'vectors.json'
            else:
                file_path = self.storage_path / 'vectors.pkl'
            
            if not file_path.exists():
                return False
            
            if format == 'json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Convert lists back to numpy arrays
                self.vectors = {
                    doc_id: np.array(vec)
                    for doc_id, vec in data.get('vectors', {}).items()
                }
            else:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                self.vectors = data.get('vectors', {})
            
            self.metadata = data.get('metadata', {})
            self.doc_ids = list(self.vectors.keys())
            
            self.logger.info(f"Loaded {len(self.vectors)} vectors from disk")
            return True
        except Exception as e:
            self.logger.warning(f"Could not load vectors: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all vectors."""
        try:
            self.vectors.clear()
            self.metadata.clear()
            self.doc_ids.clear()
            self.logger.info("VectorStore cleared")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing store: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics."""
        try:
            if len(self.vectors) == 0:
                return {
                    'total_vectors': 0,
                    'avg_dimension': 0,
                    'memory_estimate_mb': 0.0,
                }
            
            # Get average dimension
            dimensions = [len(v) for v in self.vectors.values()]
            avg_dim = sum(dimensions) / len(dimensions)
            
            # Estimate memory usage
            memory_bytes = sum(v.nbytes for v in self.vectors.values())
            memory_mb = memory_bytes / (1024 * 1024)
            
            return {
                'total_vectors': len(self.vectors),
                'avg_dimension': avg_dim,
                'min_dimension': min(dimensions),
                'max_dimension': max(dimensions),
                'memory_estimate_mb': memory_mb,
                'doc_ids': self.doc_ids,
            }
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}