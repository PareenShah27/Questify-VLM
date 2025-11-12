"""
data_manager/vector_store.py - Vector Storage for VLM Embeddings

Manages storage and retrieval of multi-vector document embeddings.
"""

import torch
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np


class VectorStore:
    """
    Storage and retrieval system for multi-vector document embeddings.
    
    Supports efficient storage of variable-length multi-vector representations
    and fast similarity-based retrieval.
    """
    
    def __init__(self, storage_dir: Union[str, Path] = "vector_store"):
        """
        Initialize Vector Store.
        
        Args:
            storage_dir: Directory to store embeddings and metadata
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.embeddings_file = self.storage_dir / "embeddings.pkl"
        self.metadata_file = self.storage_dir / "metadata.json"
        
        # In-memory storage
        self.doc_embeddings: Dict[str, torch.Tensor] = {}
        self.doc_metadata: Dict[str, dict] = {}
        
        # Load existing data if available
        self.load()
    
    def add_document(
        self,
        doc_id: str,
        embedding: torch.Tensor,
        metadata: Optional[dict] = None
    ):
        """
        Add a document's embedding to the store.
        
        Args:
            doc_id: Unique document identifier
            embedding: Multi-vector embedding tensor
            metadata: Optional metadata (path, page number, etc.)
        """
        self.doc_embeddings[doc_id] = embedding.cpu()
        
        if metadata is None:
            metadata = {}
        
        metadata['doc_id'] = doc_id
        metadata['embedding_shape'] = list(embedding.shape)
        self.doc_metadata[doc_id] = metadata
    
    def add_documents_batch(
        self,
        doc_ids: List[str],
        embeddings: List[torch.Tensor],
        metadata_list: Optional[List[dict]] = None
    ):
        """
        Add multiple documents in batch.
        
        Args:
            doc_ids: List of document IDs
            embeddings: List of embedding tensors
            metadata_list: Optional list of metadata dicts
        """
        if metadata_list is None:
            metadata_list = [{}] * len(doc_ids)
        
        for doc_id, embedding, metadata in zip(doc_ids, embeddings, metadata_list):
            self.add_document(doc_id, embedding, metadata)
    
    def get_document(self, doc_id: str) -> Tuple[torch.Tensor, dict]:
        """
        Retrieve a document's embedding and metadata.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            (embedding, metadata) tuple
        """
        if doc_id not in self.doc_embeddings:
            raise KeyError(f"Document not found: {doc_id}")
        
        return self.doc_embeddings[doc_id], self.doc_metadata[doc_id]
    
    def get_all_embeddings(self) -> List[torch.Tensor]:
        """Get all document embeddings as a list."""
        return list(self.doc_embeddings.values())
    
    def get_all_doc_ids(self) -> List[str]:
        """Get all document IDs."""
        return list(self.doc_embeddings.keys())
    
    def get_all_metadata(self) -> Dict[str, dict]:
        """Get all metadata."""
        return self.doc_metadata
    
    def remove_document(self, doc_id: str):
        """Remove a document from the store."""
        if doc_id in self.doc_embeddings:
            del self.doc_embeddings[doc_id]
        if doc_id in self.doc_metadata:
            del self.doc_metadata[doc_id]
    
    def search(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 10,
        scorer = None
    ) -> List[Tuple[str, float]]:
        """
        Search for most similar documents.
        
        Args:
            query_embedding: Query embedding tensor
            top_k: Number of top results to return
            scorer: Late interaction scorer instance
        
        Returns:
            List of (doc_id, score) tuples
        """
        if len(self.doc_embeddings) == 0:
            return []
        
        if scorer is None:
            # Use simple dot product if no scorer provided
            scores = {}
            for doc_id, doc_emb in self.doc_embeddings.items():
                # Simple aggregation: mean of dot products
                score = torch.matmul(
                    query_embedding.mean(dim=0),
                    doc_emb.mean(dim=0)
                ).item()
                scores[doc_id] = score
        else:
            # Use late interaction scorer
            scores = {}
            for doc_id, doc_emb in self.doc_embeddings.items():
                score = scorer.compute_maxsim(query_embedding, doc_emb)
                scores[doc_id] = score
        
        # Sort by score
        sorted_results = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results[:top_k]
    
    def save(self):
        """Save embeddings and metadata to disk."""
        # Save embeddings
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.doc_embeddings, f)
        
        # Save metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(self.doc_metadata, f, indent=2)
        
        print(f"Saved {len(self.doc_embeddings)} documents to {self.storage_dir}")
    
    def load(self):
        """Load embeddings and metadata from disk."""
        if self.embeddings_file.exists():
            with open(self.embeddings_file, 'rb') as f:
                self.doc_embeddings = pickle.load(f)
            print(f"Loaded {len(self.doc_embeddings)} document embeddings")
        
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.doc_metadata = json.load(f)
            print(f"Loaded metadata for {len(self.doc_metadata)} documents")
    
    def clear(self):
        """Clear all stored data."""
        self.doc_embeddings.clear()
        self.doc_metadata.clear()
    
    def get_statistics(self) -> dict:
        """Get statistics about stored embeddings."""
        if len(self.doc_embeddings) == 0:
            return {
                'num_documents': 0,
                'total_vectors': 0,
                'avg_vectors_per_doc': 0,
                'embedding_dim': 0
            }
        
        total_vectors = sum(emb.shape[0] for emb in self.doc_embeddings.values())
        embedding_dim = next(iter(self.doc_embeddings.values())).shape[1]
        
        return {
            'num_documents': len(self.doc_embeddings),
            'total_vectors': total_vectors,
            'avg_vectors_per_doc': total_vectors / len(self.doc_embeddings),
            'embedding_dim': embedding_dim,
            'storage_dir': str(self.storage_dir)
        }


class FAISSVectorStore(VectorStore):
    """
    Vector store with FAISS indexing for large-scale retrieval.
    """
    
    def __init__(self, storage_dir: Union[str, Path] = "vector_store", index_type: str = "IVF"):
        """
        Initialize FAISS-backed vector store.
        
        Args:
            storage_dir: Directory to store data
            index_type: 'Flat', 'IVF', or 'HNSW'
        """
        super().__init__(storage_dir)
        
        self.index_type = index_type
        self.faiss_available = False
        
        try:
            import faiss
            self.faiss = faiss
            self.faiss_available = True
        except ImportError:
            print("Warning: FAISS not available. Using standard vector store.")
            print("Install with: pip install faiss-cpu (or faiss-gpu)")
        
        self.index = None
        self.index_to_doc_id = []
    
    def build_index(self, embedding_dim: int, num_docs: int):
        """
        Build FAISS index for efficient search.
        
        Args:
            embedding_dim: Dimension of embeddings
            num_docs: Number of documents (for IVF)
        """
        if not self.faiss_available:
            return
        
        if self.index_type == "Flat":
            self.index = self.faiss.IndexFlatIP(embedding_dim)
        elif self.index_type == "IVF":
            nlist = min(100, num_docs // 10)
            quantizer = self.faiss.IndexFlatIP(embedding_dim)
            self.index = self.faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
        elif self.index_type == "HNSW":
            self.index = self.faiss.IndexHNSWFlat(embedding_dim, 32)
        else:
            self.index = self.faiss.IndexFlatIP(embedding_dim)
        
        print(f"Built FAISS {self.index_type} index for dimension {embedding_dim}")
    
    def add_to_index(self):
        """Add all embeddings to FAISS index (using mean pooling)."""
        if not self.faiss_available or len(self.doc_embeddings) == 0:
            return
        
        # Get mean embeddings for indexing
        mean_embeddings = []
        self.index_to_doc_id = []
        
        for doc_id, emb in self.doc_embeddings.items():
            mean_emb = emb.mean(dim=0).numpy()
            mean_embeddings.append(mean_emb)
            self.index_to_doc_id.append(doc_id)
        
        mean_embeddings = np.array(mean_embeddings).astype('float32')
        
        # Build and train index if needed
        if self.index is None:
            self.build_index(mean_embeddings.shape[1], len(mean_embeddings))
        
        if self.index is not None and self.index_type == "IVF":
            if not self.index.is_trained:
                self.index.train(len(mean_embeddings), mean_embeddings)
        
        if self.index is not None:
            self.index.add(len(mean_embeddings), mean_embeddings)
        print(f"Added {len(mean_embeddings)} documents to FAISS index")