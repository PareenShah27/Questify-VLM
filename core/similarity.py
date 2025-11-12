"""
core/similarity.py - Similarity Calculators for Questify VLM

Supports:
- Cosine similarity for text vectors (TF-IDF)
- Similarity for VLM embeddings
- Hybrid similarity combining both
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity


class CosineSimilarityCalculator:
    """Calculate cosine similarity for text vectors."""
    
    def __init__(self, config):
        """Initialize cosine similarity calculator."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @staticmethod
    def calculate(vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Similarity score [0, 1]
        """
        try:
            if vector1.size == 0 or vector2.size == 0:
                return 0.0
            
            # Reshape for cosine_similarity
            v1 = vector1.reshape(1, -1)
            v2 = vector2.reshape(1, -1)
            
            similarity = cosine_similarity(v1, v2)[0, 0]
            return max(0.0, float(similarity))
        except Exception:
            return 0.0
    
    @staticmethod
    def calculate_batch(query_vector: np.ndarray,
                       document_vectors: np.ndarray) -> np.ndarray:
        """
        Calculate similarity between query and multiple documents.
        
        Args:
            query_vector: Query vector (1D)
            document_vectors: Document vectors matrix (NxD)
            
        Returns:
            Array of similarity scores
        """
        try:
            if len(document_vectors) == 0:
                return np.array([])
            
            query_vec = query_vector.reshape(1, -1)
            similarities = cosine_similarity(query_vec, document_vectors)[0]
            
            return np.clip(similarities, 0, 1)
        except Exception:
            return np.array([])


class VLMSimilarityCalculator:
    """Calculate similarity for VLM embeddings."""
    
    def __init__(self, config):
        """Initialize VLM similarity calculator."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @staticmethod
    def calculate(query_embedding: np.ndarray,
                 doc_embedding: np.ndarray,
                 method: str = 'cosine') -> float:
        """
        Calculate VLM similarity between embeddings.
        
        Args:
            query_embedding: Query embedding vector
            doc_embedding: Document embedding vector
            method: 'cosine', 'l2', 'dot'
            
        Returns:
            Similarity score
        """
        try:
            if query_embedding.size == 0 or doc_embedding.size == 0:
                return 0.0
            
            if method == 'cosine':
                q = query_embedding.reshape(1, -1)
                d = doc_embedding.reshape(1, -1)
                return float(cosine_similarity(q, d)[0, 0])
            
            elif method == 'l2':
                distance = np.linalg.norm(query_embedding - doc_embedding)
                return float(1.0 / (1.0 + distance))
            
            elif method == 'dot':
                return float(np.dot(query_embedding, doc_embedding))
            
            else:
                return 0.0
        except Exception:
            return 0.0
    
    @staticmethod
    def calculate_batch(query_embedding: np.ndarray,
                       doc_embeddings: np.ndarray,
                       method: str = 'cosine') -> np.ndarray:
        """
        Batch VLM similarity calculation.
        
        Args:
            query_embedding: Query embedding (1D)
            doc_embeddings: Document embeddings (NxD)
            method: Similarity method
            
        Returns:
            Array of similarity scores
        """
        try:
            if len(doc_embeddings) == 0:
                return np.array([])
            
            if method == 'cosine':
                q = query_embedding.reshape(1, -1)
                return cosine_similarity(q, doc_embeddings)[0]
            
            elif method == 'l2':
                distances = np.linalg.norm(doc_embeddings - query_embedding, axis=1)
                return 1.0 / (1.0 + distances)
            
            elif method == 'dot':
                return np.dot(doc_embeddings, query_embedding)
            
            else:
                return np.zeros(len(doc_embeddings))
        except Exception:
            return np.array([])
    
    @staticmethod
    def late_interaction_score(query_tokens: List[np.ndarray],
                              doc_tokens: List[np.ndarray]) -> float:
        """
        Calculate late interaction score (ColPali approach).
        
        Args:
            query_tokens: List of query token embeddings
            doc_tokens: List of document token embeddings
            
        Returns:
            Late interaction score
        """
        try:
            if not query_tokens or not doc_tokens:
                return 0.0
            
            score = 0.0
            for q_token in query_tokens:
                # Max similarity to any document token
                similarities = cosine_similarity(
                    q_token.reshape(1, -1),
                    np.array(doc_tokens)
                )[0]
                score += max(similarities) if len(similarities) > 0 else 0.0
            
            # Normalize by query length
            return score / len(query_tokens) if query_tokens else 0.0
        except Exception:
            return 0.0


class HybridSimilarityCalculator:
    """Combine text and VLM similarity scores."""
    
    def __init__(self, config):
        """Initialize hybrid similarity calculator."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @staticmethod
    def combine_scores(text_score: float,
                      vlm_score: float,
                      text_weight: float = 0.5,
                      vlm_weight: float = 0.5,
                      method: str = 'rrf') -> float:
        """
        Combine text and VLM similarity scores.
        
        Args:
            text_score: Text similarity score [0, 1]
            vlm_score: VLM similarity score [0, 1]
            text_weight: Weight for text score
            vlm_weight: Weight for VLM score
            method: 'rrf', 'average', 'max', 'product'
            
        Returns:
            Combined score
        """
        try:
            # Normalize scores
            text_score = np.clip(text_score, 0, 1)
            vlm_score = np.clip(vlm_score, 0, 1)
            
            if method == 'average':
                return (text_weight * text_score + vlm_weight * vlm_score) / 2.0
            
            elif method == 'max':
                return max(text_weight * text_score, vlm_weight * vlm_score)
            
            elif method == 'product':
                return (text_weight * text_score) * (vlm_weight * vlm_score)
            
            elif method == 'rrf':  # Reciprocal Rank Fusion
                # Convert scores to ranks (1-100)
                text_rank = 1.0 / (1.0 + (100 * (1 - text_score)))
                vlm_rank = 1.0 / (1.0 + (100 * (1 - vlm_score)))
                return text_weight * text_rank + vlm_weight * vlm_rank
            
            else:
                return (text_score + vlm_score) / 2.0
        except Exception:
            return 0.0
    
    @staticmethod
    def combine_batch_scores(text_scores: np.ndarray,
                            vlm_scores: np.ndarray,
                            weights: Dict[str, float],
                            method: str = 'rrf') -> np.ndarray:
        """
        Batch combination of scores.
        
        Args:
            text_scores: Array of text scores
            vlm_scores: Array of VLM scores
            weights: Dict with 'text_weight' and 'vlm_weight'
            method: Combination method
            
        Returns:
            Array of combined scores
        """
        try:
            if len(text_scores) == 0:
                return np.array([])
            
            text_weight = weights.get('text_weight', 0.5)
            vlm_weight = weights.get('vlm_weight', 0.5)
            
            combined = np.zeros_like(text_scores, dtype=float)
            
            for i in range(len(text_scores)):
                combined[i] = HybridSimilarityCalculator.combine_scores(
                    text_scores[i],
                    vlm_scores[i] if i < len(vlm_scores) else 0.0,
                    text_weight,
                    vlm_weight,
                    method
                )
            
            return combined
        except Exception:
            return np.zeros_like(text_scores, dtype=float)