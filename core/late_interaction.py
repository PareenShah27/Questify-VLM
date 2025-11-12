"""
core/late_interaction.py - Late Interaction Scoring (ColPali)

Implements late interaction scoring for improved VLM-based retrieval.
Based on ColPali and related approaches for document retrieval.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class LateInteractionScorer:
    """Calculate late interaction scores for VLM-based retrieval."""
    
    def __init__(self, config):
        """Initialize late interaction scorer."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def score_pair(self,
                  query_tokens: np.ndarray,
                  doc_tokens: np.ndarray,
                  method: str = 'maxsim') -> float:
        """
        Calculate late interaction score between query and document.
        
        Args:
            query_tokens: Query token embeddings (NxD)
            doc_tokens: Document token embeddings (MxD)
            method: Scoring method ('maxsim', 'sum', 'attention')
            
        Returns:
            Late interaction score
        """
        try:
            if query_tokens.size == 0 or doc_tokens.size == 0:
                return 0.0
            
            if method == 'maxsim':
                return self._maxsim_score(query_tokens, doc_tokens)
            elif method == 'sum':
                return self._sum_score(query_tokens, doc_tokens)
            elif method == 'attention':
                return self._attention_score(query_tokens, doc_tokens)
            else:
                return self._maxsim_score(query_tokens, doc_tokens)
        except Exception as e:
            self.logger.error(f"Error calculating late interaction score: {e}")
            return 0.0
    
    def score_batch(self,
                   query_tokens: np.ndarray,
                   doc_tokens_list: List[np.ndarray],
                   method: str = 'maxsim') -> np.ndarray:
        """
        Calculate late interaction scores for batch of documents.
        
        Args:
            query_tokens: Query token embeddings (NxD)
            doc_tokens_list: List of document token embeddings
            method: Scoring method
            
        Returns:
            Array of scores
        """
        try:
            scores = []
            for doc_tokens in doc_tokens_list:
                score = self.score_pair(query_tokens, doc_tokens, method)
                scores.append(score)
            return np.array(scores)
        except Exception as e:
            self.logger.error(f"Error in batch scoring: {e}")
            return np.array([])
    
    @staticmethod
    def _maxsim_score(query_tokens: np.ndarray,
                     doc_tokens: np.ndarray) -> float:
        """
        MaxSim scoring: for each query token, find max similarity to any doc token.
        
        This is the core approach from ColPali.
        """
        try:
            # Compute similarity matrix (NxM)
            similarities = np.dot(query_tokens, doc_tokens.T)  # (N, M)
            
            # For each query token, get max similarity
            max_sims = np.max(similarities, axis=1)  # (N,)
            
            # Average across query tokens
            score = np.mean(max_sims)
            return float(score)
        except Exception:
            return 0.0
    
    @staticmethod
    def _sum_score(query_tokens: np.ndarray,
                  doc_tokens: np.ndarray) -> float:
        """Sum of maximum similarities (variant of MaxSim)."""
        try:
            similarities = np.dot(query_tokens, doc_tokens.T)
            max_sims = np.max(similarities, axis=1)
            return float(np.sum(max_sims))
        except Exception:
            return 0.0
    
    @staticmethod
    def _attention_score(query_tokens: np.ndarray,
                        doc_tokens: np.ndarray) -> float:
        """
        Attention-based scoring with softmax weighting.
        """
        try:
            # Compute similarity matrix
            similarities = np.dot(query_tokens, doc_tokens.T)  # (N, M)
            
            # Apply softmax along document dimension
            exp_sims = np.exp(similarities - np.max(similarities, axis=1, keepdims=True))
            softmax_sims = exp_sims / np.sum(exp_sims, axis=1, keepdims=True)
            
            # Weighted score
            scores = np.max(softmax_sims, axis=1)
            return float(np.mean(scores))
        except Exception:
            return 0.0
    
    def extract_query_tokens(self,
                            query_embedding: np.ndarray,
                            method: str = 'kmeans') -> Optional[np.ndarray]:
        """
        Extract token embeddings from query.
        
        Args:
            query_embedding: Query embedding vector (D,)
            method: Extraction method
            
        Returns:
            Token embeddings (NxD)
        """
        try:
            # Simplified: return as single token
            # In actual implementation, would use more sophisticated tokenization
            return query_embedding.reshape(1, -1)
        except Exception as e:
            self.logger.error(f"Error extracting query tokens: {e}")
            return None
    
    def extract_document_tokens(self,
                               doc_embedding: np.ndarray,
                               num_tokens: int = 10,
                               method: str = 'stride') -> Optional[np.ndarray]:
        """
        Extract token embeddings from document.
        
        Args:
            doc_embedding: Document embedding (D,) or chunked
            num_tokens: Number of tokens to extract
            method: Extraction method ('stride', 'kmeans', 'random')
            
        Returns:
            Token embeddings (NxD)
        """
        try:
            if len(doc_embedding.shape) == 1:
                # Single vector - return as single token
                return doc_embedding.reshape(1, -1)
            
            elif len(doc_embedding.shape) == 2:
                # Multiple token embeddings already
                if doc_embedding.shape[0] <= num_tokens:
                    return doc_embedding
                
                if method == 'stride':
                    stride = doc_embedding.shape[0] // num_tokens
                    indices = range(0, doc_embedding.shape[0], stride)[:num_tokens]
                    return doc_embedding[list(indices)]
                
                elif method == 'kmeans':
                    # Simplified: sample evenly
                    indices = np.linspace(0, doc_embedding.shape[0]-1, num_tokens, dtype=int)
                    return doc_embedding[indices]
                
                else:  # random
                    indices = np.random.choice(doc_embedding.shape[0], num_tokens, replace=False)
                    return doc_embedding[indices]
            
            return None
        except Exception as e:
            self.logger.error(f"Error extracting document tokens: {e}")
            return None
    
    def normalize_score(self,
                       score: float,
                       min_val: float = 0.0,
                       max_val: float = 1.0) -> float:
        """Normalize score to range [min_val, max_val]."""
        try:
            # Clip to reasonable range first
            clipped = np.clip(score, -1, 2)
            # Normalize to [0, 1]
            normalized = (clipped + 1) / 3
            # Scale to desired range
            scaled = min_val + normalized * (max_val - min_val)
            return float(scaled)
        except Exception:
            return min_val


class DocumentTokenizer:
    """Tokenize documents for late interaction scoring."""
    
    def __init__(self, config):
        """Initialize document tokenizer."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.chunk_size = config.get('vlm.chunk_size', 512)
        self.stride = config.get('vlm.stride', 256)
    
    def tokenize_document(self,
                         doc_text: str,
                         embedding_func=None) -> Optional[np.ndarray]:
        """
        Tokenize document into chunks and generate embeddings.
        
        Args:
            doc_text: Document text
            embedding_func: Function to generate embeddings for each chunk
            
        Returns:
            Token embeddings (NxD)
        """
        try:
            # Split into chunks
            chunks = self._chunk_text(doc_text)
            
            if not chunks or embedding_func is None:
                return None
            
            # Generate embeddings for each chunk
            embeddings = []
            for chunk in chunks:
                emb = embedding_func(chunk)
                if emb is not None:
                    embeddings.append(emb)
            
            if embeddings:
                return np.array(embeddings)
            return None
        except Exception as e:
            self.logger.error(f"Error tokenizing document: {e}")
            return None
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        try:
            words = text.split()
            chunks = []
            
            for i in range(0, len(words), self.stride):
                chunk = ' '.join(words[i:i + self.chunk_size])
                if chunk.strip():
                    chunks.append(chunk)
            
            return chunks
        except Exception:
            return [text] if text else []