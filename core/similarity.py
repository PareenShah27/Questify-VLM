"""
Cosine similarity calculation module for Questify search engine.
Implements efficient cosine similarity computation between queries and documents.
"""

import math
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F


class CosineSimilarityCalculator:
    """Calculates cosine similarity between query and document vectors."""
    
    def __init__(self):
        """Initialize similarity calculator."""
        pass
    
    def calculate_similarity(self, query_vector: Dict[str, float], 
                           document_vector: Dict[str, float],
                           document_norm: float) -> float:
        """
        Calculate cosine similarity between query and document vectors.
        
        Args:
            query_vector: Query TF-IDF vector
            document_vector: Document TF-IDF vector  
            document_norm: Precomputed L2 norm of document vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        if not query_vector or not document_vector or document_norm == 0:
            return 0.0
        
        # Calculate dot product
        dot_product = 0.0
        for term, query_score in query_vector.items():
            if term in document_vector:
                dot_product += query_score * document_vector[term]
        
        if dot_product == 0:
            return 0.0
        
        # Calculate query norm
        query_norm = math.sqrt(sum(score**2 for score in query_vector.values()))
        
        if query_norm == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = dot_product / (query_norm * document_norm)
        
        return similarity
    
    def batch_calculate_similarities(self, query_vector: Dict[str, float],
                                   document_vectors: Dict[str, Dict[str, float]],
                                   document_norms: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        Calculate similarities for multiple documents efficiently.
        
        Args:
            query_vector: Query TF-IDF vector
            document_vectors: Dictionary of document TF-IDF vectors
            document_norms: Dictionary of document L2 norms
            
        Returns:
            List of (doc_id, similarity_score) tuples
        """
        results = []
        
        for doc_id, doc_vector in document_vectors.items():
            doc_norm = document_norms.get(doc_id, 0.0)
            similarity = self.calculate_similarity(query_vector, doc_vector, doc_norm)
            if similarity > 0:  # Only include non-zero similarities
                results.append((doc_id, similarity))
        
        return results

class VLMSimilarityCalculator:
    """
    NEW: Calculate VLM-based similarity using late interaction (MaxSim).
    Works independently from CosineSimilarityCalculator.
    """
    
    def __init__(self, similarity_metric: str = "cosine"):
        """
        Initialize VLM Similarity Calculator.
        
        Args:
            similarity_metric: 'cosine' or 'dot_product'
        """
        self.similarity_metric = similarity_metric
    
    def calculate_maxsim(
        self,
        query_embedding: torch.Tensor,
        doc_embedding: torch.Tensor
    ) -> float:
        """
        Calculate MaxSim score (late interaction).
        
        Args:
            query_embedding: Query embedding (query_len, dim)
            doc_embedding: Document embedding (doc_len, dim)
        
        Returns:
            MaxSim similarity score
        """
        # Normalize if using cosine
        if self.similarity_metric == "cosine":
            query_embedding = F.normalize(query_embedding, p=2, dim=-1)
            doc_embedding = F.normalize(doc_embedding, p=2, dim=-1)
        
        # Compute similarity matrix: (query_len, doc_len)
        sim_matrix = torch.matmul(
            query_embedding,
            doc_embedding.transpose(0, 1)
        )
        
        # MaxSim: for each query token, find max similarity with doc tokens
        max_sim_per_token = sim_matrix.max(dim=1)[0]  # (query_len,)
        
        # Sum across query tokens
        score = max_sim_per_token.sum().item()
        
        return score
    
    def batch_calculate_similarities(
        self,
        query_embedding: torch.Tensor,
        doc_embeddings: Dict[str, torch.Tensor]
    ) -> List[Tuple[str, float]]:
        """
        Calculate VLM similarities for multiple documents.
        
        Args:
            query_embedding: Single query embedding tensor
            doc_embeddings: Dict of doc_id -> embedding tensor
        
        Returns:
            List of (doc_id, similarity_score) tuples
        """
        results = []
        
        for doc_id, doc_emb in doc_embeddings.items():
            try:
                score = self.calculate_maxsim(query_embedding, doc_emb)
                if score > 0:
                    results.append((doc_id, score))
            except Exception as e:
                print(f"Warning: Error calculating similarity for {doc_id}: {e}")
        
        return results


class HybridSimilarityCalculator:
    """
    NEW: Combines text and VLM similarities for hybrid search.
    Uses both original CosineSimilarityCalculator and new VLMSimilarityCalculator.
    """
    
    def __init__(
        self,
        text_weight: float = 0.5,
        vlm_weight: float = 0.5
    ):
        """
        Initialize Hybrid Similarity Calculator.
        
        Args:
            text_weight: Weight for text similarity scores
            vlm_weight: Weight for VLM similarity scores
        """
        if abs(text_weight + vlm_weight - 1.0) > 1e-5:
            raise ValueError("Weights must sum to 1.0")
        
        self.text_weight = text_weight
        self.vlm_weight = vlm_weight
        
        # Initialize both calculators
        self.text_calculator = CosineSimilarityCalculator()
        self.vlm_calculator = VLMSimilarityCalculator()
    
    def combine_similarities(
        self,
        text_similarities: List[Tuple[str, float]],
        vlm_similarities: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Combine text and VLM similarities into hybrid scores.
        
        Args:
            text_similarities: List of (doc_id, text_score) tuples
            vlm_similarities: List of (doc_id, vlm_score) tuples
        
        Returns:
            List of (doc_id, combined_score) tuples
        """
        # Create score maps
        text_scores = {doc_id: score for doc_id, score in text_similarities}
        vlm_scores = {doc_id: score for doc_id, score in vlm_similarities}
        
        # Get all document IDs from both sources
        all_doc_ids = set(text_scores.keys()) | set(vlm_scores.keys())
        
        # Combine scores
        combined_results = []
        for doc_id in all_doc_ids:
            text_score = text_scores.get(doc_id, 0.0)
            vlm_score = vlm_scores.get(doc_id, 0.0)
            
            # Weighted combination
            combined_score = (
                self.text_weight * text_score +
                self.vlm_weight * vlm_score
            )
            
            if combined_score > 0:
                combined_results.append((doc_id, combined_score))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        return combined_results
    
    def calculate_hybrid_similarity(
        self,
        query_text_vector: Dict[str, float],
        query_vlm_embedding: torch.Tensor,
        doc_id: str,
        doc_text_vector: Dict[str, float],
        doc_text_norm: float,
        doc_vlm_embedding: torch.Tensor
    ) -> float:
        """
        Calculate hybrid similarity for a single document.
        
        Args:
            query_text_vector: Text query TF-IDF vector
            query_vlm_embedding: VLM query embedding
            doc_id: Document ID
            doc_text_vector: Document TF-IDF vector
            doc_text_norm: Document vector norm
            doc_vlm_embedding: Document VLM embedding
        
        Returns:
            Combined similarity score
        """
        # Calculate text similarity
        text_score = self.text_calculator.calculate_similarity(
            query_text_vector,
            doc_text_vector,
            doc_text_norm
        )
        
        # Calculate VLM similarity
        vlm_score = self.vlm_calculator.calculate_maxsim(
            query_vlm_embedding,
            doc_vlm_embedding
        )
        
        # Combine
        hybrid_score = (
            self.text_weight * text_score +
            self.vlm_weight * vlm_score
        )
        
        return hybrid_score