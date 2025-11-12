"""
Cosine similarity calculation module for Questify search engine.
Implements efficient cosine similarity computation between queries and documents.
"""

import math
from typing import Dict, List, Tuple


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