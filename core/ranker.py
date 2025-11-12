"""
Result ranking module for Questify search engine.
Handles sorting and formatting of search results.
"""

from typing import List, Dict, Tuple, Optional


class ResultRanker:
    """Ranks and formats search results based on similarity scores."""
    
    def __init__(self, max_results: int = 10, min_similarity_score: float = 0.01):
        """
        Initialize result ranker.
        
        Args:
            max_results: Maximum number of results to return
            min_similarity_score: Minimum similarity score threshold
        """
        self.max_results = max_results
        self.min_similarity_score = min_similarity_score
    
    def rank_results(self, similarities: List[Tuple[str, float]], 
                     document_store=None) -> Dict:
        """
        Rank and format search results.
        
        Args:
            similarities: List of (doc_id, similarity_score) tuples
            document_store: Optional document store for retrieving document content
            
        Returns:
            Dictionary with formatted search results
        """
        # Filter by minimum similarity score
        filtered_results = [
            (doc_id, score) for doc_id, score in similarities 
            if score >= self.min_similarity_score
        ]
        
        # Sort by similarity score (descending)
        sorted_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)
        
        # Limit number of results
        top_results = sorted_results[:self.max_results]
        
        # Format results
        formatted_results = []
        for doc_id, similarity_score in top_results:
            result = {
                'doc_id': doc_id,
                'similarity_score': round(similarity_score, 4),
                'rank': len(formatted_results) + 1
            }
            
            # Add document content if document store is available
            if document_store:
                content = document_store.get_document_content(doc_id)
                if content:
                    result['content'] = content
                    result['preview'] = self._create_preview(content)
            
            formatted_results.append(result)
        
        return {
            'results': formatted_results,
            'total_results': len(formatted_results),
            'total_candidates': len(similarities),
            'search_stats': {
                'filtered_by_threshold': len(similarities) - len(filtered_results),
                'returned_results': len(formatted_results)
            }
        }
    
    def _create_preview(self, content: str, max_length: int = 200) -> str:
        """
        Create a preview snippet from document content.
        
        Args:
            content: Full document content
            max_length: Maximum length of preview
            
        Returns:
            Preview snippet
        """
        if not content:
            return ""
        
        if len(content) <= max_length:
            return content
        
        # Try to cut at word boundary
        preview = content[:max_length]
        last_space = preview.rfind(' ')
        
        if last_space > max_length * 0.8:  # If we find a space reasonably close to the end
            preview = preview[:last_space]
        
        return preview + "..."
    
    def get_ranking_stats(self, similarities: List[Tuple[str, float]]) -> Dict:
        """
        Get statistics about the ranking process.
        
        Args:
            similarities: List of (doc_id, similarity_score) tuples
            
        Returns:
            Dictionary with ranking statistics
        """
        if not similarities:
            return {
                'total_candidates': 0,
                'avg_similarity': 0.0,
                'max_similarity': 0.0,
                'min_similarity': 0.0
            }
        
        scores = [score for _, score in similarities]
        
        return {
            'total_candidates': len(similarities),
            'avg_similarity': sum(scores) / len(scores),
            'max_similarity': max(scores),
            'min_similarity': min(scores),
            'above_threshold': len([s for s in scores if s >= self.min_similarity_score])
        }