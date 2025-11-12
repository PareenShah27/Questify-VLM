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

class VLMResultRanker:
    """
    NEW: Ranks VLM search results.
    Works independently from original ResultRanker.
    """
    
    def __init__(self, max_results: int = 10, min_similarity_score: float = 0.0):
        """
        Initialize VLM Result Ranker.
        
        Args:
            max_results: Maximum number of results
            min_similarity_score: Minimum score threshold (VLM scores are typically higher)
        """
        self.max_results = max_results
        self.min_similarity_score = min_similarity_score
    
    def rank_results(
        self,
        similarities: List[Tuple[str, float]],
        image_store=None
    ) -> Dict:
        """
        Rank VLM search results.
        
        Args:
            similarities: List of (doc_id, vlm_score) tuples
            image_store: Optional image store for metadata
        
        Returns:
            Formatted results dictionary
        """
        # Filter by threshold
        filtered = [
            (doc_id, score) for doc_id, score in similarities
            if score >= self.min_similarity_score
        ]
        
        # Sort by score (descending)
        sorted_results = sorted(filtered, key=lambda x: x[1], reverse=True)
        
        # Limit results
        top_results = sorted_results[:self.max_results]
        
        # Format results
        formatted_results = []
        for doc_id, score in top_results:
            result = {
                'doc_id': doc_id,
                'vlm_score': round(score, 4),
                'rank': len(formatted_results) + 1,
                'result_type': 'vlm'
            }
            
            # Add metadata if image store available
            if image_store:
                try:
                    _, metadata = image_store.get_document(doc_id)
                    result['image_path'] = metadata.get('path')
                    result['image_format'] = metadata.get('image_format')
                except Exception as e:
                    print(f"Warning: Could not get metadata for {doc_id}: {e}")
            
            formatted_results.append(result)
        
        return {
            'results': formatted_results,
            'total_results': len(formatted_results),
            'total_candidates': len(similarities),
            'search_mode': 'vlm',
            'search_stats': {
                'filtered_by_threshold': len(similarities) - len(filtered),
                'returned_results': len(formatted_results)
            }
        }
    
    def get_ranking_stats(self, similarities: List[Tuple[str, float]]) -> Dict:
        """Get VLM ranking statistics."""
        if not similarities:
            return {
                'total_candidates': 0,
                'avg_score': 0.0,
                'max_score': 0.0,
                'min_score': 0.0
            }
        
        scores = [score for _, score in similarities]
        
        return {
            'total_candidates': len(similarities),
            'avg_score': sum(scores) / len(scores),
            'max_score': max(scores),
            'min_score': min(scores),
            'above_threshold': len([s for s in scores if s >= self.min_similarity_score])
        }


class HybridResultRanker:
    """
    NEW: Ranks results combining both text and VLM scores.
    Uses both original ResultRanker and new VLMResultRanker.
    """
    
    def __init__(
        self,
        max_results: int = 10,
        text_weight: float = 0.5,
        vlm_weight: float = 0.5,
        text_threshold: float = 0.01,
        vlm_threshold: float = 0.0
    ):
        """
        Initialize Hybrid Result Ranker.
        
        Args:
            max_results: Maximum results to return
            text_weight: Weight for text scores
            vlm_weight: Weight for VLM scores
            text_threshold: Text score threshold
            vlm_threshold: VLM score threshold
        """
        self.max_results = max_results
        self.text_weight = text_weight
        self.vlm_weight = vlm_weight
        
        # Initialize both rankers
        self.text_ranker = ResultRanker(max_results, text_threshold)
        self.vlm_ranker = VLMResultRanker(max_results, vlm_threshold)
    
    def rank_hybrid_results(
        self,
        text_similarities: List[Tuple[str, float]],
        vlm_similarities: List[Tuple[str, float]],
        document_store=None,
        image_store=None
    ) -> Dict:
        """
        Rank results using hybrid scores.
        
        Args:
            text_similarities: Text similarity scores
            vlm_similarities: VLM similarity scores
            document_store: Optional text document store
            image_store: Optional image store
        
        Returns:
            Combined ranked results
        """
        # Create score maps
        text_scores = {doc_id: score for doc_id, score in text_similarities}
        vlm_scores = {doc_id: score for doc_id, score in vlm_similarities}
        
        # Get all document IDs
        all_doc_ids = set(text_scores.keys()) | set(vlm_scores.keys())
        
        # Calculate hybrid scores
        hybrid_scores = []
        for doc_id in all_doc_ids:
            text_score = text_scores.get(doc_id, 0.0)
            vlm_score = vlm_scores.get(doc_id, 0.0)
            
            # Weighted combination
            combined_score = (
                self.text_weight * text_score +
                self.vlm_weight * vlm_score
            )
            
            if combined_score > 0:
                hybrid_scores.append((doc_id, combined_score, text_score, vlm_score))
        
        # Sort by combined score
        sorted_results = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
        
        # Limit results
        top_results = sorted_results[:self.max_results]
        
        # Format results
        formatted_results = []
        for doc_id, combined_score, text_score, vlm_score in top_results:
            result = {
                'doc_id': doc_id,
                'hybrid_score': round(combined_score, 4),
                'text_score': round(text_score, 4),
                'vlm_score': round(vlm_score, 4),
                'rank': len(formatted_results) + 1,
                'result_type': 'hybrid'
            }
            
            # Add content/metadata if available
            if document_store:
                try:
                    content = document_store.get_document_content(doc_id)
                    if content:
                        result['content_preview'] = content[:200] + "..."
                except:
                    pass
            
            if image_store:
                try:
                    _, metadata = image_store.get_document(doc_id)
                    result['image_path'] = metadata.get('path')
                except:
                    pass
            
            formatted_results.append(result)
        
        return {
            'results': formatted_results,
            'total_results': len(formatted_results),
            'search_mode': 'hybrid',
            'weights': {
                'text_weight': self.text_weight,
                'vlm_weight': self.vlm_weight
            },
            'search_stats': {
                'text_candidates': len(text_similarities),
                'vlm_candidates': len(vlm_similarities),
                'total_candidates': len(all_doc_ids),
                'returned_results': len(formatted_results)
            }
        }
    
    def rank_ensemble(
        self,
        text_results: Dict,
        vlm_results: Dict,
        method: str = 'rrf'
    ) -> Dict:
        """
        Ensemble results from separate text and VLM rankings.
        
        Args:
            text_results: Results from text ranker
            vlm_results: Results from VLM ranker
            method: 'rrf' (reciprocal rank fusion) or 'weighted'
        
        Returns:
            Ensembled results
        """
        # Extract results
        text_docs = {
            r['doc_id']: r['rank']
            for r in text_results.get('results', [])
        }
        vlm_docs = {
            r['doc_id']: r['rank']
            for r in vlm_results.get('results', [])
        }
        
        all_docs = set(text_docs.keys()) | set(vlm_docs.keys())
        
        # Calculate ensemble scores
        ensemble_scores = []
        for doc_id in all_docs:
            if method == 'rrf':
                # Reciprocal Rank Fusion
                text_rank = text_docs.get(doc_id, 1000)
                vlm_rank = vlm_docs.get(doc_id, 1000)
                score = 1/(60 + text_rank) + 1/(60 + vlm_rank)
            else:  # weighted
                text_rank = text_docs.get(doc_id, 1000)
                vlm_rank = vlm_docs.get(doc_id, 1000)
                score = self.text_weight * (1/text_rank) + self.vlm_weight * (1/vlm_rank)
            
            ensemble_scores.append((doc_id, score))
        
        # Sort and format
        sorted_scores = sorted(ensemble_scores, key=lambda x: x[1], reverse=True)
        top_results = sorted_scores[:self.max_results]
        
        formatted = []
        for doc_id, score in top_results:
            formatted.append({
                'doc_id': doc_id,
                'ensemble_score': round(score, 4),
                'rank': len(formatted) + 1,
                'in_text_results': doc_id in text_docs,
                'in_vlm_results': doc_id in vlm_docs
            })
        
        return {
            'results': formatted,
            'total_results': len(formatted),
            'search_mode': f'ensemble_{method}',
            'method': method
        }