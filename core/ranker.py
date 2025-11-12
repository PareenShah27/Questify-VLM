"""
core/ranker.py - Result Ranking for Questify VLM

Supports:
- Text-based result ranking
- VLM-based result ranking
- Hybrid result ranking
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


class ResultRanker:
    """Rank text search results."""
    
    def __init__(self, config):
        """Initialize result ranker."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def rank_results(self,
                    results: List[Dict[str, Any]],
                    scores: np.ndarray,
                    top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Rank and filter results by score.
        
        Args:
            results: List of result dictionaries
            scores: Array of relevance scores
            top_k: Number of top results to return
            
        Returns:
            Top-k ranked results with scores
        """
        try:
            min_score = self.config.get('search.min_similarity_score', 0.01)
            
            # Combine results with scores
            ranked = []
            for i, (result, score) in enumerate(zip(results, scores)):
                if score >= min_score:
                    result_copy = result.copy()
                    result_copy['score'] = float(score)
                    result_copy['rank'] = len(ranked) + 1
                    ranked.append(result_copy)
            
            # Sort by score descending
            ranked.sort(key=lambda x: x['score'], reverse=True)
            
            # Return top-k
            return ranked[:top_k]
        except Exception as e:
            self.logger.error(f"Error ranking results: {e}")
            return []
    
    def rerank_results(self,
                      results: List[Dict[str, Any]],
                      query: str) -> List[Dict[str, Any]]:
        """
        Apply reranking logic (optional enhancement).
        
        Args:
            results: List of results to rerank
            query: Original query
            
        Returns:
            Reranked results
        """
        try:
            # Placeholder for reranking logic
            # Could use cross-encoder models, BM25, etc.
            return results
        except Exception as e:
            self.logger.error(f"Error reranking: {e}")
            return results


class VLMResultRanker:
    """Rank VLM search results."""
    
    def __init__(self, config):
        """Initialize VLM result ranker."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def rank_results(self,
                    results: List[Dict[str, Any]],
                    scores: np.ndarray,
                    top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Rank VLM results.
        
        Args:
            results: List of result dictionaries
            scores: Array of VLM similarity scores
            top_k: Number of results to return
            
        Returns:
            Top-k ranked VLM results
        """
        try:
            min_score = self.config.get('search.min_similarity_score', 0.01)
            
            # Normalize scores to [0, 1]
            if len(scores) > 0:
                scores = np.clip(scores, 0, 1)
            
            # Combine and filter
            ranked = []
            for i, (result, score) in enumerate(zip(results, scores)):
                if score >= min_score:
                    result_copy = result.copy()
                    result_copy['vlm_score'] = float(score)
                    result_copy['rank'] = len(ranked) + 1
                    ranked.append(result_copy)
            
            # Sort by score descending
            ranked.sort(key=lambda x: x['vlm_score'], reverse=True)
            
            # Apply diversity penalty if configured
            diversity_penalty = self.config.get('hybrid.diversity_penalty', 0.0)
            if diversity_penalty > 0:
                ranked = self.apply_diversity_penalty(ranked, diversity_penalty)
            
            return ranked[:top_k]
        except Exception as e:
            self.logger.error(f"Error ranking VLM results: {e}")
            return []
    
    def apply_diversity_penalty(self,
                               results: List[Dict[str, Any]],
                               penalty: float = 0.1) -> List[Dict[str, Any]]:
        """
        Penalize similar results to increase diversity.
        
        Args:
            results: List of results
            penalty: Penalty factor [0, 1]
            
        Returns:
            Results with adjusted scores
        """
        try:
            if len(results) <= 1:
                return results
            
            penalized = [results[0].copy()]
            
            # Simple diversity: penalize consecutive similar results
            for i in range(1, len(results)):
                result = results[i].copy()
                
                # Penalize based on similarity to already selected
                adjusted_score = result.get('vlm_score', 0)
                adjusted_score *= (1.0 - penalty)
                
                result['vlm_score'] = adjusted_score
                penalized.append(result)
            
            # Re-sort with penalty applied
            penalized.sort(key=lambda x: x['vlm_score'], reverse=True)
            
            return penalized
        except Exception as e:
            self.logger.error(f"Error applying diversity penalty: {e}")
            return results


class HybridResultRanker:
    """Rank hybrid search results."""
    
    def __init__(self, config):
        """Initialize hybrid result ranker."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def rank_results(self,
                    text_results: List[Dict[str, Any]],
                    vlm_results: List[Dict[str, Any]],
                    top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Merge and rank hybrid search results.
        
        Args:
            text_results: Results from text search
            vlm_results: Results from VLM search
            top_k: Number of results to return
            
        Returns:
            Top-k merged and ranked results
        """
        try:
            # Merge results
            merged = self.merge_results(text_results, vlm_results)
            
            # Calculate hybrid scores
            for result in merged:
                text_score = result.get('score', 0.0)
                vlm_score = result.get('vlm_score', 0.0)
                
                text_weight = self.config.get('hybrid.text_weight', 0.5)
                vlm_weight = self.config.get('hybrid.vlm_weight', 0.5)
                
                # Combine scores
                hybrid_score = text_weight * text_score + vlm_weight * vlm_score
                result['hybrid_score'] = hybrid_score
            
            # Sort by hybrid score
            merged.sort(key=lambda x: x['hybrid_score'], reverse=True)
            
            # Update ranks
            for i, result in enumerate(merged):
                result['rank'] = i + 1
            
            return merged[:top_k]
        except Exception as e:
            self.logger.error(f"Error ranking hybrid results: {e}")
            return []
    
    def merge_results(self,
                     text_results: List[Dict[str, Any]],
                     vlm_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge results from text and VLM searches.
        
        Args:
            text_results: Text search results
            vlm_results: VLM search results
            
        Returns:
            Merged results with duplicates handled
        """
        try:
            # Create ID-based lookup
            merged_dict = {}
            
            # Add text results
            for result in text_results:
                doc_id = result.get('doc_id', result.get('id'))
                if doc_id:
                    merged_dict[doc_id] = result.copy()
            
            # Merge VLM results
            for result in vlm_results:
                doc_id = result.get('doc_id', result.get('id'))
                if doc_id:
                    if doc_id in merged_dict:
                        # Merge with existing result
                        merged_dict[doc_id]['vlm_score'] = result.get('vlm_score', 0.0)
                    else:
                        # Add new result
                        merged_dict[doc_id] = result.copy()
            
            # Fill missing scores with 0
            for result in merged_dict.values():
                if 'score' not in result:
                    result['score'] = 0.0
                if 'vlm_score' not in result:
                    result['vlm_score'] = 0.0
            
            return list(merged_dict.values())
        except Exception as e:
            self.logger.error(f"Error merging results: {e}")
            return text_results + vlm_results