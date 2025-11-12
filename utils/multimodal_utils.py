"""
utils/multimodal_utils.py - Multimodal Processing Utilities

Shared utilities for handling both text and visual document processing.
"""

from typing import List, Dict, Tuple, Union, Optional
from pathlib import Path
import torch


class DocumentTypeDetector:
    """
    Detect document type to route to appropriate processing pipeline.
    """
    
    @staticmethod
    def detect_type(content: Union[str, Path]) -> str:
        """
        Detect if content is text or image/visual.
        
        Args:
            content: File path or text string
        
        Returns:
            'text', 'image', 'pdf', or 'unknown'
        """
        if isinstance(content, str):
            # Check if it's a file path
            if Path(content).exists():
                path = Path(content)
                suffix = path.suffix.lower()
                
                if suffix in ['.txt', '.md', '.csv', '.json']:
                    return 'text'
                elif suffix in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
                    return 'image'
                elif suffix == '.pdf':
                    return 'pdf'
            else:
                # Assume it's text content
                return 'text'
        
        return 'unknown'
    
    @staticmethod
    def detect_batch_types(
        contents: List[Union[str, Path]]
    ) -> Dict[str, List[Union[str, Path]]]:
        """
        Categorize multiple items by type.
        
        Args:
            contents: List of paths or content strings
        
        Returns:
            Dictionary with categorized items
        """
        categorized = {'text': [], 'image': [], 'pdf': [], 'unknown': []}
        
        for content in contents:
            doc_type = DocumentTypeDetector.detect_type(content)
            categorized[doc_type].append(content)
        
        return categorized


class EmbeddingCombiner:
    """
    Combine embeddings from multiple modalities (text + visual).
    """
    
    def __init__(self, text_weight: float = 0.5, visual_weight: float = 0.5):
        """
        Initialize Embedding Combiner.
        
        Args:
            text_weight: Weight for text embeddings
            visual_weight: Weight for visual embeddings
        """
        self.text_weight = text_weight
        self.visual_weight = visual_weight
        
        # Validate weights
        if abs(self.text_weight + self.visual_weight - 1.0) > 1e-5:
            raise ValueError("Weights must sum to 1.0")
    
    def combine_scores(
        self,
        text_scores: torch.Tensor,
        visual_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine text and visual similarity scores.
        
        Args:
            text_scores: Text similarity scores
            visual_scores: Visual similarity scores
        
        Returns:
            Combined scores
        """
        return (
            self.text_weight * text_scores +
            self.visual_weight * visual_scores
        )
    
    def combine_rankings(
        self,
        text_results: List[Tuple[str, float]],
        visual_results: List[Tuple[str, float]],
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Combine rankings from text and visual search.
        
        Args:
            text_results: List of (doc_id, score) from text search
            visual_results: List of (doc_id, score) from visual search
            k: Number of results to return
        
        Returns:
            Combined top-k results
        """
        # Create score maps
        text_scores = {doc_id: score for doc_id, score in text_results}
        visual_scores = {doc_id: score for doc_id, score in visual_results}
        
        all_docs = set(text_scores.keys()) | set(visual_scores.keys())
        combined_scores = {}
        
        for doc_id in all_docs:
            text_score = text_scores.get(doc_id, 0.0)
            visual_score = visual_scores.get(doc_id, 0.0)
            combined_scores[doc_id] = self.combine_scores(
                torch.tensor(text_score),
                torch.tensor(visual_score)
            ).item()
        
        # Sort and return top-k
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results[:k]


class QueryExpander:
    """
    Expand queries for better retrieval across modalities.
    """
    
    @staticmethod
    def expand_text_query(query: str) -> List[str]:
        """
        Generate query variations for text search.
        
        Args:
            query: Original text query
        
        Returns:
            List of query variations
        """
        variations = [query]
        
        # Add variations
        variations.append(query.lower())
        variations.append(query.upper())
        
        # Add synonyms (placeholder - can integrate with thesaurus)
        if 'document' in query.lower():
            variations.append(query.replace('document', 'page'))
            variations.append(query.replace('document', 'file'))
        
        return variations
    
    @staticmethod
    def generate_visual_query_prompts(query: str) -> List[str]:
        """
        Generate visual prompts from text query for image search.
        
        Args:
            query: Text query
        
        Returns:
            List of visual search prompts
        """
        prompts = [
            f"Document containing: {query}",
            f"Page with: {query}",
            f"Visual content about: {query}",
            f"Text describing: {query}",
            query  # Keep original as well
        ]
        
        return prompts


class MetadataManager:
    """
    Manage metadata across text and visual documents.
    """
    
    def __init__(self):
        """Initialize metadata manager."""
        self.metadata: Dict[str, dict] = {}
    
    def add_metadata(self, doc_id: str, metadata: dict):
        """Add metadata for document."""
        self.metadata[doc_id] = metadata
    
    def get_metadata(self, doc_id: str) -> dict:
        """Get metadata for document."""
        return self.metadata.get(doc_id, {})
    
    def filter_by_metadata(
        self,
        doc_ids: List[str],
        filters: Dict[str, Union[str, List[str]]]
    ) -> List[str]:
        """
        Filter documents by metadata criteria.
        
        Args:
            doc_ids: Document IDs to filter
            filters: Metadata filters {key: value or [values]}
        
        Returns:
            Filtered document IDs
        """
        filtered = []
        
        for doc_id in doc_ids:
            meta = self.metadata.get(doc_id, {})
            match = True
            
            for key, value in filters.items():
                if isinstance(value, list):
                    if meta.get(key) not in value:
                        match = False
                        break
                else:
                    if meta.get(key) != value:
                        match = False
                        break
            
            if match:
                filtered.append(doc_id)
        
        return filtered
    
    def merge_results(
        self,
        text_results: List[Tuple[str, float]],
        visual_results: List[Tuple[str, float]],
        strategy: str = 'average'
    ) -> List[Tuple[str, float]]:
        """
        Merge results from multiple searches with metadata.
        
        Args:
            text_results: Results from text search
            visual_results: Results from visual search
            strategy: 'average', 'max', 'weighted'
        
        Returns:
            Merged results
        """
        scores = {}
        counts = {}
        
        # Process text results
        for doc_id, score in text_results:
            scores[doc_id] = scores.get(doc_id, 0) + score
            counts[doc_id] = counts.get(doc_id, 0) + 1
        
        # Process visual results
        for doc_id, score in visual_results:
            scores[doc_id] = scores.get(doc_id, 0) + score
            counts[doc_id] = counts.get(doc_id, 0) + 1
        
        # Combine based on strategy
        if strategy == 'average':
            final_scores = {
                doc_id: score / counts[doc_id]
                for doc_id, score in scores.items()
            }
        elif strategy == 'max':
            final_scores = scores
        else:  # weighted (prioritize documents found in both)
            final_scores = {
                doc_id: score * counts[doc_id] / 2
                for doc_id, score in scores.items()
            }
        
        # Sort and return
        return sorted(
            final_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )


class PerformanceMetrics:
    """
    Track performance metrics for quality attributes.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = {
            'indexing_time': [],
            'search_time': [],
            'memory_usage': [],
            'accuracy_scores': [],
            'throughput': []
        }
    
    def log_indexing_time(self, time_seconds: float, num_docs: int):
        """Log indexing performance."""
        self.metrics['indexing_time'].append({
            'time': time_seconds,
            'num_docs': num_docs,
            'docs_per_sec': num_docs / time_seconds if time_seconds > 0 else 0
        })
    
    def log_search_time(self, time_seconds: float):
        """Log search performance."""
        self.metrics['search_time'].append(time_seconds)
    
    def log_accuracy(self, metric_name: str, score: float):
        """Log accuracy metric (MRR, NDCG, P@K)."""
        self.metrics['accuracy_scores'].append({
            'metric': metric_name,
            'score': score
        })
    
    def get_summary(self) -> dict:
        """Get performance metrics summary."""
        summary = {}
        
        if self.metrics['indexing_time']:
            times = [t['time'] for t in self.metrics['indexing_time']]
            summary['avg_indexing_time'] = sum(times) / len(times)
            summary['total_indexing_time'] = sum(times)
        
        if self.metrics['search_time']:
            times = self.metrics['search_time']
            summary['avg_search_time'] = sum(times) / len(times)
            summary['min_search_time'] = min(times)
            summary['max_search_time'] = max(times)
        
        if self.metrics['accuracy_scores']:
            summary['accuracy_samples'] = len(self.metrics['accuracy_scores'])
        
        return summary