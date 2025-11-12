"""
Query processing module for Questify search engine.
Handles user input parsing, validation, and query preprocessing.
"""

import re
from typing import List, Optional, Dict, Union
from pathlib import Path
import torch
from utils.text_preprocessor import TextPreprocessor


class QueryProcessor:
    """Processes and validates user search queries."""
    
    def __init__(self, preprocessor: TextPreprocessor):
        """
        Initialize query processor.
        
        Args:
            preprocessor: Text preprocessor instance
        """
        self.preprocessor = preprocessor
    
    def process_query(self, raw_query: str) -> List[str]:
        """
        Process raw query string into clean tokens.
        
        Args:
            raw_query: Raw user input query
            
        Returns:
            List of processed query terms
        """
        if not raw_query or not raw_query.strip():
            return []
        
        # Basic input validation and cleaning
        cleaned_query = self._clean_query(raw_query)
        
        # Preprocess using text preprocessor
        query_terms = self.preprocessor.preprocess(cleaned_query)
        
        return query_terms
    
    def _clean_query(self, query: str) -> str:
        """
        Clean raw query by removing excessive whitespace and invalid characters.
        
        Args:
            query: Raw query string
            
        Returns:
            Cleaned query string
        """
        # Remove any potentially harmful characters (basic sanitization)
        # Keep alphanumeric, spaces, and basic punctuation
        cleaned = re.sub(r'[^a-zA-Z0-9\\s\\-_.,!?]', ' ', query)
        
        return cleaned
    
    def validate_query(self, query_terms: List[str]) -> bool:
        """
        Validate processed query terms.
        
        Args:
            query_terms: List of processed query terms
            
        Returns:
            True if query is valid, False otherwise
        """
        # Check if we have at least one valid term
        if not query_terms:
            return False
        
        # Check if terms are not empty after processing
        valid_terms = [term for term in query_terms if term.strip()]
        
        return len(valid_terms) > 0
    
    def get_query_info(self, query_terms: List[str]) -> dict:
        """
        Get information about the processed query.
        
        Args:
            query_terms: List of processed query terms
            
        Returns:
            Dictionary with query statistics
        """
        return {
            'term_count': len(query_terms),
            'unique_terms': len(set(query_terms)),
            'terms': query_terms,
            'is_valid': self.validate_query(query_terms)
        }

class VLMQueryProcessor:
    """
    NEW: Process queries for VLM-based search.
    Works independently from original QueryProcessor.
    """
    
    def __init__(self, vlm_embedder=None):
        """
        Initialize VLM Query Processor.
        
        Args:
            vlm_embedder: VLMEmbedder instance (can be set later)
        """
        self.vlm_embedder = vlm_embedder
    
    def set_vlm_embedder(self, vlm_embedder):
        """
        Set or update VLM embedder.
        
        Args:
            vlm_embedder: VLMEmbedder instance
        """
        self.vlm_embedder = vlm_embedder
    
    def process_query(self, raw_query: str) -> Dict:
        """
        Process query for VLM search (minimal preprocessing).
        
        Args:
            raw_query: Raw query string
        
        Returns:
            Processed query dict with metadata
        """
        # Minimal preprocessing - preserve semantic meaning
        cleaned_query = raw_query.strip()
        cleaned_query = ' '.join(cleaned_query.split())  # Remove extra whitespace
        
        result = {
            'original_query': raw_query,
            'processed_query': cleaned_query,
            'is_valid': len(cleaned_query) > 0,
            'query_length': len(cleaned_query),
            'embedding': None
        }
        
        return result
    
    def embed_query(self, query: str) -> Optional[torch.Tensor]:
        """
        Generate VLM embedding for query.
        
        Args:
            query: Query string
        
        Returns:
            Query embedding tensor or None if no embedding was produced
        
        Raises:
            RuntimeError: If VLM embedder not set
        """
        if self.vlm_embedder is None:
            raise RuntimeError(
                "VLM embedder not initialized. "
                "Call set_vlm_embedder() first or pass embedder in __init__"
            )
        
        # Process query
        processed = self.process_query(query)
        
        if not processed['is_valid']:
            raise ValueError("Invalid query: empty or whitespace only")
        
        # Generate embedding
        embeddings = self.vlm_embedder.embed_queries([processed['processed_query']])
        
        return embeddings[0] if len(embeddings) > 0 else None
    
    def validate_query(self, query: str) -> bool:
        """
        Validate VLM query.
        
        Args:
            query: Query string
        
        Returns:
            True if valid, False otherwise
        """
        if not query or not query.strip():
            return False
        
        if len(query) > 2000:  # Max length for VLM
            return False
        
        return True
    
    def expand_query(self, query: str) -> List[str]:
        """
        Generate query variations for better VLM retrieval.
        
        Args:
            query: Original query
        
        Returns:
            List of query variations
        """
        variations = [query]
        
        # Add contextual variations
        variations.append(f"Document containing: {query}")
        variations.append(f"Page with: {query}")
        variations.append(f"Visual content about: {query}")
        
        return variations
    
    def get_query_info(self, query: str) -> Dict:
        """
        Get information about VLM query.
        
        Args:
            query: Query string
        
        Returns:
            Query information dictionary
        """
        processed = self.process_query(query)
        
        return {
            'original': query,
            'processed': processed['processed_query'],
            'is_valid': processed['is_valid'],
            'length': processed['query_length'],
            'variations_available': len(self.expand_query(query)),
            'embedder_ready': self.vlm_embedder is not None
        }


class MultimodalQueryRouter:
    """
    NEW: Routes queries to appropriate processor (text or VLM).
    Detects query type and delegates to correct processor.
    """
    
    def __init__(
        self,
        text_preprocessor: TextPreprocessor,
        vlm_embedder=None
    ):
        """
        Initialize Multimodal Query Router.
        
        Args:
            text_preprocessor: TextPreprocessor for text queries
            vlm_embedder: Optional VLMEmbedder for VLM queries
        """
        self.text_processor = QueryProcessor(text_preprocessor)
        self.vlm_processor = VLMQueryProcessor(vlm_embedder)
    
    def detect_query_type(self, query: Union[str, Path]) -> str:
        """
        Detect if query is text or image path.
        
        Args:
            query: Query (string or Path)
        
        Returns:
            'text' or 'image'
        """
        if isinstance(query, (Path, str)):
            query_str = str(query)
            
            # Check if it's a file path
            if Path(query_str).exists():
                suffix = Path(query_str).suffix.lower()
                if suffix in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.pdf']:
                    return 'image'
            
            # Otherwise treat as text
            return 'text'
        
        return 'text'
    
    def process_query(
        self,
        query: Union[str, Path],
        mode: str = 'auto'
    ) -> Dict:
        """
        Process query using appropriate processor.
        
        Args:
            query: Query string or image path
            mode: 'text', 'vlm', or 'auto' (auto-detect)
        
        Returns:
            Processed query dictionary
        """
        # Determine query type
        if mode == 'auto':
            query_type = self.detect_query_type(query)
        else:
            query_type = mode if mode in ['text', 'vlm'] else 'text'
        
        result: Dict[str, Union[str, List[str], bool]] = {
            'query': str(query),
            'detected_type': query_type,
            'mode': mode
        }
        
        if query_type == 'text':
            # Use original text processor
            terms = self.text_processor.process_query(str(query))
            result['processed_terms'] = terms
            result['is_valid'] = self.text_processor.validate_query(terms)
            result['processor'] = 'text'
        
        else:  # vlm or image
            # Use VLM processor
            vlm_result = self.vlm_processor.process_query(str(query))
            result['processed_query'] = vlm_result['processed_query']
            result['is_valid'] = vlm_result['is_valid']
            result['processor'] = 'vlm'
        
        return result