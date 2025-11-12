"""
core/query_processor.py - Query Processing for Questify VLM

Supports:
- Text query processing for TF-IDF search
- VLM query processing for visual search
- Multimodal query routing
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import re


class QueryProcessor:
    """Process text queries for TF-IDF search."""
    
    def __init__(self, config, text_preprocessor):
        """Initialize query processor."""
        self.config = config
        self.text_preprocessor = text_preprocessor
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def process(self, query: str) -> str:
        """
        Process and normalize a query.
        
        Args:
            query: Raw query text
            
        Returns:
            Processed query string
        """
        try:
            # Basic preprocessing
            processed = query.strip()
            
            # Lowercase if configured
            if self.config.get('text_preprocessing.lowercase', True):
                processed = processed.lower()
            
            # Remove extra whitespace
            processed = re.sub(r'\s+', ' ', processed)
            
            # Use text preprocessor if available
            if self.text_preprocessor:
                processed = self.text_preprocessor.preprocess(processed)
            
            return processed
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return query.lower().strip()
    
    def process_batch(self, queries: List[str]) -> List[str]:
        """
        Process multiple queries.
        
        Args:
            queries: List of query strings
            
        Returns:
            List of processed queries
        """
        return [self.process(q) for q in queries]


class VLMQueryProcessor:
    """Process queries for VLM search."""
    
    def __init__(self, config, vlm_embedder):
        """Initialize VLM query processor."""
        self.config = config
        self.vlm_embedder = vlm_embedder
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def encode_query(self, query: str) -> Optional[np.ndarray]:
        """
        Encode query text to VLM embedding.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding vector or None
        """
        try:
            embedding = self.vlm_embedder.encode_text(query)
            return embedding
        except Exception as e:
            self.logger.error(f"Error encoding query: {e}")
            return None
    
    def encode_batch(self, queries: List[str]) -> Optional[np.ndarray]:
        """
        Encode multiple queries.
        
        Args:
            queries: List of query texts
            
        Returns:
            Embedding matrix (NxD) or None
        """
        try:
            embeddings = []
            for query in queries:
                emb = self.encode_query(query)
                if emb is not None:
                    embeddings.append(emb)
            
            if embeddings:
                return np.array(embeddings)
            return None
        except Exception as e:
            self.logger.error(f"Error encoding batch: {e}")
            return None
    
    def encode_image(self, image_path: str) -> Optional[np.ndarray]:
        """Encode image to VLM embedding."""
        try:
            return self.vlm_embedder.encode_image(image_path)
        except Exception as e:
            self.logger.error(f"Error encoding image: {e}")
            return None


class MultimodalQueryRouter:
    """Route query to appropriate search mode."""
    
    def __init__(self, config):
        """Initialize query router."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def determine_search_mode(self, query: str) -> str:
        """
        Determine appropriate search mode for query.
        
        Args:
            query: Query text
            
        Returns:
            Search mode: 'text', 'vlm', or 'hybrid'
        """
        mode = self.config.get('search.search_mode', 'auto')
        
        if mode != 'auto':
            return mode
        
        # Auto-detect based on query characteristics
        if self._is_visual_query(query):
            return 'vlm'
        else:
            return 'text'
    
    def should_use_vlm(self, query: str) -> bool:
        """Check if VLM search is appropriate."""
        mode = self.determine_search_mode(query)
        return mode in ['vlm', 'hybrid']
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """
        Route query and prepare for processing.
        
        Args:
            query: Query text
            
        Returns:
            Routing information dict
        """
        mode = self.determine_search_mode(query)
        
        return {
            'query': query,
            'search_mode': mode,
            'use_text': mode in ['text', 'hybrid'],
            'use_vlm': mode in ['vlm', 'hybrid'],
            'confidence': self._get_mode_confidence(query),
        }
    
    def _is_visual_query(self, query: str) -> bool:
        """
        Heuristic to detect visual queries.
        
        Visual indicators:
        - Contains image-related keywords
        - Contains visual descriptors
        """
        visual_keywords = [
            'image', 'picture', 'photo', 'visual', 'layout',
            'diagram', 'chart', 'graph', 'screenshot', 'design',
            'color', 'shape', 'look', 'appearance', 'show me'
        ]
        
        query_lower = query.lower()
        for keyword in visual_keywords:
            if keyword in query_lower:
                return True
        
        return False
    
    def _get_mode_confidence(self, query: str) -> float:
        """Get confidence score for selected mode."""
        if self._is_visual_query(query):
            return 0.8
        return 0.7