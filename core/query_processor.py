"""
Query processing module for Questify search engine.
Handles user input parsing, validation, and query preprocessing.
"""

import re
from typing import List, Optional
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