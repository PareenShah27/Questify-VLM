"""
Text preprocessing module for Questify search engine.
Handles tokenization, normalization, stopword removal, and text cleaning.
"""

import re
import string
from typing import List, Set


class TextPreprocessor:
    """Handles text preprocessing operations for search engine."""
    
    # Common English stopwords
    STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
        'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if',
        'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her',
        'would', 'make', 'like', 'into', 'him', 'two', 'more', 'very',
        'what', 'know', 'just', 'first', 'get', 'over', 'think', 'also',
        'your', 'work', 'life', 'only', 'can', 'still', 'should', 'after',
        'being', 'now', 'made', 'before', 'here', 'through', 'when', 'where'
    }
    
    def __init__(self, remove_stopwords: bool = True, min_token_length: int = 3):
        """
        Initialize text preprocessor.
        
        Args:
            remove_stopwords: Whether to remove stopwords
            min_token_length: Minimum length for tokens to be kept
        """
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length
    
    def preprocess(self, text: str) -> List[str]:
        """
        Preprocess text by tokenizing, normalizing, and filtering.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            List of preprocessed tokens
        """
        if not text:
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^a-zA-Z0-9\\s]', ' ', text)
        
        # Tokenize by splitting on whitespace
        tokens = text.split()
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            # Skip short tokens
            if len(token) < self.min_token_length:
                continue
                
            # Skip stopwords if enabled
            if self.remove_stopwords and token in self.STOPWORDS:
                continue
                
            filtered_tokens.append(token)
        
        return filtered_tokens
