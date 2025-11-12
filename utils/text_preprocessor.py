"""
utils/text_preprocessor.py - Text Preprocessing Utilities

Handles text normalization, tokenization, and cleaning.
Used by QueryProcessor from core/query_processor.py
"""

import logging
import re
from typing import List, Dict, Optional, Set, Any
from pathlib import Path


class TextPreprocessor:
    """Preprocess and normalize text for search."""
    
    def __init__(self, config):
        """Initialize text preprocessor."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load stopwords if configured
        self.stopwords = set()
        if config.get('text_preprocessing.remove_stopwords', True):
            self._load_stopwords()
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess text with all configured steps.
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        try:
            # Lowercase
            if self.config.get('text_preprocessing.lowercase', True):
                text = text.lower()
            
            # Remove URLs
            text = self._remove_urls(text)
            
            # Remove special characters but keep spaces
            if self.config.get('text_preprocessing.remove_punctuation', True):
                text = self._remove_punctuation(text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Remove stopwords
            if self.config.get('text_preprocessing.remove_stopwords', True):
                text = self._remove_stopwords(text)
            
            # Min token length filtering
            min_length = self.config.get('text_preprocessing.min_token_length', 3)
            if min_length > 1:
                text = self._filter_by_length(text, min_length)
            
            return text
        except Exception as e:
            self.logger.error(f"Error preprocessing text: {e}")
            return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        try:
            text = self.preprocess(text)
            tokens = text.split()
            return tokens
        except Exception as e:
            self.logger.error(f"Error tokenizing: {e}")
            return []
    
    def batch_preprocess(self, texts: List[str]) -> List[str]:
        """Preprocess multiple texts."""
        return [self.preprocess(text) for text in texts]
    
    def batch_tokenize(self, texts: List[str]) -> List[List[str]]:
        """Tokenize multiple texts."""
        return [self.tokenize(text) for text in texts]
    
    @staticmethod
    def _remove_urls(text: str) -> str:
        """Remove URLs from text."""
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    @staticmethod
    def _remove_punctuation(text: str) -> str:
        """Remove punctuation."""
        return re.sub(r'[^\w\s]', '', text)
    
    def _remove_stopwords(self, text: str) -> str:
        """Remove stopwords."""
        tokens = text.split()
        filtered = [t for t in tokens if t not in self.stopwords]
        return ' '.join(filtered)
    
    @staticmethod
    def _filter_by_length(text: str, min_length: int) -> str:
        """Filter tokens by minimum length."""
        tokens = text.split()
        filtered = [t for t in tokens if len(t) >= min_length]
        return ' '.join(filtered)
    
    def _load_stopwords(self) -> None:
        """Load English stopwords."""
        try:
            # Basic English stopwords
            stopwords = {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
                'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
                'or', 'that', 'the', 'to', 'was', 'will', 'with', 'you',
                'this', 'but', 'they', 'have', 'had', 'do', 'does', 'did',
                'can', 'could', 'should', 'would', 'may', 'might', 'must',
                'am', 'been', 'being', 'having', 'doing', 'i', 'me', 'my',
                'we', 'us', 'him', 'her', 'them', 'who', 'which', 'what',
                'where', 'when', 'why', 'how', 'all', 'each', 'every',
                'both', 'few', 'more', 'most', 'other', 'some', 'such',
                'no', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
                'very', 'just', 'if', 'because', 'while', 'during', 'after'
            }
            self.stopwords = stopwords
        except Exception as e:
            self.logger.warning(f"Error loading stopwords: {e}")
            self.stopwords = set()


class TokenAnalyzer:
    """Analyze token statistics and patterns."""
    
    def __init__(self, config):
        """Initialize token analyzer."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_token_frequency(self, tokens: List[str]) -> Dict[str, int]:
        """Get frequency of each token."""
        frequency = {}
        for token in tokens:
            frequency[token] = frequency.get(token, 0) + 1
        return dict(sorted(frequency.items(), key=lambda x: x[1], reverse=True))
    
    def get_top_tokens(self, tokens: List[str], top_k: int = 10) -> List[str]:
        """Get top-k most frequent tokens."""
        freq = self.get_token_frequency(tokens)
        return list(freq.keys())[:top_k]
    
    def get_token_statistics(self, tokens: List[str]) -> Dict[str, Any]:
        """Get statistics about tokens."""
        try:
            if not tokens:
                return {'count': 0, 'unique': 0, 'avg_length': 0}
            
            return {
                'total_count': len(tokens),
                'unique_count': len(set(tokens)),
                'avg_length': sum(len(t) for t in tokens) / len(tokens),
                'min_length': min(len(t) for t in tokens),
                'max_length': max(len(t) for t in tokens),
                'vocabulary_richness': len(set(tokens)) / len(tokens),
            }
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}