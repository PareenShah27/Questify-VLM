"""
Questify: A high-performance text search engine using TF-IDF and cosine similarity.

This package provides a complete search engine implementation with:
- TF-IDF vectorization
- Cosine similarity calculation  
- Document storage and indexing
- Web interface with Streamlit
"""

from .files.main import QuestifySearchEngine
from .files.config import config

__version__ = "1.0.0"
__author__ = "Questify Development Team"

# Main exports
__all__ = [
    'QuestifySearchEngine',
    'config'
]
