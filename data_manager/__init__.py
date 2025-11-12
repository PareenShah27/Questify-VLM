"""
Storage components for document management.
"""

from .document_store import DocumentStore
from .image_store import ImageStore
from .vector_store import VectorStore

__all__ = [
    'DocumentStore',
    'ImageStore',
    'VectorStore'    
]