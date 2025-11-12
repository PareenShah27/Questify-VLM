import time
import os
from typing import Dict, List, Optional, Any
from pathlib import Path

# Core search components
from core.indexer import TFIDFIndexer
from core.similarity import CosineSimilarityCalculator
from core.query_processor import QueryProcessor
from core.ranker import ResultRanker
from utils.text_preprocessor import TextPreprocessor
from data_manager.document_store import DocumentStore

# Import config with proper instantiation
from files.config import config


class QuestifySearchEngine:
    """
    FIXED: Main search engine class supporting text search.
    Now properly handles all initialization and provides clean API.
    """
    
    def __init__(self, custom_config: Optional[Dict] = None):
        """
        Initialize Questify Search Engine.
        
        Args:
            custom_config: Optional custom configuration dictionary
        """
        # Apply custom configuration if provided
        if custom_config:
            for section, settings in custom_config.items():
                config.update_section(section, settings)
        
        # Initialize text search components
        self.preprocessor = TextPreprocessor(
            remove_stopwords=config.get('text_preprocessing.remove_stopwords', True),
            min_token_length=config.get('text_preprocessing.min_token_length', 3)
        )
        
        self.text_indexer = TFIDFIndexer(self.preprocessor)
        self.text_similarity = CosineSimilarityCalculator()
        self.query_processor = QueryProcessor(self.preprocessor)
        self.ranker = ResultRanker(
            max_results=config.get('search.max_results', 10),
            min_similarity_score=config.get('search.min_similarity_score', 0.01)
        )
        
        # Initialize document store
        self.document_store = DocumentStore(
            storage_path=config.get('storage.documents_path', 'documents')
        )
        
        # VLM components (optional)
        self.vlm_enabled = False
        self.vlm_embedder = None
        self.vlm_indexer = None
        
        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'text_searches': 0,
            'average_search_time': 0.0,
            'last_search_time': 0.0
        }
        
        # Load existing documents
        self._load_documents_from_store()
    
    def _load_documents_from_store(self) -> None:
        """Load documents from storage and build index."""
        try:
            documents = self.document_store.get_all_documents()
            if documents:
                self.add_documents(documents)
                self.build_text_index()
                print(f"✓ Loaded {len(documents)} documents from storage")
        except Exception as e:
            print(f"⚠ Could not load documents from store: {e}")
    
    def add_documents(self, documents: Dict[str, str]) -> None:
        """
        Add text documents to search engine.
        
        Args:
            documents: Dictionary mapping doc_id to document content
        """
        try:
            # Add to document store
            for doc_id, content in documents.items():
                self.document_store.add_document(doc_id, content)
            
            # Add to indexer
            self.text_indexer.add_documents(documents)
            print(f"✓ Added {len(documents)} documents")
        except Exception as e:
            print(f"✗ Error adding documents: {e}")
    
    def build_text_index(self) -> None:
        """Build the text search index."""
        try:
            start_time = time.time()
            self.text_indexer.build_index()
            build_time = time.time() - start_time
            print(f"✓ Text index built in {build_time:.4f} seconds")
        except Exception as e:
            print(f"✗ Error building index: {e}")
    
    # Alias for backward compatibility
    def build_index(self) -> None:
        """Alias for build_text_index() for backward compatibility."""
        self.build_text_index()
    
    def search(self, query: str, mode: str = 'text', top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Search documents using text search.
        
        Args:
            query: Search query string
            mode: Search mode (currently only 'text' supported in this class)
            top_k: Number of results (default: from config)
        
        Returns:
            Dictionary with search results and metadata
        """
        start_time = time.time()
        
        if top_k is None:
            top_k = config.get('search.max_results', 10)
        
        self.ranker.max_results = top_k if top_k is not None else self.ranker.max_results
        
        try:
            # Process query
            query_terms = self.query_processor.process_query(query)
            
            if not self.query_processor.validate_query(query_terms):
                return self._error_response("Invalid or empty query", 0)
            
            # Get query vector
            query_vector = self.text_indexer.get_query_vector(query_terms)
            if not query_vector:
                return self._error_response("No matching terms found", 0)
            
            # Get candidates
            candidates = self.text_indexer.get_candidate_documents(query_terms)
            if not candidates:
                return {
                    'results': [],
                    'total_results': 0,
                    'total_candidates': 0,
                    'search_mode': 'text'
                }
            
            # Calculate similarities
            doc_vectors = {
                doc_id: self.text_indexer.tfidf_vectors[doc_id]
                for doc_id in candidates
            }
            doc_norms = {
                doc_id: self.text_indexer.document_norms[doc_id]
                for doc_id in candidates
            }
            
            similarities = self.text_similarity.batch_calculate_similarities(
                query_vector, doc_vectors, doc_norms
            )
            
            # Rank results
            results = self.ranker.rank_results(similarities, self.document_store)
            results['search_mode'] = mode
            results['total_candidates'] = len(candidates)
            
            # Update statistics
            search_time = time.time() - start_time
            self._update_search_stats(search_time, 'text')
            results['search_time'] = search_time
            
            self.search_stats['total_searches'] += 1
            
            return results
        
        except Exception as e:
            return self._error_response(f"Search error: {str(e)}", time.time() - start_time)
    
    def _update_search_stats(self, search_time: float, search_type: str = 'text') -> None:
        """Update search statistics."""
        self.search_stats['last_search_time'] = search_time
        
        if search_type == 'text':
            self.search_stats['text_searches'] += 1
        
        if self.search_stats['total_searches'] > 0:
            total_time = self.search_stats.get('total_search_time', 0) + search_time
            self.search_stats['total_search_time'] = total_time
            self.search_stats['average_search_time'] = (
                total_time / self.search_stats['total_searches']
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        try:
            text_stats = self.text_indexer.get_statistics()
            storage_stats = self.document_store.get_storage_stats()
            
            return {
                'search_stats': self.search_stats.copy(),
                'text_indexer': text_stats,
                'storage': storage_stats,
                'vlm_enabled': self.vlm_enabled
            }
        except Exception as e:
            print(f"Warning: Could not get statistics: {e}")
            return {'error': str(e)}
    
    def list_documents(self) -> List[Dict]:
        """List all documents."""
        try:
            return self.document_store.list_documents()
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove a document and rebuild index."""
        try:
            success = self.document_store.remove_document(doc_id)
            if success:
                # Rebuild index
                all_docs = self.document_store.get_all_documents()
                self.text_indexer = TFIDFIndexer(self.preprocessor)
                if all_docs:
                    self.text_indexer.add_documents(all_docs)
                    self.build_text_index()
            return success
        except Exception as e:
            print(f"Error removing document: {e}")
            return False
    
    @staticmethod
    def _error_response(error_msg: str, search_time: float) -> Dict[str, Any]:
        """Create error response."""
        return {
            'results': [],
            'total_results': 0,
            'query_info': {'error': error_msg},
            'search_time': search_time
        }


# FIXED: Proper exports and aliases
__all__ = [
    'QuestifySearchEngine',
]

# Alias for different import styles
QuestifyVLMSearchEngine = QuestifySearchEngine