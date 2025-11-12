"""
Main Questify search engine module.
Integrates all components to provide a unified search engine interface.
"""

import time
from typing import Dict, List, Optional, Any
from pathlib import Path

from core.indexer import TFIDFIndexer
from core.similarity import CosineSimilarityCalculator
from core.query_processor import QueryProcessor
from core.ranker import ResultRanker
from utils.text_preprocessor import TextPreprocessor
from data_manager.document_store import DocumentStore
from main.config import config


class QuestifySearchEngine:
    """Main search engine class integrating all components."""
    
    def __init__(self, custom_config: Optional[Dict] = None):
        """
        Initialize Questify search engine.
        
        Args:
            custom_config: Optional custom configuration dictionary
        """
        # Apply custom configuration if provided
        if custom_config:
            for section, settings in custom_config.items():
                config.update_section(section, settings)
        
        # Initialize components
        self.preprocessor = TextPreprocessor(
            remove_stopwords=config.get('text_preprocessing.remove_stopwords', True),
            min_token_length=config.get('text_preprocessing.min_token_length', 3)
        )
        
        self.indexer = TFIDFIndexer(self.preprocessor)
        self.similarity_calculator = CosineSimilarityCalculator()
        self.query_processor = QueryProcessor(self.preprocessor)
        
        self.ranker = ResultRanker(
            max_results=config.get('search.max_results', 10),
            min_similarity_score=config.get('search.min_similarity_score', 0.01)
        )
        
        self.document_store = DocumentStore(
            storage_path=config.get('storage.documents_path', 'documents')
        )
        
        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'total_search_time': 0.0,
            'average_search_time': 0.0,
            'last_search_time': 0.0
        }
        
        # Load existing documents
        self.load_documents_from_store()
        
    def load_documents_from_store(self) -> None:
        """Load documents from document store and build index."""
        documents = self.document_store.get_all_documents()
        if documents:
            self.add_documents(documents)
            self.build_index()
    
    def add_documents(self, documents: Dict[str, str]) -> None:
        """
        Add multiple documents to the search engine.
        
        Args:
            documents: Dictionary mapping doc_id to document content
        """
        # Add to document store
        for doc_id, content in documents.items():
            self.document_store.add_document(doc_id, content)
        
        # Add to indexer
        self.indexer.add_documents(documents)
    
    def add_document_from_file(self, file_path: str, doc_id: Optional[str] = None) -> Optional[str]:
        """
        Add a document from a file.
        
        Args:
            file_path: Path to the document file
            doc_id: Optional document ID
            
        Returns:
            Document ID if successful, None otherwise
        """
        # Add to document store
        added_doc_id = self.document_store.add_document_from_file(file_path, doc_id)
        
        if added_doc_id:
            # Get the document content and add to indexer
            content = self.document_store.get_document_content(added_doc_id)
            if content:
                self.indexer.add_documents({added_doc_id: content})
                # Rebuild index to include new document
                self.build_index()
        
        return added_doc_id
    
    def build_index(self) -> None:
        """Build the search index."""
        start_time = time.time()
        self.indexer.build_index()
        build_time = time.time() - start_time
        print(f"Index built in {build_time:.4f} seconds")
    
    def search(self, query: str) -> Dict[str, Any]:
        """
        Search for documents matching the query.
        
        Args:
            query: Search query string
            
        Returns:
            Dictionary with search results and metadata
        """
        start_time = time.time()
        
        try:
            # Process query
            query_terms = self.query_processor.process_query(query)
            
            if not self.query_processor.validate_query(query_terms):
                return {
                    'results': [],
                    'total_results': 0,
                    'query_info': {'error': 'Invalid or empty query'},
                    'search_time': 0.0
                }
            
            # Get query vector
            query_vector = self.indexer.get_query_vector(query_terms)
            
            if not query_vector:
                return {
                    'results': [],
                    'total_results': 0,
                    'query_info': {'error': 'No matching terms found in vocabulary'},
                    'search_time': time.time() - start_time
                }
            
            # Get candidate documents
            candidates = self.indexer.get_candidate_documents(query_terms)
            
            if not candidates:
                return {
                    'results': [],
                    'total_results': 0,
                    'query_info': {'message': 'No documents contain the query terms'},
                    'search_time': time.time() - start_time
                }
            
            # Calculate similarities for candidates only
            candidate_vectors = {
                doc_id: self.indexer.tfidf_vectors[doc_id] 
                for doc_id in candidates
            }
            candidate_norms = {
                doc_id: self.indexer.document_norms[doc_id]
                for doc_id in candidates
            }
            
            similarities = self.similarity_calculator.batch_calculate_similarities(
                query_vector, candidate_vectors, candidate_norms
            )
            
            # Rank results
            results = self.ranker.rank_results(similarities, self.document_store)
            
            # Add query information
            search_time = time.time() - start_time
            results['query_info'] = self.query_processor.get_query_info(query_terms)
            results['search_time'] = search_time
            results['total_candidates'] = len(candidates)
            
            # Update search statistics
            self._update_search_stats(search_time)
            
            return results
            
        except Exception as e:
            return {
                'results': [],
                'total_results': 0,
                'query_info': {'error': f'Search error: {str(e)}'},
                'search_time': time.time() - start_time
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive search engine statistics.
        
        Returns:
            Dictionary with various statistics
        """
        indexer_stats = self.indexer.get_statistics()
        storage_stats = self.document_store.get_storage_stats()
        
        return {
            'indexer_stats': indexer_stats,
            'storage_stats': storage_stats,
            'search_stats': self.search_stats.copy(),
            'configuration': {
                'max_results': self.ranker.max_results,
                'min_similarity_score': self.ranker.min_similarity_score,
                'remove_stopwords': self.preprocessor.remove_stopwords,
                'min_token_length': self.preprocessor.min_token_length
            }
        }
    
    def list_documents(self) -> List[Dict]:
        """List all documents in the search engine."""
        return self.document_store.list_documents()
    
    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the search engine.
        
        Args:
            doc_id: Document ID to remove
            
        Returns:
            True if successful, False otherwise
        """
        # Remove from document store
        success = self.document_store.remove_document(doc_id)
        
        if success:
            # Rebuild index without the removed document
            # This is not optimal for performance but ensures consistency
            all_documents = self.document_store.get_all_documents()
            
            # Reinitialize indexer
            self.indexer = TFIDFIndexer(self.preprocessor)
            if all_documents:
                self.indexer.add_documents(all_documents)
                self.build_index()
        
        return success
    
    def _update_search_stats(self, search_time: float) -> None:
        """Update search performance statistics."""
        self.search_stats['total_searches'] += 1
        self.search_stats['total_search_time'] += search_time
        self.search_stats['last_search_time'] = search_time
        
        if self.search_stats['total_searches'] > 0:
            self.search_stats['average_search_time'] = (
                self.search_stats['total_search_time'] / 
                self.search_stats['total_searches']
            )