"""
files/main.py - Questify VLM Search Engine Core

Main orchestrator integrating:
- Phase 1: QuestifyConfig
- Phase 2: Core search components (indexing, similarity, ranking)
- Phase 3: Data storage (documents, images, vectors)
- Phase 4/5: Utilities (preprocessing, format handling)

Provides unified API for text, VLM, and hybrid multimodal search.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from scipy.sparse import csr_matrix
import time
import numpy as np

from files.config import QuestifyConfig
from core import (
    TFIDFIndexer, VLMDocumentIndexer, HybridIndexManager,
    CosineSimilarityCalculator, VLMSimilarityCalculator, HybridSimilarityCalculator,
    QueryProcessor, VLMQueryProcessor, MultimodalQueryRouter,
    ResultRanker, VLMResultRanker, HybridResultRanker,
    VLMEmbedder, LateInteractionScorer, DocumentTokenizer
)
from data_manager import DocumentStore, ImageStore, VectorStore
from utils import (
    TextPreprocessor, TokenAnalyzer, ImagePreprocessor, PDFProcessor,
    DocumentTypeDetector, DocumentProcessor,
)


class QuestifySearchEngine:
    """
    Main Questify VLM Search Engine.
    
    Orchestrates all components for unified multimodal search:
    - Text search (TF-IDF)
    - VLM search (Vision-Language Models)
    - Hybrid search (combined)
    """
    
    def __init__(self, config: Optional[QuestifyConfig] = None):
        """
        Initialize search engine with all components.
        
        Args:
            config: QuestifyConfig instance or None (loads default)
        """
        self.config = config or QuestifyConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize utility components
        self.logger.info("Initializing Questify VLM Search Engine...")
        self._initialize_utils()
        
        # Initialize core search components
        self._initialize_core()
        
        # Initialize data storage components
        self._initialize_storage()
        
        # Initialize search components
        self._initialize_search()
        
        self.logger.info("✅ Search engine initialized successfully")
    
    def _initialize_utils(self) -> None:
        """Initialize utility components."""
        try:
            self.text_preprocessor = TextPreprocessor(self.config)
            self.token_analyzer = TokenAnalyzer(self.config)
            self.image_preprocessor = ImagePreprocessor(self.config)
            self.pdf_processor = PDFProcessor(self.config)
            self.doc_type_detector = DocumentTypeDetector(self.config)
            self.document_processor = DocumentProcessor(
                self.config, self.text_preprocessor, self.image_preprocessor
            )
            self.logger.info("✓ Utility components initialized")
        except Exception as e:
            self.logger.error(f"Error initializing utils: {e}")
            raise
    
    def _initialize_core(self) -> None:
        """Initialize core search components."""
        try:
            # Embedders
            self.vlm_embedder = VLMEmbedder(self.config)
            self.late_interaction_scorer = LateInteractionScorer(self.config)
            
            # Indexers
            self.tfidf_indexer = TFIDFIndexer(self.config)
            self.vlm_indexer = VLMDocumentIndexer(self.config, self.vlm_embedder)
            self.hybrid_index_manager = HybridIndexManager(
                self.config, self.tfidf_indexer, self.vlm_indexer
            )
            
            # Similarity calculators
            self.cosine_similarity = CosineSimilarityCalculator(self.config)
            self.vlm_similarity = VLMSimilarityCalculator(self.config)
            self.hybrid_similarity = HybridSimilarityCalculator(self.config)
            
            # Query processors
            self.query_processor = QueryProcessor(self.config, self.text_preprocessor)
            self.vlm_query_processor = VLMQueryProcessor(self.config, self.vlm_embedder)
            self.query_router = MultimodalQueryRouter(self.config)
            
            # Rankers
            self.text_ranker = ResultRanker(self.config)
            self.vlm_ranker = VLMResultRanker(self.config)
            self.hybrid_ranker = HybridResultRanker(self.config)
            
            self.logger.info("✓ Core components initialized")
        except Exception as e:
            self.logger.error(f"Error initializing core: {e}")
            raise
    
    def _initialize_storage(self) -> None:
        """Initialize data storage components."""
        try:
            self.document_store = DocumentStore(self.config)
            self.image_store = ImageStore(self.config)
            self.vector_store = VectorStore(self.config)
            self.logger.info("✓ Storage components initialized")
        except Exception as e:
            self.logger.error(f"Error initializing storage: {e}")
            raise
    
    def _initialize_search(self) -> None:
        """Initialize search state."""
        self.search_history = []
        self.performance_metrics = {
            'text_searches': 0,
            'vlm_searches': 0,
            'hybrid_searches': 0,
            'total_queries': 0,
            'avg_search_time': 0.0,
        }
    
    # ==================== TEXT SEARCH ====================
    
    def add_text_documents(self, documents: Dict[str, str]) -> Dict[str, bool]:
        """
        Add text documents for TF-IDF indexing.
        
        Args:
            documents: Dict mapping doc_id -> text content
            
        Returns:
            Dict mapping doc_id -> success
        """
        try:
            # Store documents
            store_results = self.document_store.add_documents(documents)
            
            # Index for search
            self.tfidf_indexer.add_documents(documents)
            
            self.logger.info(f"Added {len(documents)} text documents")
            return store_results
        except Exception as e:
            self.logger.error(f"Error adding text documents: {e}")
            return {doc_id: False for doc_id in documents}
    
    def search_text(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search using text-based TF-IDF.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of ranked results
        """
        try:
            start_time = time.time()
            
            # Process query
            processed_query = self.query_processor.process(query)
            # transform returns a scipy sparse matrix; convert to dense array before indexing
            matrix = csr_matrix(self.tfidf_indexer.vectorizer.transform([processed_query]))
            query_vector = matrix.toarray()[0].flatten()
            
            # Get document vectors
            doc_vectors = self.tfidf_indexer.get_all_vectors()
            # Calculate similarities
            if len(doc_vectors) > 0:
                similarities = self.cosine_similarity.calculate_batch(query_vector, doc_vectors)
            else:
                similarities = np.array([])
            
            # Create results
            results = []
            for doc_id in self.tfidf_indexer.doc_ids:
                idx = self.tfidf_indexer.doc_ids.index(doc_id)
                results.append({
                    'doc_id': doc_id,
                    'content': self.document_store.get_document(doc_id),
                    'metadata': self.document_store.get_document_metadata(doc_id),
                })
            
            # Rank results
            ranked_results = self.text_ranker.rank_results(results, np.asarray(similarities), top_k)
            ranked_results = self.text_ranker.rank_results(results, similarities, top_k)
            
            # Record metrics
            search_time = time.time() - start_time
            self._record_search('text', search_time)
            
            self.logger.info(f"Text search completed in {search_time:.3f}s")
            return ranked_results
        except Exception as e:
            self.logger.error(f"Error in text search: {e}")
            return []
    
    # ==================== VLM SEARCH ====================
    
    def add_visual_documents(self, 
                           image_paths: List[str],
                           doc_ids: List[str],
                           batch_size: int = 1) -> Dict[str, bool]:
        """
        Add visual documents for VLM indexing.
        
        Args:
            image_paths: List of image/PDF paths
            doc_ids: List of document IDs
            batch_size: Batch size for processing
            
        Returns:
            Dict mapping doc_id -> success
        """
        try:
            # Store image metadata
            docs_dict = {doc_id: path for doc_id, path in zip(doc_ids, image_paths)}
            store_results = self.image_store.add_visual_documents(docs_dict)
            
            # Index with VLM
            index_results = self.vlm_indexer.add_visual_documents(image_paths, doc_ids, batch_size)
            
            self.logger.info(f"Added {index_results['indexed']} visual documents")
            return store_results
        except Exception as e:
            self.logger.error(f"Error adding visual documents: {e}")
            return {doc_id: False for doc_id in doc_ids}
    
    def search_vlm(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search using VLM embeddings.
        
        Args:
            query: Search query (text)
            top_k: Number of results to return
            
        Returns:
            List of ranked results
        """
        try:
            start_time = time.time()
            
            # Encode query
            query_embedding = self.vlm_query_processor.encode_query(query)
            
            if query_embedding is None:
                self.logger.warning("VLM query encoding failed")
                return []
            
            # Search in vector store
            results = self.vector_store.similarity_search(
                query_embedding,
                top_k=min(top_k * 2, 100),  # Get more for reranking
                threshold=self.config.get('search.min_similarity_score', 0.01)
            )
            
            # Create result dicts
            ranked_results = []
            for doc_id, score in results:
                doc_info = self.image_store.get_document_info(doc_id)
                if doc_info:
                    ranked_results.append({
                        'doc_id': doc_id,
                        'vlm_score': score,
                        'path': doc_info.get('path'),
                        'metadata': doc_info.get('metadata'),
                    })
            
            # Record metrics
            search_time = time.time() - start_time
            self._record_search('vlm', search_time)
            
            self.logger.info(f"VLM search completed in {search_time:.3f}s")
            return ranked_results[:top_k]
        except Exception as e:
            self.logger.error(f"Error in VLM search: {e}")
            return []
    
    # ==================== HYBRID SEARCH ====================
    
    def search_hybrid(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search using combined text and VLM.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of ranked hybrid results
        """
        try:
            start_time = time.time()
            
            # Get text results
            text_results = self.search_text(query, top_k * 2)
            
            # Get VLM results
            vlm_results = self.search_vlm(query, top_k * 2)
            
            # Merge and rank
            merged_results = self.hybrid_ranker.rank_results(
                text_results, vlm_results, top_k
            )
            
            # Record metrics
            search_time = time.time() - start_time
            self._record_search('hybrid', search_time)
            
            self.logger.info(f"Hybrid search completed in {search_time:.3f}s")
            return merged_results
        except Exception as e:
            self.logger.error(f"Error in hybrid search: {e}")
            return []
    
    def search(self, query: str, search_mode: Optional[str] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Unified search interface.
        
        Args:
            query: Search query
            search_mode: 'text', 'vlm', 'hybrid', or None (auto-detect)
            top_k: Number of results
            
        Returns:
            List of ranked results
        """
        try:
            # Route query if mode not specified
            if search_mode is None:
                routing = self.query_router.route_query(query)
                search_mode = routing['search_mode']
            
            # Execute search
            if search_mode == 'text':
                results = self.search_text(query, top_k)
            elif search_mode == 'vlm':
                results = self.search_vlm(query, top_k)
            elif search_mode == 'hybrid':
                results = self.search_hybrid(query, top_k)
            else:
                results = self.search_hybrid(query, top_k)  # Default to hybrid
            
            # Record in history
            self.search_history.append({
                'query': query,
                'mode': search_mode,
                'results_count': len(results),
                'timestamp': time.time(),
            })
            
            return results
        except Exception as e:
            self.logger.error(f"Error in search: {e}")
            return []
    
    # ==================== UTILITY METHODS ====================
    
    def _record_search(self, search_type: str, duration: float) -> None:
        """Record search metrics."""
        self.performance_metrics['total_queries'] += 1
        self.performance_metrics[f'{search_type}_searches'] += 1
        
        # Update average time
        total_time = self.performance_metrics['avg_search_time'] * (self.performance_metrics['total_queries'] - 1)
        self.performance_metrics['avg_search_time'] = (total_time + duration) / self.performance_metrics['total_queries']
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            'documents': {
                'text': self.document_store.get_document_count(),
                'visual': self.image_store.get_document_count(),
                'vectors': self.vector_store.get_vector_count(),
            },
            'performance': self.performance_metrics,
            'search_history_size': len(self.search_history),
        }
    
    def clear_all(self) -> bool:
        """Clear all stored data."""
        try:
            self.document_store.clear()
            self.image_store.remove_documents(self.image_store.list_documents())
            self.vector_store.clear()
            self.logger.info("All stores cleared")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing stores: {e}")
            return False
    
    def save_state(self) -> bool:
        """Save engine state to disk."""
        try:
            self.document_store.save_to_disk()
            self.image_store.save_metadata_to_disk()
            self.vector_store.save_vectors_to_disk()
            self.logger.info("Engine state saved")
            return True
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
            return False


# ==================== EXAMPLE USAGE ====================

if __name__ == '__main__':
    # Initialize engine
    config = QuestifyConfig()
    engine = QuestifySearchEngine(config)
    
    # Add sample documents
    sample_docs = {
        'doc1': 'Machine learning is a subset of artificial intelligence',
        'doc2': 'Deep learning uses neural networks with multiple layers',
        'doc3': 'Natural language processing helps computers understand text',
    }
    engine.add_text_documents(sample_docs)
    
    # Search
    results = engine.search('machine learning', search_mode='text', top_k=3)
    print(f"Found {len(results)} results")
    for result in results:
        print(f"  - {result.get('doc_id')}: {result.get('score', 0):.3f}")
    
    # Get stats
    stats = engine.get_statistics()
    print(f"\nEngine stats: {stats}")