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
from data_manager.image_store import ImageStore

# Import config with proper instantiation
from files.config import config


class QuestifySearchEngine:
    """
    FIXED & ENHANCED: Main search engine class supporting text and visual search.
    Now properly handles all initialization and provides clean API with VLM support.
    """
    
    def __init__(self, custom_config: Optional[Dict] = None):
        """
        Initialize Questify Search Engine with VLM support.
        
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
        
        # Initialize document stores
        self.document_store = DocumentStore(
            storage_path=config.get('storage.documents_path', 'documents')
        )
        
        # NEW: Initialize image store for VLM
        self.image_store = ImageStore(
            storage_path=config.get('storage.images_path', 'images')
        )
        
        # VLM components (optional)
        self.vlm_enabled = config.get('vlm.enabled', False)
        self.vlm_embedder = None
        self.vlm_indexer = None
        
        # Initialize VLM if enabled and available
        if self.vlm_enabled:
            try:
                from core.vlm_embedding import VLMEmbedder
                from core.late_interaction import LateInteractionScorer
                
                self.vlm_embedder = VLMEmbedder(
                    model_name=config.get('vlm.model_name', 'vidore/colqwen2-v1.0'),
                    device=config.get('vlm.device', 'auto')
                )
                self.vlm_indexer = {}
                print("✓ VLM initialized successfully")
            except ImportError as e:
                print(f"⚠ VLM dependencies not available: {e}")
                self.vlm_enabled = False
            except Exception as e:
                print(f"⚠ VLM initialization failed: {e}")
                self.vlm_enabled = False
        
        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'text_searches': 0,
            'vlm_searches': 0,
            'hybrid_searches': 0,
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
            print(f"✓ Added {len(documents)} text documents")
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
    
    # 🆕 NEW METHOD: Add visual documents for VLM search
    def add_visual_documents(self, file_paths: List[str], doc_ids: List[str]) -> Dict[str, Any]:
        """
        Add visual documents (PDFs, images) for VLM-based search.
        
        Args:
            file_paths: List of file paths to visual documents
            doc_ids: List of document IDs corresponding to files
        
        Returns:
            Dictionary with indexing statistics
        """
        if not self.vlm_enabled or not self.vlm_embedder:
            return {
                'success': False,
                'error': 'VLM is not enabled or not available',
                'indexed_documents': 0
            }
        
        try:
            indexed_count = 0
            
            for file_path, doc_id in zip(file_paths, doc_ids):
                try:
                    file_path = Path(file_path)
                    
                    if not file_path.exists():
                        print(f"⚠ File not found: {file_path}")
                        continue
                    
                    # Get embeddings from VLM
                    embeddings = self.vlm_embedder.embed_document(str(file_path))
                    
                    if embeddings is not None:
                        # Store visual document metadata
                        self.image_store.add_image(doc_id, str(file_path))
                        
                        # Store embeddings in indexer
                        self.vlm_indexer[doc_id] = {
                            'embeddings': embeddings,
                            'file_path': str(file_path),
                            'indexed_at': time.time()
                        }
                        
                        indexed_count += 1
                        print(f"✓ Indexed VLM document: {doc_id}")
                    else:
                        print(f"⚠ Could not generate embeddings for {doc_id}")
                
                except Exception as e:
                    print(f"✗ Error indexing {doc_id}: {e}")
            
            return {
                'success': indexed_count > 0,
                'indexed_documents': indexed_count,
                'total_attempted': len(file_paths)
            }
        
        except Exception as e:
            print(f"✗ Error in add_visual_documents: {e}")
            return {
                'success': False,
                'error': str(e),
                'indexed_documents': 0
            }
    
    # 🆕 NEW METHOD: List visual documents
    def list_images(self) -> List[Dict]:
        """
        List all indexed visual documents.
        
        Returns:
            List of visual document metadata dictionaries
        """
        try:
            return self.image_store.list_images()
        except Exception as e:
            print(f"Error listing images: {e}")
            return []
    
    # 🆕 NEW METHOD: Remove visual document
    def remove_image(self, doc_id: str) -> bool:
        """
        Remove a visual document from index.
        
        Args:
            doc_id: Document ID to remove
        
        Returns:
            True if successfully removed, False otherwise
        """
        try:
            # Remove from image store
            success = self.image_store.remove_image(doc_id)
            
            # Remove from VLM indexer
            if doc_id in self.vlm_indexer:
                del self.vlm_indexer[doc_id]
            
            if success:
                print(f"✓ Removed visual document: {doc_id}")
            
            return success
        except Exception as e:
            print(f"Error removing image: {e}")
            return False
    
    def search(self, query: str, mode: str = 'text', top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Search documents using specified mode.
        
        Args:
            query: Search query string
            mode: Search mode ('text', 'vlm', 'hybrid', or 'auto')
            top_k: Number of results (default: from config)
        
        Returns:
            Dictionary with search results and metadata
        """
        start_time = time.time()

        top_k = int(top_k) if top_k else int(config.get('search.max_results', 10))


        self.ranker.max_results = top_k if not None else config.get('search.max_results', 10)
        
        try:
            # Route to appropriate search mode
            if mode == 'vlm' and self.vlm_enabled:
                results = self._vlm_search(query, top_k)
            elif mode == 'hybrid' and self.vlm_enabled:
                results = self._hybrid_search(query, top_k)
            else:
                # Default to text search
                results = self._text_search(query, top_k)
            
            search_time = time.time() - start_time
            results['search_time'] = search_time
            
            # Update statistics
            self.search_stats['total_searches'] += 1
            self.search_stats['last_search_time'] = search_time
            
            total_time = self.search_stats.get('total_search_time', 0) + search_time
            self.search_stats['total_search_time'] = total_time
            self.search_stats['average_search_time'] = (
                total_time / self.search_stats['total_searches']
            )
            
            return results
        
        except Exception as e:
            return self._error_response(f"Search error: {str(e)}", time.time() - start_time)
    
    def _text_search(self, query: str, top_k: int) -> Dict[str, Any]:
        """Perform text-based TF-IDF search."""
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
            results['search_mode'] = 'text'
            results['total_candidates'] = len(candidates)
            
            self.search_stats['text_searches'] += 1
            
            return results
        
        except Exception as e:
            return self._error_response(f"Text search error: {str(e)}", 0)
    
    def _vlm_search(self, query: str, top_k: int) -> Dict[str, Any]:
        """Perform VLM-based visual search."""
        try:
            if not self.vlm_enabled or not self.vlm_indexer:
                return self._error_response("VLM search not available", 0)
            
            # Get query embedding from VLM
            query_embedding = self.vlm_embedder.embed_query(query)
            
            if query_embedding is None:
                return self._error_response("Could not generate query embedding", 0)
            
            # Score all visual documents
            results_list = []
            for doc_id, doc_data in self.vlm_indexer.items():
                embeddings = doc_data.get('embeddings')
                if embeddings is not None:
                    # Calculate late interaction score
                    score = self._late_interaction_score(query_embedding, embeddings)
                    results_list.append({
                        'doc_id': doc_id,
                        'vlm_score': score
                    })
            
            # Sort by score
            results_list.sort(key=lambda x: x['vlm_score'], reverse=True)
            results_list = results_list[:top_k]
            
            self.search_stats['vlm_searches'] += 1
            
            return {
                'results': results_list,
                'total_results': len(results_list),
                'total_candidates': len(self.vlm_indexer),
                'search_mode': 'vlm'
            }
        
        except Exception as e:
            return self._error_response(f"VLM search error: {str(e)}", 0)
    
    def _hybrid_search(self, query: str, top_k: int) -> Dict[str, Any]:
        """Perform hybrid search combining text and VLM results."""
        try:
            # Get text results
            text_results = self._text_search(query, top_k)
            
            # Get VLM results
            vlm_results = self._vlm_search(query, top_k)
            
            # Combine results
            combined = {}
            
            # Add text results
            for result in text_results.get('results', []):
                doc_id = result['doc_id']
                combined[doc_id] = {
                    'doc_id': doc_id,
                    'text_score': result.get('similarity_score', 0),
                    'vlm_score': 0
                }
            
            # Add VLM results
            for result in vlm_results.get('results', []):
                doc_id = result['doc_id']
                if doc_id in combined:
                    combined[doc_id]['vlm_score'] = result.get('vlm_score', 0)
                else:
                    combined[doc_id] = {
                        'doc_id': doc_id,
                        'text_score': 0,
                        'vlm_score': result.get('vlm_score', 0)
                    }
            
            # Calculate hybrid score
            text_weight = config.get('hybrid.text_weight', 0.5)
            vlm_weight = config.get('hybrid.vlm_weight', 0.5)
            
            for doc_id in combined:
                combined[doc_id]['hybrid_score'] = (
                    combined[doc_id]['text_score'] * text_weight +
                    combined[doc_id]['vlm_score'] * vlm_weight
                )
            
            # Sort by hybrid score
            sorted_results = sorted(
                combined.values(),
                key=lambda x: x['hybrid_score'],
                reverse=True
            )[:top_k]
            
            self.search_stats['hybrid_searches'] += 1
            
            return {
                'results': sorted_results,
                'total_results': len(sorted_results),
                'total_candidates': len(combined),
                'search_mode': 'hybrid'
            }
        
        except Exception as e:
            return self._error_response(f"Hybrid search error: {str(e)}", 0)
    
    @staticmethod
    def _late_interaction_score(query_embedding, doc_embeddings, k: int = 1) -> float:
        """
        Calculate late interaction score (MaxSim).
        
        Args:
            query_embedding: Query embedding vector
            doc_embeddings: Document embedding vectors
            k: Top-k value
        
        Returns:
            Similarity score
        """
        try:
            import numpy as np
            
            # Normalize embeddings
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            
            # Calculate similarities
            if isinstance(doc_embeddings, list):
                doc_embeddings = np.array(doc_embeddings)
            
            doc_norms = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-8)
            
            # MaxSim: take maximum similarity
            similarities = np.dot(doc_norms, query_norm)
            
            # Return top-k average
            top_k_scores = np.sort(similarities)[-k:] if len(similarities) >= k else similarities
            return float(np.mean(top_k_scores))
        
        except Exception as e:
            print(f"Error calculating late interaction score: {e}")
            return 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        try:
            text_stats = self.text_indexer.get_statistics()
            storage_stats = self.document_store.get_storage_stats()
            
            vlm_stats = {}
            if self.vlm_enabled and self.vlm_indexer:
                vlm_stats = {
                    'total_vlm_documents': len(self.vlm_indexer),
                    'avg_embedding_size': sum(
                        len(d.get('embeddings', [])) for d in self.vlm_indexer.values()
                    ) / max(len(self.vlm_indexer), 1)
                }
            
            return {
                'search_stats': self.search_stats.copy(),
                'text_indexer': text_stats,
                'storage': storage_stats,
                'vlm_enabled': self.vlm_enabled,
                'vlm_indexer': vlm_stats,
                'hybrid_weights': {
                    'text_weight': config.get('hybrid.text_weight', 0.5),
                    'vlm_weight': config.get('hybrid.vlm_weight', 0.5)
                }
            }
        except Exception as e:
            print(f"Warning: Could not get statistics: {e}")
            return {'error': str(e)}
    
    def list_documents(self) -> List[Dict]:
        """List all text documents."""
        try:
            return self.document_store.list_documents()
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove a text document and rebuild index."""
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