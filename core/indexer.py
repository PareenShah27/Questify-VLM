"""
TF-IDF indexing module for Questify search engine.
Implements TF-IDF vectorization and inverted index construction.
"""
import time
import math
from collections import defaultdict, Counter
from typing import Dict, List, Set,Union, Tuple
from pathlib import Path
from utils.text_preprocessor import TextPreprocessor


class TFIDFIndexer:
    """Builds TF-IDF vectors and inverted index for documents."""
    
    def __init__(self, preprocessor: TextPreprocessor):
        """
        Initialize TF-IDF indexer.
        
        Args:
            preprocessor: Text preprocessor instance
        """
        self.preprocessor = preprocessor
        self.documents = {}  # doc_id -> processed tokens
        self.vocabulary = {}  # term -> term_id
        self.inverted_index = defaultdict(set)  # term -> set of doc_ids
        self.document_frequencies = {}  # term -> number of docs containing term
        self.document_norms = {}  # doc_id -> L2 norm of document vector
        self.tfidf_vectors = {}  # doc_id -> {term: tfidf_score}
        self.total_documents = 0
        
    def add_documents(self, documents: Dict[str, str]) -> None:
        """
        Add documents to the index.
        
        Args:
            documents: Dictionary mapping doc_id to document text
        """
        for doc_id, text in documents.items():
            tokens = self.preprocessor.preprocess(text)
            self.documents[doc_id] = tokens
            
            # Update vocabulary and inverted index
            unique_terms = set(tokens)
            for term in unique_terms:
                if term not in self.vocabulary:
                    self.vocabulary[term] = len(self.vocabulary)
                self.inverted_index[term].add(doc_id)
        
        self.total_documents = len(self.documents)
        
    def build_index(self) -> None:
        """Build TF-IDF vectors and compute document norms."""
        # Calculate document frequencies
        for term in self.vocabulary:
            self.document_frequencies[term] = len(self.inverted_index[term])
        
        # Build TF-IDF vectors
        for doc_id, tokens in self.documents.items():
            term_counts = Counter(tokens)
            doc_length = len(tokens)
            tfidf_vector = {}
            
            for term, count in term_counts.items():
                # Calculate TF
                tf = count / doc_length
                
                # Calculate IDF
                idf = math.log(self.total_documents / self.document_frequencies[term])
                
                # Calculate TF-IDF
                tfidf_score = tf * idf
                tfidf_vector[term] = tfidf_score
            
            self.tfidf_vectors[doc_id] = tfidf_vector
            
            # Calculate document norm (L2 norm)
            norm = math.sqrt(sum(score**2 for score in tfidf_vector.values()))
            self.document_norms[doc_id] = norm
    
    def get_candidate_documents(self, query_terms: List[str]) -> Set[str]:
        """
        Get documents that contain at least one query term.
        
        Args:
            query_terms: List of query terms
            
        Returns:
            Set of candidate document IDs
        """
        candidates = set()
        for term in query_terms:
            if term in self.inverted_index:
                candidates.update(self.inverted_index[term])
        return candidates
    
    def get_query_vector(self, query_terms: List[str]) -> Dict[str, float]:
        """
        Convert query to TF-IDF vector.
        
        Args:
            query_terms: List of query terms
            
        Returns:
            Query TF-IDF vector
        """
        if not query_terms:
            return {}
            
        term_counts = Counter(query_terms)
        query_length = len(query_terms)
        query_vector = {}
        
        for term, count in term_counts.items():
            if term in self.vocabulary:
                # Calculate TF
                tf = count / query_length
                
                # Calculate IDF
                idf = math.log(self.total_documents / self.document_frequencies[term])
                
                # Calculate TF-IDF
                query_vector[term] = tf * idf
        
        return query_vector
    
    def get_statistics(self) -> Dict:
        """Get indexer statistics."""
        return {
            'total_documents': self.total_documents,
            'vocabulary_size': len(self.vocabulary),
            'average_document_length': sum(len(tokens) for tokens in self.documents.values()) / max(1, self.total_documents)
        }

class VLMDocumentIndexer:
    """
    NEW: Indexes visual documents using VLM embeddings.
    Works alongside TFIDFIndexer without interfering.
    """
    
    def __init__(self, storage_dir: str = "vector_store"):
        """
        Initialize VLM Document Indexer.
        
        Args:
            storage_dir: Directory to store VLM embeddings
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.vlm_embeddings = {}  # doc_id -> embedding tensor
        self.doc_metadata = {}  # doc_id -> metadata dict
        self.indexed_doc_ids = []
        self.indexing_time = 0.0
    
    def index_visual_documents(
        self,
        image_paths: List[Union[str, Path]],
        doc_ids: List[str],
        vlm_embedder,
        batch_size: int = 4
    ) -> Dict:
        """
        Index visual documents using VLM embedder.
        
        Args:
            image_paths: Paths to image files
            doc_ids: Corresponding document IDs
            vlm_embedder: VLMEmbedder instance from core.vlm_embedding
            batch_size: Batch processing size
        
        Returns:
            Indexing statistics
        """
        from PIL import Image
        
        start_time = time.time()
        indexed_count = 0
        
        print(f"VLM Indexing: Processing {len(image_paths)} visual documents...")
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_ids = doc_ids[i:i + batch_size]
            
            try:
                # Load images
                images = [Image.open(p).convert('RGB') for p in batch_paths]
                
                # Generate VLM embeddings
                embeddings = vlm_embedder.embed_images(images, batch_size=len(images))
                
                # Store embeddings and metadata
                for doc_id, embedding, path in zip(batch_ids, embeddings, batch_paths):
                    self.vlm_embeddings[doc_id] = embedding.cpu()
                    self.doc_metadata[doc_id] = {
                        'path': str(path),
                        'embedding_shape': list(embedding.shape),
                        'indexed_time': time.time()
                    }
                    indexed_count += 1
                
                print(f"  Indexed {indexed_count}/{len(image_paths)} documents")
            
            except Exception as e:
                print(f"  Warning: Error indexing batch: {e}")
        
        self.indexing_time = time.time() - start_time
        self.indexed_doc_ids = list(self.vlm_embeddings.keys())
        
        stats = {
            'indexed_documents': indexed_count,
            'indexing_time_seconds': self.indexing_time,
            'docs_per_second': indexed_count / self.indexing_time if self.indexing_time > 0 else 0
        }
        
        print(f"VLM Indexing complete: {indexed_count} docs in {self.indexing_time:.2f}s")
        return stats
    
    def get_embedding(self, doc_id: str):
        """Get VLM embedding for document."""
        return self.vlm_embeddings.get(doc_id)
    
    def get_all_embeddings(self) -> Dict:
        """Get all VLM embeddings."""
        return self.vlm_embeddings
    
    def get_statistics(self) -> Dict:
        """Get VLM indexing statistics."""
        return {
            'total_vlm_documents': len(self.vlm_embeddings),
            'indexing_time': self.indexing_time,
            'avg_embedding_size': sum(
                e.shape[0] for e in self.vlm_embeddings.values()
            ) / max(1, len(self.vlm_embeddings))
        }


class HybridIndexManager:
    """
    NEW: Manages both text (TF-IDF) and VLM indexes.
    Provides unified interface without modifying original classes.
    """
    
    def __init__(self, text_preprocessor: TextPreprocessor):
        """
        Initialize Hybrid Index Manager.
        
        Args:
            text_preprocessor: Text preprocessor for TF-IDF indexer
        """
        # Initialize both indexers
        self.text_indexer = TFIDFIndexer(text_preprocessor)
        self.vlm_indexer = VLMDocumentIndexer()
        
        self.mode = 'text'  # Default mode
    
    def index_text_documents(self, documents: Dict[str, str]) -> None:
        """
        Index text documents using original TF-IDF indexer.
        
        Args:
            documents: Dict mapping doc_id to text content
        """
        self.text_indexer.add_documents(documents)
        self.text_indexer.build_index()
        print(f"Text indexing: {len(documents)} documents indexed")
    
    def index_vlm_documents(
        self,
        image_paths: List,
        doc_ids: List[str],
        vlm_embedder,
        batch_size: int = 4
    ) -> Dict:
        """
        Index visual documents using VLM indexer.
        
        Args:
            image_paths: Paths to images
            doc_ids: Document IDs
            vlm_embedder: VLMEmbedder instance
            batch_size: Batch size
        
        Returns:
            Indexing statistics
        """
        stats = self.vlm_indexer.index_visual_documents(
            image_paths, doc_ids, vlm_embedder, batch_size
        )
        return stats
    
    def set_mode(self, mode: str):
        """
        Set search mode.
        
        Args:
            mode: 'text', 'vlm', or 'hybrid'
        """
        if mode not in ['text', 'vlm', 'hybrid']:
            raise ValueError(f"Invalid mode: {mode}. Use 'text', 'vlm', or 'hybrid'")
        self.mode = mode
    
    def get_statistics(self) -> Dict:
        """Get combined statistics from both indexers."""
        text_stats = self.text_indexer.get_statistics()
        vlm_stats = self.vlm_indexer.get_statistics()
        
        return {
            'mode': self.mode,
            'text_index': text_stats,
            'vlm_index': vlm_stats,
            'total_documents': text_stats['total_documents'] + vlm_stats['total_vlm_documents']
        }