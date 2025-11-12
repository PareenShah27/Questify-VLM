"""
TF-IDF indexing module for Questify search engine.
Implements TF-IDF vectorization and inverted index construction.
"""

import math
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
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