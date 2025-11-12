"""
tests/test_engine.py - Comprehensive Test Suite for Questify VLM

Tests for:
- Core search components
- Data storage
- Utility functions
- Integration tests
"""

import unittest
import logging
from pathlib import Path
from typing import Dict, List, cast
import tempfile
import numpy as np

from files.config import QuestifyConfig
from files.main import QuestifySearchEngine
from core import (
    TFIDFIndexer, VLMDocumentIndexer,
    CosineSimilarityCalculator, VLMSimilarityCalculator,
    QueryProcessor, MultimodalQueryRouter,
    ResultRanker, VLMResultRanker,
)
from data_manager import DocumentStore, ImageStore, VectorStore
from utils import (
    TextPreprocessor, TokenAnalyzer, ImagePreprocessor,
    DocumentTypeDetector, DocumentProcessor,
)


class TestConfig(unittest.TestCase):
    """Test QuestifyConfig."""
    
    def setUp(self):
        self.config = QuestifyConfig()
    
    def test_config_initialization(self):
        """Test config loads successfully."""
        self.assertIsNotNone(self.config)
    
    def test_config_get_method(self):
        """Test config get method with defaults."""
        value = self.config.get('search.search_mode', 'hybrid')
        self.assertEqual(value, 'hybrid')
    
    def test_config_summary(self):
        """Test config summary generation."""
        summary = self.config.summary()
        self.assertIsNotNone(summary)
        self.assertIsInstance(summary, str)


class TestTFIDFIndexer(unittest.TestCase):
    """Test TF-IDF text indexing."""
    
    def setUp(self):
        self.config = QuestifyConfig()
        self.indexer = TFIDFIndexer(self.config)
    
    def test_add_documents(self):
        """Test adding documents."""
        docs = {
            'doc1': 'machine learning algorithms',
            'doc2': 'deep neural networks',
        }
        self.indexer.add_documents(docs)
        self.assertEqual(self.indexer.index_size(), 2)
    
    def test_get_document_vector(self):
        """Test getting document vector."""
        docs = {'doc1': 'test content here'}
        self.indexer.add_documents(docs)
        
        vector = self.indexer.get_document_vector('doc1')
        self.assertIsNotNone(vector)
        vector = cast(np.ndarray, vector)
        self.assertGreater(len(vector), 0)
    
    def test_remove_documents(self):
        """Test removing documents."""
        docs = {'doc1': 'content', 'doc2': 'more content'}
        self.indexer.add_documents(docs)
        self.indexer.remove_documents(['doc1'])
        self.assertEqual(self.indexer.index_size(), 1)


class TestCosineSimilarity(unittest.TestCase):
    """Test cosine similarity calculator."""
    
    def setUp(self):
        self.config = QuestifyConfig()
        self.calc = CosineSimilarityCalculator(self.config)
    
    def test_similarity_calculation(self):
        """Test similarity between vectors."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([1, 0, 0])
        
        similarity = self.calc.calculate(v1, v2)
        self.assertAlmostEqual(similarity, 1.0, places=5)
    
    def test_orthogonal_vectors(self):
        """Test orthogonal vectors have zero similarity."""
        v1 = np.array([1, 0])
        v2 = np.array([0, 1])
        
        similarity = self.calc.calculate(v1, v2)
        self.assertAlmostEqual(similarity, 0.0, places=5)
    
    def test_batch_similarity(self):
        """Test batch similarity calculation."""
        query = np.array([1, 0, 0])
        docs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        similarities = self.calc.calculate_batch(query, docs)
        self.assertEqual(len(similarities), 3)


class TestQueryProcessor(unittest.TestCase):
    """Test query processing."""
    
    def setUp(self):
        self.config = QuestifyConfig()
        self.text_prep = TextPreprocessor(self.config)
        self.processor = QueryProcessor(self.config, self.text_prep)
    
    def test_query_preprocessing(self):
        """Test query preprocessing."""
        query = "  HELLO  world  "
        processed = self.processor.process(query)
        
        self.assertIsNotNone(processed)
        self.assertIsInstance(processed, str)
    
    def test_batch_processing(self):
        """Test batch query processing."""
        queries = ['query one', 'query two', 'query three']
        processed = self.processor.process_batch(queries)
        
        self.assertEqual(len(processed), 3)


class TestDocumentStore(unittest.TestCase):
    """Test document storage."""
    
    def setUp(self):
        self.config = QuestifyConfig()
        self.store = DocumentStore(self.config)
    
    def test_add_document(self):
        """Test adding document."""
        result = self.store.add_document('doc1', 'test content')
        self.assertTrue(result)
    
    def test_get_document(self):
        """Test retrieving document."""
        self.store.add_document('doc1', 'test content')
        content = self.store.get_document('doc1')
        
        self.assertEqual(content, 'test content')
    
    def test_remove_document(self):
        """Test removing document."""
        self.store.add_document('doc1', 'content')
        result = self.store.remove_document('doc1')
        
        self.assertTrue(result)
        self.assertEqual(self.store.get_document_count(), 0)
    
    def test_document_statistics(self):
        """Test document statistics."""
        self.store.add_document('doc1', 'hello world test')
        stats = self.store.get_statistics()
        
        self.assertIn('total_documents', stats)
        self.assertEqual(stats['total_documents'], 1)


class TestVectorStore(unittest.TestCase):
    """Test vector storage."""
    
    def setUp(self):
        self.config = QuestifyConfig()
        self.store = VectorStore(self.config)
    
    def test_add_vector(self):
        """Test adding vector."""
        vector = np.array([0.1, 0.2, 0.3])
        result = self.store.add_vector('vec1', vector)
        
        self.assertTrue(result)
    
    def test_get_vector(self):
        """Test retrieving vector."""
        vector = np.array([0.1, 0.2, 0.3])
        self.store.add_vector('vec1', vector)
        
        retrieved = self.store.get_vector('vec1')
        self.assertIsNotNone(retrieved)
        # cast to ndarray to satisfy type-checkers before passing to numpy assertion
        np.testing.assert_array_almost_equal(cast(np.ndarray, retrieved), vector)
    
    def test_similarity_search(self):
        """Test similarity search."""
        # Add vectors
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.9, 0.1, 0.0])
        v3 = np.array([0.0, 0.0, 1.0])
        
        self.store.add_vector('v1', v1)
        self.store.add_vector('v2', v2)
        self.store.add_vector('v3', v3)
        
        # Search
        query = np.array([1.0, 0.0, 0.0])
        results = self.store.similarity_search(query, top_k=2)
        
        self.assertEqual(len(results), 2)
        # Most similar should be v1
        self.assertEqual(results[0][0], 'v1')


class TestTextPreprocessor(unittest.TestCase):
    """Test text preprocessing."""
    
    def setUp(self):
        self.config = QuestifyConfig()
        self.preprocessor = TextPreprocessor(self.config)
    
    def test_preprocess(self):
        """Test text preprocessing."""
        text = "  Hello WORLD!  "
        result = self.preprocessor.preprocess(text)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
    
    def test_tokenize(self):
        """Test tokenization."""
        text = "hello world test"
        tokens = self.preprocessor.tokenize(text)
        
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
    
    def test_stopword_removal(self):
        """Test stopword removal."""
        text = "the quick brown fox"
        tokens = self.preprocessor.tokenize(text)
        
        # 'the' should be removed
        self.assertNotIn('the', tokens)


class TestTokenAnalyzer(unittest.TestCase):
    """Test token analysis."""
    
    def setUp(self):
        self.config = QuestifyConfig()
        self.analyzer = TokenAnalyzer(self.config)
    
    def test_token_frequency(self):
        """Test token frequency calculation."""
        tokens = ['apple', 'banana', 'apple', 'cherry']
        freq = self.analyzer.get_token_frequency(tokens)
        
        self.assertEqual(freq['apple'], 2)
        self.assertEqual(freq['banana'], 1)
    
    def test_top_tokens(self):
        """Test getting top tokens."""
        tokens = ['a', 'b', 'a', 'c', 'a', 'd']
        top = self.analyzer.get_top_tokens(tokens, top_k=2)
        
        self.assertEqual(top[0], 'a')


class TestDocumentTypeDetector(unittest.TestCase):
    """Test document type detection."""
    
    def setUp(self):
        self.config = QuestifyConfig()
        self.detector = DocumentTypeDetector(self.config)
    
    def test_detect_image(self):
        """Test image detection."""
        doc_type = self.detector.detect_document_type('image.png')
        self.assertEqual(doc_type, 'image')
    
    def test_detect_pdf(self):
        """Test PDF detection."""
        doc_type = self.detector.detect_document_type('document.pdf')
        self.assertEqual(doc_type, 'pdf')
    
    def test_detect_text(self):
        """Test text file detection."""
        doc_type = self.detector.detect_document_type('readme.txt')
        self.assertEqual(doc_type, 'text')


class TestSearchEngine(unittest.TestCase):
    """Integration tests for search engine."""
    
    def setUp(self):
        self.config = QuestifyConfig()
        self.engine = QuestifySearchEngine(self.config)
    
    def test_engine_initialization(self):
        """Test engine initializes correctly."""
        self.assertIsNotNone(self.engine)
        self.assertIsNotNone(self.engine.tfidf_indexer)
        self.assertIsNotNone(self.engine.document_store)
    
    def test_add_and_search_text(self):
        """Test adding and searching text."""
        # Add documents
        docs = {
            'doc1': 'machine learning is great',
            'doc2': 'deep learning models',
            'doc3': 'natural language processing',
        }
        self.engine.add_text_documents(docs)
        
        # Search
        results = self.engine.search_text('machine learning', top_k=3)
        
        self.assertGreater(len(results), 0)
        # First result should be doc1
        self.assertEqual(results[0]['doc_id'], 'doc1')
    
    def test_get_statistics(self):
        """Test getting engine statistics."""
        self.engine.add_text_documents({
            'doc1': 'test content'
        })
        
        stats = self.engine.get_statistics()
        
        self.assertIn('documents', stats)
        self.assertEqual(stats['documents']['text'], 1)
    
    def test_clear_all(self):
        """Test clearing all data."""
        self.engine.add_text_documents({'doc1': 'content'})
        self.engine.clear_all()
        
        stats = self.engine.get_statistics()
        self.assertEqual(stats['documents']['text'], 0)


class TestQueryRouter(unittest.TestCase):
    """Test multimodal query routing."""
    
    def setUp(self):
        self.config = QuestifyConfig()
        self.router = MultimodalQueryRouter(self.config)
    
    def test_text_query_routing(self):
        """Test text query routing."""
        query = "find algorithms"
        routing = self.router.route_query(query)
        
        self.assertIn('search_mode', routing)
        self.assertEqual(routing['search_mode'], 'text')
    
    def test_visual_query_routing(self):
        """Test visual query routing."""
        query = "find similar images"
        routing = self.router.route_query(query)
        
        self.assertIn('search_mode', routing)


def run_tests():
    """Run all tests."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestTFIDFIndexer))
    suite.addTests(loader.loadTestsFromTestCase(TestCosineSimilarity))
    suite.addTests(loader.loadTestsFromTestCase(TestQueryProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestDocumentStore))
    suite.addTests(loader.loadTestsFromTestCase(TestVectorStore))
    suite.addTests(loader.loadTestsFromTestCase(TestTextPreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestTokenAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestDocumentTypeDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestSearchEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestQueryRouter))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_tests()
    exit(0 if result.wasSuccessful() else 1)