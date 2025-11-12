#!/usr/bin/env python3
"""
test_engine.py - comprehensive test suite for local testing without Streamlit.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_subheader(text):
    """Print formatted subheader."""
    print(f"\n▶ {text}")
    print("-" * 70)


def test_imports():
    """Test that all required imports work."""
    print_header("STEP 1: Testing Imports")
    
    try:
        from files.main import QuestifySearchEngine
        print("✓ Successfully imported QuestifySearchEngine")
        
        from files.config import config, QuestifyConfig
        print("✓ Successfully imported config and QuestifyConfig")
        
        return True, (QuestifySearchEngine, config, QuestifyConfig)
    except ImportError as e:
        print(f"✗ Import Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure you're in the project root directory")
        print("  2. Check that 'files' folder exists with main.py and config.py")
        print("  3. Verify core/, utils/, and data_manager/ folders exist")
        return False, None


def test_engine_initialization(engine_class, config):
    """Test engine initialization."""
    print_header("STEP 2: Testing Engine Initialization")
    
    try:
        print("Creating QuestifySearchEngine instance...")
        start_time = time.time()
        engine = engine_class()
        init_time = time.time() - start_time
        
        print(f"✓ Engine initialized in {init_time:.4f} seconds")
        print(f"✓ Engine type: {type(engine).__name__}")
        
        return True, engine
    except Exception as e:
        print(f"✗ Error initializing engine: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_text_search(engine):
    """Test text search functionality."""
    print_header("STEP 3: Testing Text Search")
    
    try:
        # Sample documents
        test_docs = {
            "doc1": "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "doc2": "Python is a powerful programming language widely used for data science.",
            "doc3": "Natural language processing enables computers to understand human language.",
            "doc4": "Data science combines statistics, programming, and domain expertise.",
            "doc5": "Deep learning uses neural networks with multiple layers to model complex patterns."
        }
        
        print_subheader("Adding documents...")
        start_time = time.time()
        engine.add_documents(test_docs)
        engine.build_text_index()
        index_time = time.time() - start_time
        
        print(f"✓ Added {len(test_docs)} documents")
        print(f"✓ Index built in {index_time:.4f} seconds")
        
        # Test search
        print_subheader("Testing search query...")
        query = "machine learning algorithms"
        
        start_time = time.time()
        results = engine.search(query, mode="text", top_k=5)
        search_time = time.time() - start_time
        
        print(f"Query: '{query}'")
        print(f"✓ Search completed in {search_time:.6f} seconds")
        print(f"✓ Found {results['total_results']} results (from {results['total_candidates']} candidates)")
        
        # Display results
        if results['total_results'] > 0:
            print("\nTop Results:")
            for i, result in enumerate(results['results'], 1):
                doc_id = result.get('doc_id', 'Unknown')
                score = result.get('similarity_score', 0)
                print(f"  {i}. {doc_id:<15} (Score: {score:.4f})")
        
        # Get statistics
        print_subheader("Engine Statistics")
        stats = engine.get_statistics()
        
        text_stats = stats.get('text_indexer', {})
        search_stats = stats.get('search_stats', {})
        
        print("Text Indexer:")
        print(f"  - Total Documents: {text_stats.get('total_documents', 'N/A')}")
        print(f"  - Vocabulary Size: {text_stats.get('vocabulary_size', 'N/A')}")
        print(f"  - Avg Doc Length: {text_stats.get('average_document_length', 'N/A'):.1f}")
        
        print("\nSearch Performance:")
        print(f"  - Total Searches: {search_stats.get('total_searches', 0)}")
        print(f"  - Text Searches: {search_stats.get('text_searches', 0)}")
        print(f"  - Avg Search Time: {search_stats.get('average_search_time', 0):.6f}s")
        print(f"  - Last Search Time: {search_stats.get('last_search_time', 0):.6f}s")
        
        print("\n✓ Text search test PASSED!")
        return True
    
    except Exception as e:
        print(f"✗ Text search test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_searches(engine):
    """Test multiple search queries."""
    print_header("STEP 4: Testing Multiple Searches")
    
    try:
        test_queries = [
            "artificial intelligence",
            "programming language",
            "data analysis",
            "neural networks"
        ]
        
        print("Running batch searches...")
        total_time = 0
        
        for query in test_queries:
            start_time = time.time()
            results = engine.search(query, mode="text")
            search_time = time.time() - start_time
            total_time += search_time
            
            status = "✓" if results['total_results'] > 0 else "○"
            print(f"  {status} '{query}' → {results['total_results']} result(s) in {search_time:.6f}s")
        
        print(f"\n✓ Batch search complete ({total_time:.4f}s total)")
        print("\n✓ Multiple searches test PASSED!")
        return True
    
    except Exception as e:
        print(f"✗ Multiple searches test FAILED: {e}")
        return False


def test_configuration(config):
    """Test configuration system."""
    print_header("STEP 5: Testing Configuration System")
    
    try:
        print("Testing configuration access...")
        
        tests = [
            ('search.max_results', "Max results"),
            ('search.min_similarity_score', "Min similarity"),
            ('search.search_mode', "Search mode"),
            ('text_preprocessing.remove_stopwords', "Remove stopwords"),
            ('storage.documents_path', "Documents path"),
        ]
        
        for key, desc in tests:
            value = config.get(key)
            status = "✓" if value is not None else "○"
            print(f"  {status} {desc:<25} = {value}")
        
        print("\n✓ Configuration test PASSED!")
        return True
    
    except Exception as e:
        print(f"✗ Configuration test FAILED: {e}")
        return False


def test_document_management(engine):
    """Test document management operations."""
    print_header("STEP 6: Testing Document Management")
    
    try:
        # List documents
        docs = engine.list_documents()
        print(f"✓ Listed {len(docs)} documents")
        
        # Add a new document
        new_doc = {
            "test_doc": "This is a test document for validation."
        }
        engine.add_documents(new_doc)
        engine.build_text_index()
        
        updated_docs = engine.list_documents()
        print(f"✓ Added 1 document, now have {len(updated_docs)} total")
        
        # Get statistics
        stats = engine.get_statistics()
        print(f"✓ Retrieved statistics: {len(stats)} sections")
        
        print("\n✓ Document management test PASSED!")
        return True
    
    except Exception as e:
        print(f"✗ Document management test FAILED: {e}")
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "Questify - Engine Test Suite" + " " * 25 + "║")
    print("╚" + "=" * 68 + "╝")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Imports
    tests_total += 1
    success, imports = test_imports()
    if success and imports:
        tests_passed += 1
        engine_class, config, config_class = imports
    else:
        print("\n✗ CRITICAL: Imports failed. Cannot continue.")
        return False
    
    # Test 2: Engine initialization
    tests_total += 1
    success, engine = test_engine_initialization(engine_class, config)
    if success and engine:
        tests_passed += 1
    else:
        print("\n✗ CRITICAL: Engine initialization failed.")
        return False
    
    # Test 3: Text search
    tests_total += 1
    if test_text_search(engine):
        tests_passed += 1
    
    # Test 4: Multiple searches
    tests_total += 1
    if test_multiple_searches(engine):
        tests_passed += 1
    
    # Test 5: Configuration
    tests_total += 1
    if test_configuration(config):
        tests_passed += 1
    
    # Test 6: Document management
    tests_total += 1
    if test_document_management(engine):
        tests_passed += 1
    
    # Summary
    print_header("TEST SUMMARY")
    print(f"Tests Passed: {tests_passed}/{tests_total}")
    print(f"Success Rate: {(tests_passed/tests_total)*100:.1f}%")
    
    if tests_passed == tests_total:
        print("\n✓✓✓ ALL TESTS PASSED! 🎉 ✓✓✓")
        print("\nQuestify is ready to use!")
        print("\nNext steps:")
        print("  1. Start the web UI: streamlit run streamlit_app.py")
        print("  2. Or use programmatically in your code")
        return True
    else:
        print(f"\n⚠ {tests_total - tests_passed} test(s) failed")
        print("Please review the errors above.")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)