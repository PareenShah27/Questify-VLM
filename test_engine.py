#!/usr/bin/env python3
"""
Simple test script to verify Questify search engine functionality.
Run this to test the core components before launching the Streamlit app.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

try:
    from main.main import QuestifySearchEngine

    print("Testing Questify Search Engine...")

    # Initialize engine
    engine = QuestifySearchEngine()
    print("Engine initialized successfully")

    # Add test documents
    test_docs = {
        "ai_doc": "Artificial intelligence and machine learning are transforming technology.",
        "python_doc": "Python is a versatile programming language for data science.",
        "web_doc": "Web development involves HTML, CSS, JavaScript and frameworks."
    }

    engine.add_documents(test_docs)
    engine.build_index()
    print(f"Added {len(test_docs)} documents and built index")

    # Test search
    query = "machine learning"
    results = engine.search(query)

    print(f"Search for '{query}' returned {results['total_results']} results")

    # Display results
    if results['total_results'] > 0:
        for i, result in enumerate(results['results'][:3], 1):
            print(f"   {i}. {result['doc_id']} (score: {result['similarity_score']:.4f})")

    # Get statistics
    stats = engine.get_statistics()
    print(f"Engine statistics:")
    print(f"   - Total documents: {stats['indexer_stats']['total_documents']}")
    print(f"   - Vocabulary size: {stats['indexer_stats']['vocabulary_size']}")
    print(f"   - Average search time: {stats['search_stats']['average_search_time']:.6f}s")

    print("\n🎉 All tests passed! Questify is ready to use.")
    print("\nTo run the Streamlit app:")
    print("   streamlit run ui/streamlit_app.py")

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're in the questify directory and have installed requirements.")
except Exception as e:
    print(f"Error: {e}")
    print("Something went wrong during testing.")
