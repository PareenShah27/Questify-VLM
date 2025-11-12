"""
Streamlit web interface for Questify search engine.
Provides an interactive web UI for searching documents and managing the document store.
"""

import streamlit as st
import time
import os
from pathlib import Path
import sys

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from main.main import QuestifySearchEngine
from main.config import config


def initialize_search_engine():
    """Initialize the search engine with session state persistence."""
    if 'search_engine' not in st.session_state:
        st.session_state.search_engine = QuestifySearchEngine()
        
        # Add some sample documents if no documents exist
        documents = st.session_state.search_engine.document_store.get_all_documents()
        if not documents:
            sample_docs = {
                "doc1": "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
                "doc2": "Python is a powerful programming language widely used for data science and web development.",
                "doc3": "Natural language processing enables computers to understand and interpret human language.",
                "doc4": "Data science combines statistics, programming, and domain expertise to extract insights from data.",
                "doc5": "Deep learning uses neural networks with multiple layers to model complex patterns in data."
            }
            st.session_state.search_engine.add_documents(sample_docs)
            st.session_state.search_engine.build_index()


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title=config.get('ui.page_title', 'Questify - Text Search Engine'),
        page_icon="🔍",
        layout="wide"
    )
    
    # Initialize search engine
    initialize_search_engine()
    engine = st.session_state.search_engine
    
    # Main title
    st.title("🔍 Questify - Text Search Engine")
    st.markdown("*A high-performance search engine using TF-IDF and cosine similarity*")
    
    # Sidebar for configuration and document management
    with st.sidebar:
        st.header("📚 Document Management")
        
        # File upload section
        if config.get('ui.enable_file_upload', True):
            uploaded_file = st.file_uploader(
                "Upload a text document",
                type=['txt', 'md'],
                help=f"Maximum file size: {config.get('ui.max_file_size_mb', 10)}MB"
            )
            
            if uploaded_file is not None:
                if st.button("Add Document"):
                    try:
                        # Read file content
                        content = uploaded_file.read().decode('utf-8')
                        doc_id = uploaded_file.name.split('.')[0]  # Use filename without extension as doc_id
                        
                        # Add document
                        engine.document_store.add_document(
                            doc_id=doc_id,
                            content=content,
                            filename=uploaded_file.name
                        )
                        
                        # Update indexer
                        engine.indexer.add_documents({doc_id: content})
                        engine.build_index()
                        
                        st.success(f"Document '{uploaded_file.name}' added successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error adding document: {e}")
        
        # Manual document addition
        st.subheader("Add Document Manually")
        with st.form("add_document_form"):
            doc_id = st.text_input("Document ID")
            doc_content = st.text_area("Document Content", height=150)
            
            if st.form_submit_button("Add Document"):
                if doc_id and doc_content:
                    try:
                        engine.document_store.add_document(doc_id, doc_content)
                        engine.indexer.add_documents({doc_id: doc_content})
                        engine.build_index()
                        st.success(f"Document '{doc_id}' added successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding document: {e}")
                else:
                    st.error("Please provide both document ID and content.")
        
        # Document list
        st.subheader("📋 Document Library")
        documents = engine.list_documents()
        
        if documents:
            st.write(f"**Total Documents:** {len(documents)}")
            
            for doc in documents[:10]:  # Show first 10 documents
                with st.expander(f"{doc['doc_id']} ({doc['content_length']} chars)"):
                    st.write(f"**Filename:** {doc.get('filename', 'N/A')}")
                    content = engine.document_store.get_document_content(doc['doc_id'])
                    if content:
                        preview = content[:200] + "..." if len(content) > 200 else content
                        st.write(f"**Preview:** {preview}")
                    
                    if st.button(f"Remove {doc['doc_id']}", key=f"remove_{doc['doc_id']}"):
                        if engine.remove_document(doc['doc_id']):
                            st.success(f"Document '{doc['doc_id']}' removed!")
                            st.rerun()
                        else:
                            st.error("Failed to remove document.")
            
            if len(documents) > 10:
                st.info(f"Showing first 10 of {len(documents)} documents")
        else:
            st.info("No documents in the library yet.")
    
    # Main search interface
    st.header("🔎 Search Documents")
    
    # Search input
    search_query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., machine learning algorithms",
        help="Enter keywords to search across all documents"
    )
    
    # Search settings
    col1, col2, col3 = st.columns(3)
    with col1:
        max_results = st.slider("Max Results", 1, 20, config.get('search.max_results', 10))
    with col2:
        min_similarity = st.slider("Min Similarity", 0.0, 1.0, config.get('search.min_similarity_score', 0.01), 0.01)
    with col3:
        show_scores = st.checkbox("Show Similarity Scores", True)
    
    # Update ranker settings
    engine.ranker.max_results = max_results
    engine.ranker.min_similarity_score = min_similarity
    
    # Search execution
    if st.button("Search", type="primary") or (search_query and len(search_query) > 2):
        if search_query:
            with st.spinner("Searching..."):
                start_time = time.time()
                results = engine.search(search_query)
                search_time = time.time() - start_time
            
            # Display results
            st.subheader("📊 Search Results")
            
            if results['total_results'] > 0:
                # Results summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Results Found", results['total_results'])
                with col2:
                    st.metric("Search Time", f"{search_time:.4f}s")
                with col3:
                    st.metric("Candidates Processed", results.get('total_candidates', 0))
                
                # Individual results
                for i, result in enumerate(results['results'], 1):
                    with st.container():
                        st.markdown("---")
                        
                        # Result header
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.subheader(f"#{i}: {result['doc_id']}")
                        with col2:
                            if show_scores:
                                st.metric("Similarity", f"{result['similarity_score']:.4f}")
                        
                        # Document content
                        if 'content' in result:
                            st.write("**Content:**")
                            # Show preview or full content
                            content = result['content']
                            if len(content) > 500:
                                with st.expander("Show full content"):
                                    st.text(content)
                                st.text(content[:500] + "...")
                            else:
                                st.text(content)
                        
                        # Metadata
                        metadata = engine.document_store.get_document_metadata(result['doc_id'])
                        if metadata:
                            with st.expander("Document Details"):
                                st.json(metadata)
            
            else:
                st.warning("No results found for your query.")
                
                # Show query info
                if 'query_info' in results:
                    query_info = results['query_info']
                    if 'error' in query_info:
                        st.error(f"Query Error: {query_info['error']}")
                    elif 'message' in query_info:
                        st.info(query_info['message'])
                    
                    # Show processed query terms
                    if 'terms' in query_info:
                        st.write("**Processed Query Terms:**", query_info['terms'])
        
        else:
            st.warning("Please enter a search query.")
    
    # Statistics section
    st.header("📈 Engine Statistics")
    
    with st.expander("View Detailed Statistics"):
        stats = engine.get_statistics()
        
        # Create tabs for different statistics
        tab1, tab2, tab3, tab4 = st.tabs(["Search Stats", "Index Stats", "Storage Stats", "Configuration"])
        
        with tab1:
            st.subheader("Search Performance")
            search_stats = stats['search_stats']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Searches", search_stats['total_searches'])
                st.metric("Average Search Time", f"{search_stats['average_search_time']:.6f}s")
            with col2:
                st.metric("Total Search Time", f"{search_stats['total_search_time']:.6f}s")
                st.metric("Last Search Time", f"{search_stats['last_search_time']:.6f}s")
        
        with tab2:
            st.subheader("Index Information")
            indexer_stats = stats['indexer_stats']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Documents", indexer_stats['total_documents'])
                st.metric("Vocabulary Size", indexer_stats['vocabulary_size'])
            with col2:
                st.metric("Avg Document Length", f"{indexer_stats['average_document_length']:.1f} tokens")
                efficiency = indexer_stats['vocabulary_size'] / max(1, indexer_stats['total_documents'])
                st.metric("Vocabulary Efficiency", f"{efficiency:.2f}x")
        
        with tab3:
            st.subheader("Storage Information")
            storage_stats = stats['storage_stats']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents Stored", storage_stats['total_documents'])
                st.metric("Total Size", f"{storage_stats['total_size_chars']} characters")
            with col2:
                st.metric("Average Doc Size", f"{storage_stats['avg_document_size']:.1f} chars")
                st.info(f"Storage Path: {storage_stats['storage_path']}")
        
        with tab4:
            st.subheader("Current Configuration")
            config_stats = stats['configuration']
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Search Settings:**")
                st.write(f"- Max Results: {config_stats['max_results']}")
                st.write(f"- Min Similarity Score: {config_stats['min_similarity_score']}")
            with col2:
                st.write("**Text Processing:**")
                st.write(f"- Remove Stopwords: {config_stats['remove_stopwords']}")
                st.write(f"- Min Token Length: {config_stats['min_token_length']}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with ❤️ using Streamlit | Questify Search Engine v1.0</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()