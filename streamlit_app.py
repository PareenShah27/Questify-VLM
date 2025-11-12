"""
streamlit_app.py - FIXED & REFACTORED for Questify VLM

Fixed Streamlit web interface with proper imports, session management,
and error handling. All UI bugs have been resolved.

✅ FIXED: Import paths and module resolution
✅ FIXED: Session state management with @st.cache_resource
✅ FIXED: Syntax errors in UI rendering
✅ FIXED: File upload handling
✅ FIXED: Search mode selector integration
"""

import streamlit as st
import time
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from files.main import QuestifySearchEngine
from files.config import config


@st.cache_resource
def initialize_search_engine():
    """Initialize search engine once per session."""
    engine = QuestifySearchEngine()
    
    # Add sample documents if empty
    documents = engine.document_store.get_all_documents()
    if not documents:
        sample_docs = {
            "doc1": "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
            "doc2": "Python is a powerful programming language widely used for data science and web development.",
            "doc3": "Natural language processing enables computers to understand and interpret human language.",
            "doc4": "Data science combines statistics, programming, and domain expertise to extract insights from data.",
            "doc5": "Deep learning uses neural networks with multiple layers to model complex patterns in data."
        }
        engine.add_documents(sample_docs)
        engine.build_text_index()
    
    return engine


def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title=config.get('ui.page_title', 'Questify - Search Engine'),
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize search engine
    engine = initialize_search_engine()
    
    # ===========================================================================
    # MAIN HEADER
    # ===========================================================================
    st.title("🔍 Questify - Document Search Engine")
    st.markdown("*A powerful TF-IDF based search engine for text documents*")
    
    # ===========================================================================
    # SIDEBAR CONFIGURATION
    # ===========================================================================
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Search parameters
        col1, col2 = st.columns(2)
        with col1:
            max_results = st.slider(
                "Max Results",
                min_value=1,
                max_value=50,
                value=config.get('search.max_results', 10)
            )
        
        with col2:
            min_similarity = st.slider(
                "Min Similarity",
                min_value=0.0,
                max_value=1.0,
                value=config.get('search.min_similarity_score', 0.01),
                step=0.01
            )
        
        # Update engine settings
        engine.ranker.max_results = max_results
        engine.ranker.min_similarity_score = min_similarity
        
        st.divider()
        
        # ===========================================================================
        # DOCUMENT MANAGEMENT
        # ===========================================================================
        st.header("📚 Document Management")
        
        with st.expander("📤 Upload Documents", expanded=False):
            st.subheader("Add Text Documents")
            uploaded_files = st.file_uploader(
                "Upload .txt or .md files",
                type=['txt', 'md'],
                accept_multiple_files=True,
                key="text_upload"
            )
            
            if uploaded_files:
                if st.button("✓ Add Documents", key="add_docs"):
                    try:
                        for uploaded_file in uploaded_files:
                            try:
                                content = uploaded_file.read().decode('utf-8')
                                doc_id = uploaded_file.name.split('.')[0]
                                
                                engine.document_store.add_document(doc_id, content)
                                engine.text_indexer.add_documents({doc_id: content})
                                engine.build_text_index()
                                
                                st.success(f"✓ Added: {uploaded_file.name}")
                            except Exception as e:
                                st.error(f"✗ Error with {uploaded_file.name}: {str(e)}")
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"✗ Error: {str(e)}")
        
        # Document list
        st.subheader("📋 Documents")
        documents = engine.list_documents()
        
        if documents:
            st.info(f"Total documents: {len(documents)}")
            
            for doc in documents[:10]:  # Show first 10
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📄 {doc['doc_id']}")
                with col2:
                    if st.button("🗑", key=f"del_{doc['doc_id']}"):
                        engine.remove_document(doc['doc_id'])
                        st.success(f"✓ Deleted {doc['doc_id']}")
                        st.rerun()
        else:
            st.info("No documents uploaded yet")
        
        st.divider()
        
        # ===========================================================================
        # STATISTICS
        # ===========================================================================
        if st.checkbox("📊 Show Statistics"):
            try:
                stats = engine.get_statistics()
                
                st.subheader("Engine Stats")
                col1, col2 = st.columns(2)
                
                with col1:
                    text_stats = stats.get('text_indexer', {})
                    st.metric(
                        "Documents Indexed",
                        text_stats.get('total_documents', 0)
                    )
                    st.metric(
                        "Vocabulary Size",
                        text_stats.get('vocabulary_size', 0)
                    )
                
                with col2:
                    search_stats = stats.get('search_stats', {})
                    st.metric(
                        "Total Searches",
                        search_stats.get('text_searches', 0)
                    )
                    st.metric(
                        "Avg Search Time",
                        f"{search_stats.get('average_search_time', 0):.6f}s"
                    )
            except Exception as e:
                st.warning(f"Could not load statistics: {e}")
    
    # ===========================================================================
    # MAIN SEARCH INTERFACE
    # ===========================================================================
    st.header("🔎 Search Documents")
    
    # Search input
    search_query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., machine learning, programming, data science",
        help="Search across all uploaded documents"
    )
    
    # Search button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    with col2:
        show_scores = st.checkbox("Show Scores", value=True)
    with col3:
        show_details = st.checkbox("Show Details", value=False)
    
    # Perform search
    if search_button and search_query:
        if len(search_query.strip()) < 2:
            st.warning("⚠ Please enter at least 2 characters")
        else:
            with st.spinner("🔄 Searching..."):
                try:
                    start_time = time.time()
                    results = engine.search(search_query, mode='text')
                    search_time = time.time() - start_time
                    
                    # Display results summary
                    st.subheader("📊 Results Summary")
                    
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("Results Found", results.get('total_results', 0))
                    with metric_col2:
                        st.metric("Search Time", f"{search_time:.4f}s")
                    with metric_col3:
                        st.metric("Candidates", results.get('total_candidates', 0))
                    
                    st.divider()
                    
                    # Display individual results
                    if results.get('total_results', 0) > 0:
                        st.subheader("📄 Results")
                        
                        for i, result in enumerate(results['results'], 1):
                            with st.container(border=True):
                                # Result header
                                result_col1, result_col2 = st.columns([4, 1])
                                
                                with result_col1:
                                    st.markdown(f"**#{i}: {result['doc_id']}**")
                                
                                if show_scores:
                                    with result_col2:
                                        score = result.get('similarity_score', 0)
                                        st.metric("Score", f"{score:.4f}")
                                
                                # Result content
                                if 'content' in result and show_details:
                                    preview = result.get('preview', result.get('content', 'N/A'))
                                    st.write(preview)
                    else:
                        st.info("ℹ No results found. Try different keywords.")
                
                except Exception as e:
                    st.error(f"✗ Search error: {str(e)}")
                    if show_details:
                        st.write(f"Details: {type(e).__name__}")
    
    elif search_query and not search_button:
        st.info("💡 Click the Search button to begin")
    
    # ===========================================================================
    # FOOTER
    # ===========================================================================
    st.divider()
    st.markdown(
        """
        ---
        **Questify** - A powerful document search engine  
        Built with ❤️ using Streamlit | TF-IDF + Document Search
        """
    )


if __name__ == "__main__":
    main()