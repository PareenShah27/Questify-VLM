"""
files/streamlit_app.py - Questify VLM Web UI

Interactive Streamlit interface for the search engine supporting:
- Text search
- Visual search
- Hybrid search
- Document management
- Configuration management
"""

import streamlit as st
import logging
from pathlib import Path
from typing import Dict, List, Optional
import time
import json

from files.main import QuestifySearchEngine
from files.config import QuestifyConfig
from utils import DocumentTypeDetector

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="Questify VLM",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM STYLING ====================

st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.2em;
        padding: 0.5rem 1rem;
    }
    .result-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f9f9f9;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================

@st.cache_resource
def get_search_engine():
    """Initialize search engine (cached)."""
    config = QuestifyConfig()
    return QuestifySearchEngine(config)

@st.cache_resource
def get_type_detector():
    """Initialize type detector (cached)."""
    config = QuestifyConfig()
    return DocumentTypeDetector(config)

# ==================== HELPER FUNCTIONS ====================

def display_result_card(result: Dict, search_type: str) -> None:
    """Display a single search result."""
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            doc_id = result.get('doc_id', 'Unknown')
            st.write(f"**Document ID:** `{doc_id}`")
            
            if search_type == 'text' or search_type == 'hybrid':
                content = result.get('content', '')
                if content:
                    preview = content[:200] + "..." if len(content) > 200 else content
                    st.write(f"**Preview:** {preview}")
            
            if search_type == 'vlm' or search_type == 'hybrid':
                path = result.get('path', '')
                if path:
                    st.write(f"**File:** {Path(path).name}")
        
        with col2:
            if search_type == 'text':
                score = result.get('score', 0)
                st.metric("Score", f"{score:.3f}")
            elif search_type == 'vlm':
                score = result.get('vlm_score', 0)
                st.metric("VLM Score", f"{score:.3f}")
            elif search_type == 'hybrid':
                score = result.get('hybrid_score', 0)
                st.metric("Hybrid Score", f"{score:.3f}")

def display_statistics(stats: Dict) -> None:
    """Display engine statistics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Text Documents", stats['documents']['text'])
    with col2:
        st.metric("Visual Documents", stats['documents']['visual'])
    with col3:
        st.metric("Vector Records", stats['documents']['vectors'])
    with col4:
        st.metric("Total Queries", stats['performance']['total_queries'])

# ==================== MAIN APP ====================

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🔍 Questify VLM - Multimodal Search Engine")
    st.markdown("**AI-Powered Text & Visual Document Search**")
    
    # Initialize engine
    engine = get_search_engine()
    type_detector = get_type_detector()
    
    # Sidebar navigation
    with st.sidebar:
        st.header("⚙️ Navigation")
        page = st.radio(
            "Select Page:",
            ["🔍 Search", "📚 Documents", "⚡ Configuration", "📊 Statistics"]
        )
        
        st.divider()
        
        st.header("🎯 Quick Settings")
        search_mode = st.selectbox(
            "Search Mode:",
            ["auto", "text", "vlm", "hybrid"],
            help="Select search type: auto=intelligent routing, text=TF-IDF, vlm=Vision-Language, hybrid=combined"
        )
        
        top_k = st.slider(
            "Top Results:",
            min_value=1,
            max_value=50,
            value=10,
            help="Number of results to return"
        )
    
    # ==================== SEARCH PAGE ====================
    
    if page == "🔍 Search":
        st.header("Search Documents")
        
        tab1, tab2, tab3 = st.tabs(["📝 Text Search", "🖼️ Visual Search", "⚡ Hybrid Search"])
        
        # Text Search Tab
        with tab1:
            st.subheader("Search Text Documents (TF-IDF)")
            
            text_query = st.text_input(
                "Enter search query:",
                placeholder="e.g., machine learning algorithms...",
                key="text_query"
            )
            
            if st.button("🔍 Search Text", key="btn_text_search"):
                if text_query:
                    with st.spinner("Searching..."):
                        start_time = time.time()
                        results = engine.search_text(text_query, top_k=top_k)
                        search_time = time.time() - start_time
                    
                    st.success(f"✅ Found {len(results)} results in {search_time:.3f}s")
                    
                    if results:
                        for i, result in enumerate(results, 1):
                            with st.container():
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"**{i}. {result.get('doc_id', 'Unknown')}**")
                                    content = result.get('content', '')
                                    if content:
                                        preview = content[:150] + "..." if len(content) > 150 else content
                                        st.caption(preview)
                                with col2:
                                    score = result.get('score', 0)
                                    st.metric("Score", f"{score:.3f}")
                                st.divider()
                else:
                    st.warning("⚠️ Please enter a search query")
        
        # Visual Search Tab
        with tab2:
            st.subheader("Search Visual Documents (VLM)")
            
            vlm_query = st.text_input(
                "Enter visual search query:",
                placeholder="e.g., find similar documents to...",
                key="vlm_query"
            )
            
            if st.button("🖼️ Search Visual", key="btn_vlm_search"):
                if vlm_query:
                    with st.spinner("Searching..."):
                        start_time = time.time()
                        results = engine.search_vlm(vlm_query, top_k=top_k)
                        search_time = time.time() - start_time
                    
                    st.success(f"✅ Found {len(results)} results in {search_time:.3f}s")
                    
                    if results:
                        for i, result in enumerate(results, 1):
                            with st.container():
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"**{i}. {result.get('doc_id', 'Unknown')}**")
                                    path = result.get('path', '')
                                    if path:
                                        st.caption(f"📄 {Path(path).name}")
                                with col2:
                                    score = result.get('vlm_score', 0)
                                    st.metric("VLM Score", f"{score:.3f}")
                                st.divider()
                else:
                    st.warning("⚠️ Please enter a search query")
        
        # Hybrid Search Tab
        with tab3:
            st.subheader("Hybrid Search (Text + Visual)")
            
            hybrid_query = st.text_input(
                "Enter search query:",
                placeholder="Search across all documents...",
                key="hybrid_query"
            )
            
            if st.button("⚡ Search Hybrid", key="btn_hybrid_search"):
                if hybrid_query:
                    with st.spinner("Searching..."):
                        start_time = time.time()
                        results = engine.search_hybrid(hybrid_query, top_k=top_k)
                        search_time = time.time() - start_time
                    
                    st.success(f"✅ Found {len(results)} results in {search_time:.3f}s")
                    
                    if results:
                        for i, result in enumerate(results, 1):
                            with st.container():
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"**{i}. {result.get('doc_id', 'Unknown')}**")
                                    if result.get('content'):
                                        st.caption(f"📝 {result['content'][:100]}...")
                                    elif result.get('path'):
                                        st.caption(f"🖼️ {Path(result['path']).name}")
                                with col2:
                                    score = result.get('hybrid_score', 0)
                                    st.metric("Score", f"{score:.3f}")
                                st.divider()
                else:
                    st.warning("⚠️ Please enter a search query")
    
    # ==================== DOCUMENTS PAGE ====================
    
    elif page == "📚 Documents":
        st.header("Document Management")
        
        tab1, tab2 = st.tabs(["➕ Add Documents", "📋 View Documents"])
        
        # Add Documents Tab
        with tab1:
            st.subheader("Add Text Documents")
            
            num_docs = st.number_input("Number of documents:", min_value=1, max_value=10, value=1)
            
            docs_to_add = {}
            for i in range(int(num_docs)):
                doc_id = st.text_input(f"Document ID {i+1}:", value=f"doc_{i+1}", key=f"doc_id_{i}")
                content = st.text_area(f"Content {i+1}:", key=f"doc_content_{i}", height=100)
                
                if doc_id and content:
                    docs_to_add[doc_id] = content
            
            if st.button("💾 Add Documents"):
                if docs_to_add:
                    with st.spinner("Adding documents..."):
                        results = engine.add_text_documents(docs_to_add)
                    
                    success_count = sum(1 for v in results.values() if v)
                    st.success(f"✅ Added {success_count}/{len(results)} documents")
                else:
                    st.warning("⚠️ Please enter document data")
        
        # View Documents Tab
        with tab2:
            st.subheader("Indexed Documents")
            
            # Text Documents
            with st.expander("📝 Text Documents"):
                text_docs = engine.document_store.list_documents()
                if text_docs:
                    st.write(f"Total: {len(text_docs)}")
                    for doc_id in text_docs:
                        with st.container():
                            st.write(f"**{doc_id}**")
                            metadata = engine.document_store.get_document_metadata(doc_id)
                            if metadata:
                                st.caption(f"Size: {metadata.get('size', 0)} bytes | Words: {metadata.get('word_count', 0)}")
                            st.divider()
                else:
                    st.info("No text documents")
            
            # Visual Documents
            with st.expander("🖼️ Visual Documents"):
                visual_docs = engine.image_store.list_documents()
                if visual_docs:
                    st.write(f"Total: {len(visual_docs)}")
                    for doc_id in visual_docs:
                        with st.container():
                            st.write(f"**{doc_id}**")
                            info = engine.image_store.get_document_info(doc_id)
                            if info:
                                st.caption(f"File: {Path(info['path']).name}")
                            st.divider()
                else:
                    st.info("No visual documents")
    
    # ==================== CONFIGURATION PAGE ====================
    
    elif page == "⚙️ Configuration":
        st.header("Configuration Management")
        
        config_display = engine.config.summary()
        st.code(config_display, language="text")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Current State"):
                with st.spinner("Saving..."):
                    if engine.save_state():
                        st.success("✅ State saved successfully")
                    else:
                        st.error("❌ Error saving state")
        
        with col2:
            if st.button("🗑️ Clear All Data"):
                with st.spinner("Clearing..."):
                    if engine.clear_all():
                        st.success("✅ All data cleared")
                    else:
                        st.error("❌ Error clearing data")
    
    # ==================== STATISTICS PAGE ====================
    
    elif page == "📊 Statistics":
        st.header("Engine Statistics")
        
        stats = engine.get_statistics()
        
        # Metrics
        display_statistics(stats)
        
        st.divider()
        
        # Performance Metrics
        st.subheader("Performance Metrics")
        perf = stats['performance']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Text Searches", perf['text_searches'])
        with col2:
            st.metric("VLM Searches", perf['vlm_searches'])
        with col3:
            st.metric("Hybrid Searches", perf['hybrid_searches'])
        with col4:
            st.metric("Avg Time (ms)", f"{perf['avg_search_time']*1000:.1f}")
        
        st.divider()
        
        # Configuration Summary
        st.subheader("Active Configuration")
        config_dict = engine.config.to_dict()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Search Settings:**")
            for key, value in config_dict.get('search', {}).items():
                st.caption(f"• {key}: {value}")
        
        with col2:
            st.write("**VLM Settings:**")
            for key, value in config_dict.get('vlm', {}).items():
                st.caption(f"• {key}: {value}")

# ==================== RUN APP ====================

if __name__ == '__main__':
    main()