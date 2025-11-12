"""
files/streamlit_app.py - Questify VLM Web UI (Enhanced)

Interactive Streamlit interface with:
- Multi-format document upload (text, images, PDFs, docx, markdown)
- Streamlined navigation (no separate config page)
- Statistics in sidebar
- Improved UX/layout
"""

import streamlit as st
import logging
from pathlib import Path
from typing import Dict, List, Optional
import time
import tempfile
import os

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
        font-size: 1.1em;
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
    .upload-box {
        border: 2px dashed #667eea;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        background: #f0f4ff;
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

def display_result_card(result: Dict, search_type: str, index: int) -> None:
    """Display a single search result."""
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            doc_id = result.get('doc_id', 'Unknown')
            st.write(f"**{index}. {doc_id}**")
            
            if search_type == 'text' or search_type == 'hybrid':
                content = result.get('content', '')
                if content:
                    preview = content[:200] + "..." if len(content) > 200 else content
                    st.caption(f"📝 {preview}")
            
            if search_type == 'vlm' or search_type == 'hybrid':
                path = result.get('path', '')
                if path:
                    st.caption(f"📄 {Path(path).name}")
        
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
        st.metric("📝 Text Docs", stats['documents']['text'])
    with col2:
        st.metric("🖼️ Visual Docs", stats['documents']['visual'])
    with col3:
        st.metric("🔢 Vectors", stats['documents']['vectors'])
    with col4:
        st.metric("🔍 Total Queries", stats['performance']['total_queries'])

def process_uploaded_file(uploaded_file, document_id: str, engine: QuestifySearchEngine) -> bool:
    """
    Process and add uploaded file to engine.
    
    Supports: TXT, PDF, PNG, JPG, JPEG, DOCX, MD, etc.
    """
    try:
        file_ext = Path(uploaded_file.name).suffix.lower()
        
        # Text formats
        if file_ext in ['.txt', '.md', '.docx']:
            content = uploaded_file.read().decode('utf-8', errors='ignore')
            engine.add_text_documents({document_id: content})
            return True
        
        # Image formats
        elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            engine.add_visual_documents([tmp_path], [document_id])
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return True
        
        # PDF format
        elif file_ext == '.pdf':
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            engine.add_visual_documents([tmp_path], [document_id])
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return True
        
        else:
            return False
    
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return False

# ==================== MAIN APP ====================

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🔍 Questify VLM")
    st.markdown("**AI-Powered Multimodal Document Search**")
    
    # Initialize engine
    engine = get_search_engine()
    type_detector = get_type_detector()
    
    # ==================== SIDEBAR ====================
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        search_mode = st.selectbox(
            "Search Mode:",
            ["auto", "text", "vlm", "hybrid"],
            help="auto=intelligent routing, text=TF-IDF, vlm=Vision-Language, hybrid=combined"
        )
        
        top_k = st.slider(
            "Results:",
            min_value=1,
            max_value=50,
            value=10,
        )
        
        st.divider()
        
        st.header("📊 Statistics")
        stats = engine.get_statistics()
        display_statistics(stats)
        
        st.divider()
        
        # Performance details
        with st.expander("⏱️ Performance", expanded=False):
            perf = stats['performance']
            st.metric("Avg Time (ms)", f"{perf['avg_search_time']*1000:.1f}")
            st.metric("Text Searches", perf['text_searches'])
            st.metric("VLM Searches", perf['vlm_searches'])
            st.metric("Hybrid Searches", perf['hybrid_searches'])
    
    # ==================== MAIN NAVIGATION ====================
    
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "📚 Upload Documents", "📋 View Documents"])
    
    # ==================== SEARCH TAB ====================
    
    with tab1:
        st.header("Search Documents")
        
        search_query = st.text_input(
            "Enter search query:",
            placeholder="e.g., machine learning, find similar documents...",
            key="search_query"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🔍 Search", key="btn_search", use_container_width=True):
                if search_query:
                    with st.spinner("Searching..."):
                        start_time = time.time()
                        results = engine.search(search_query, search_mode=search_mode if search_mode != "auto" else None, top_k=top_k)
                        search_time = time.time() - start_time
                    
                    st.success(f"✅ Found {len(results)} results in {search_time:.2f}s")
                    if results:
                        for i, result in enumerate(results, 1):
                            display_result_card(result, search_mode or "hybrid", i)
                            st.divider()
                    else:
                        st.info("No results found.")
                else:
                    st.warning("⚠️ Please enter a search query")
        
        with col2:
            st.write("")
            st.write("")
            if st.button("🗑️ Clear", key="btn_clear", use_container_width=True):
                if engine.clear_all():
                    st.success("✅ All data cleared")
                    st.rerun()
        
        with col3:
            st.write("")
            st.write("")
            if st.button("💾 Save", key="btn_save", use_container_width=True):
                if engine.save_state():
                    st.success("✅ Saved")
    
    # ==================== UPLOAD TAB ====================
    
    with tab2:
        st.header("Upload Documents")
        
        st.markdown("""
        **Supported formats:**
        - 📝 Text: `.txt`, `.md`
        - 📄 Documents: `.docx`
        - 🖼️ Images: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`, `.webp`
        - 📑 PDFs: `.pdf`
        """)
        
        st.divider()
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["txt", "md", "docx", "pdf", "png", "jpg", "jpeg", "bmp", "gif", "webp"],
            accept_multiple_files=True,
            help="Select one or more files to upload"
        )
        
        if uploaded_files:
            st.subheader("Files to Upload")
            
            files_to_upload = []
            for uploaded_file in uploaded_files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    doc_id = st.text_input(
                        f"Document ID for {uploaded_file.name}:",
                        value=Path(uploaded_file.name).stem,
                        key=f"docid_{uploaded_file.name}"
                    )
                with col2:
                    st.write("")  # Spacing
                    file_type = Path(uploaded_file.name).suffix.lower()
                    st.caption(f"Type: {file_type}")
                
                files_to_upload.append((uploaded_file, doc_id))
            
            if st.button("⬆️ Upload All", use_container_width=True):
                with st.spinner("Uploading and indexing..."):
                    success_count = 0
                    for uploaded_file, doc_id in files_to_upload:
                        if process_uploaded_file(uploaded_file, doc_id, engine):
                            success_count += 1
                            st.success(f"✅ {uploaded_file.name} uploaded as '{doc_id}'")
                        else:
                            st.error(f"❌ Failed to process {uploaded_file.name}")
                    
                    st.info(f"Successfully uploaded {success_count}/{len(files_to_upload)} files")
    
    # ==================== VIEW DOCUMENTS TAB ====================
    
    with tab3:
        st.header("Indexed Documents")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Text Documents")
            text_docs = engine.document_store.list_documents()
            
            if text_docs:
                st.write(f"**Total: {len(text_docs)}**")
                with st.container(border=True):
                    for doc_id in text_docs:
                        col_a, col_b = st.columns([2, 1])
                        with col_a:
                            st.write(f"• `{doc_id}`")
                        with col_b:
                            if st.button("🗑️", key=f"del_text_{doc_id}", help="Delete"):
                                engine.document_store.remove_document(doc_id)
                                st.rerun()
            else:
                st.info("No text documents")
        
        with col2:
            st.subheader("🖼️ Visual Documents")
            visual_docs = engine.image_store.list_documents()
            
            if visual_docs:
                st.write(f"**Total: {len(visual_docs)}**")
                with st.container(border=True):
                    for doc_id in visual_docs:
                        col_a, col_b = st.columns([2, 1])
                        with col_a:
                            info = engine.image_store.get_document_info(doc_id)
                            filename = Path(info['path']).name if info else "Unknown"
                            st.write(f"• `{doc_id}` ({filename})")
                        with col_b:
                            if st.button("🗑️", key=f"del_visual_{doc_id}", help="Delete"):
                                engine.image_store.remove_documents([doc_id])
                                st.rerun()
            else:
                st.info("No visual documents")

# ==================== RUN APP ====================

if __name__ == '__main__':
    main()