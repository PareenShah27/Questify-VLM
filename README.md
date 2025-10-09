# Questify VLM: Enhanced Multimodal Search Engine

## Overview

Questify VLM is an advanced extension of the Questify search engine that integrates Vision Language Models (VLMs) for multimodal document retrieval. Building upon the robust text-based foundation, it now supports both traditional text search and cutting-edge visual document understanding inspired by the ColPali approach.

**Key Enhancement:** The engine now processes both text documents and visual documents (PDFs, images, scanned documents) through a unified interface, leveraging deep learning for comprehensive document search that captures visual layout, textual content, and semantic meaning.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   cd questify
   streamlit run streamlit_app.py
   ```

3. **Open your browser and navigate to the URL shown in the terminal (usually http://localhost:8501)**

4. **Select Search Mode:**
   - **Text Mode:** Traditional TF-IDF based text search
   - **VLM Mode:** Vision-Language multimodal search for visual documents

## Project Structure

```
questify/
├── core/                      # Core search engine logic
│   ├── indexer.py                 # TF-IDF indexing + VLM document indexing
│   ├── similarity.py              # Cosine similarity + Late interaction similarity
│   ├── query_processor.py         # Query processing for text and VLM queries
│   ├── ranker.py                  # Result ranking for both text and multimodal
│   ├── vlm_embedding.py           # NEW: VLM embedding generation and processing
│   └── late_interaction.py        # NEW: ColBERT-style late interaction mechanism
├── utils/                     # Utility components
│   ├── text_preprocessor.py       # Text preprocessing and tokenization
│   ├── image_preprocessor.py      # NEW: Image preprocessing for VLM pipeline
│   └── multimodal_utils.py        # NEW: Utilities for handling multimodal data
├── data_manager/              # Data management
│   ├── document_store.py          # Text document storage and retrieval
│   ├── image_store.py             # NEW: Image/PDF document storage
│   └── vector_store.py            # NEW: Vector embeddings storage for VLM
├── files/                     # Application files
│   ├── main.py                    # Main search engine class (enhanced)
│   ├── config.py                  # Configuration for both text and VLM modes
│   └── vlm_config.py              # NEW: VLM-specific configurations
├── documents/                 # Text document storage folder
├── images/                    # NEW: Visual document storage folder
├── models/                    # NEW: Pre-trained model cache
├── requirements.txt           # Enhanced Python dependencies
└── streamlit_app.py           # Enhanced Streamlit interface with mode selection
```

## Enhanced Features

### Text Search (Original)
- **High-Performance Search:** Sub-millisecond search times using optimized TF-IDF
- **Document Management:** Upload and manage text documents (.txt, .md)
- **Configurable:** Adjustable search parameters and preprocessing options

### VLM Search (New)
- **Multimodal Embedding:** Vision-Language Model embeddings for document images
- **Late Interaction:** ColBERT-style fine-grained similarity computation
- **Visual Document Support:** PDFs, images, scanned documents, complex layouts
- **No OCR Required:** Direct visual understanding without text extraction
- **Layout Awareness:** Captures visual structure, tables, charts, and formatting

### Quality Attributes
- **Accuracy/Retrieval Effectiveness:** Enhanced precision through multimodal understanding
- **Robustness/Generalizability:** Works across document types, languages, and visual quality
- **Performance:** Optimized indexing and retrieval for both text and visual modes

## Usage

### Text Documents (Classic Mode)
- Upload `.txt` or `.md` files via sidebar
- Enter text queries in search box
- Adjust similarity threshold and max results
- View ranked results with similarity scores

### Visual Documents (VLM Mode)
- Upload PDF files or images (`.pdf`, `.png`, `.jpg`, `.jpeg`)
- Documents automatically converted to image patches
- Enter natural language queries describing visual or textual content
- Retrieve documents based on visual layout and semantic content

### Mode Selection
- **Auto Mode:** Automatically selects best search method based on query and available documents
- **Text Mode:** Forces traditional text-based search
- **VLM Mode:** Forces vision-language model search
- **Hybrid Mode:** Combines results from both approaches

## Configuration

### Text Search Configuration (`files/config.py`)
```python
TEXT_CONFIG = {
    'max_results': 10,
    'similarity_threshold': 0.1,
    'preprocessing': {
        'remove_stopwords': True,
        'stemming': True
    }
}
```

### VLM Configuration (`files/vlm_config.py`)
```python
VLM_CONFIG = {
    'model_name': 'colpali',  # or 'paligemma'
    'batch_size': 4,
    'max_patches': 1024,
    'embedding_dim': 128,
    'late_interaction': {
        'top_k_patches': 20,
        'similarity_function': 'maxsim'
    }
}
```

## Development

### Extending Text Search
- Add custom text preprocessors in `utils/text_preprocessor.py`
- Implement alternative similarity measures in `core/similarity.py`
- Extend text storage backends in `data_manager/document_store.py`

### Extending VLM Search
- Add new VLM models in `core/vlm_embedding.py`
- Implement custom late interaction strategies in `core/late_interaction.py`
- Extend visual document processing in `utils/image_preprocessor.py`
- Add vector storage backends in `data_manager/vector_store.py`

### Integration Points
- **Unified Query Interface:** `core/query_processor.py` handles both text and VLM queries
- **Hybrid Ranking:** `core/ranker.py` can combine and re-rank results from both modes
- **Shared Utilities:** Common preprocessing and validation logic in `utils/`

## Performance Benchmarks

### Text Search
- **Index Build Time:** < 0.01s for 100 documents
- **Search Time:** < 0.001s average
- **Memory Usage:** Optimized sparse vectors

### VLM Search
- **Index Build Time:** ~1s per document page (GPU-dependent)
- **Search Time:** < 0.1s for late interaction
- **Memory Usage:** Compressed multi-vector representations
- **Accuracy Improvement:** 15-30% better retrieval effectiveness on visual documents

## Dependencies

### Core Dependencies (Enhanced)
```
streamlit>=1.28.0
numpy>=1.24.0  
scikit-learn>=1.3.0
torch>=2.0.0
transformers>=4.30.0
pillow>=9.5.0
pdf2image>=1.16.0
```

### VLM-Specific Dependencies
```
accelerate>=0.20.0
bitsandbytes>=0.41.0
einops>=0.6.1
faiss-cpu>=1.7.4  # or faiss-gpu for GPU support
sentence-transformers>=2.2.2
```

## Model Requirements

### Supported VLM Models
- **ColPali:** Efficient document retrieval with vision-language models
- **PaliGemma:** Google's vision-language model for document understanding
- **Custom Models:** Extensible architecture for new VLM implementations

### Hardware Requirements
- **CPU:** Multi-core processor (text search)
- **GPU:** NVIDIA GPU with 8GB+ VRAM (recommended for VLM search)
- **RAM:** 16GB+ recommended for large document collections
- **Storage:** SSD recommended for model caching and vector storage

## References

- **Original Questify:** [Questify Mini Text Search Engine](https://github.com/PareenShah27/Questify-A-Mini-Text-Search-Engine)
- **ColPali Paper:** [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449)
- **ColBERT:** Late interaction retrieval methodology
- **PaliGemma:** Google's vision-language model architecture


## Versions

### v2.0.0 (VLM Integration)
- Added Vision Language Model support
- Implemented late interaction mechanism
- Added multimodal document support
- Enhanced UI with mode selection
- Added vector storage capabilities
- Improved configuration management

### v1.0.0 (Original)
- Initial text-based search engine
- TF-IDF indexing and cosine similarity
- Streamlit web interface
- Basic document management
