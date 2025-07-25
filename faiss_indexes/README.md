# FAISS Indexes Directory

## Required Index Files

Add the following FAISS indexes and their corresponding chunk files:

### Constitution
- `constitution_bgeLarge.index` - FAISS index for constitution articles
- `constitution_chunks.json` - Text chunks for constitution articles

### IPC Sections  
- `ipc_bgeLarge.index` - FAISS index for IPC sections
- `ipc_chunks.json` - Text chunks for IPC sections

### IPC Case Law
- `ipc_case_flat.index` - FAISS index for IPC case law
- `ipc_case_chunks.json` - Text chunks for IPC cases

### Statutes
- `statute_index.faiss` - FAISS index for legal statutes
- `statute_chunks.pkl` - Pickled chunks for statutes

### Q&A Texts
- `qa_faiss_index.idx` - FAISS index for legal Q&A
- `qa_text_chunks.json` - Text chunks for Q&A content

### General Case Law
- `case_faiss.index` - FAISS index for general case law
- `case_chunks.pkl` - Pickled chunks for case law

## Installation

Once you have the index files:

1. Install required dependencies:
   ```bash
   pip install faiss-cpu sentence-transformers numpy
   ```

2. The RAGService will automatically detect and load all available indexes when the server starts.

## Index Requirements

- Built using sentence transformer embeddings (BAAI/bge-large-en-v1.5)
- Compatible with FAISS CPU implementation
- Chunk files should contain legal text snippets
- JSON files should contain arrays of text chunks
- PKL files should contain pickled chunk data