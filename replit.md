# Legal RAG Analysis API

## Overview

This is a FastAPI-based legal case analysis system that combines multiple AI technologies to provide comprehensive legal verdict predictions. The system uses a Retrieval-Augmented Generation (RAG) approach with LegalBERT for initial predictions and Gemini AI for final evaluation and explanation.

## User Preferences

Preferred communication style: Simple, everyday language.
Coding style: camelCase for function names, clear variable names, efficiency over boilerplate, no comments unless asked.
Prefers to add model files and dependencies later after basic structure is ready.

## System Architecture

The application follows a microservices-inspired architecture with clear separation of concerns:

### Backend Framework
- **FastAPI** - Chosen for its high performance, automatic API documentation, and excellent type hinting support
- **Python 3.x** - Primary language for ML/AI integration and legal domain processing
- **Uvicorn** - ASGI server for production-ready deployment

### AI/ML Pipeline Architecture
The system implements a three-stage analysis pipeline:
1. **Initial Prediction** - LegalBERT model for binary classification (guilty/not guilty)
2. **Knowledge Retrieval** - RAG system using FAISS for retrieving relevant legal documents
3. **Final Evaluation** - Gemini AI for contextual analysis and explanation generation

## Key Components

### 1. LegalBERT Service (`app/services/legal_bert.py`)
- **Purpose**: Provides initial verdict predictions using a fine-tuned BERT model for legal texts
- **Technology**: Transformers library with PyTorch backend
- **Input**: Raw case text
- **Output**: Binary verdict (guilty/not guilty) with confidence scores

### 2. RAG Service (`app/services/rag_service.py`)
- **Purpose**: Retrieves relevant legal documents to support case analysis
- **Technology**: 
  - FAISS for vector similarity search
  - Sentence-BERT (BGE-Large) for text embeddings
  - Multiple legal document indexes (Constitution, IPC, case law, statutes)
- **Features**: Parallel index querying, chunk deduplication, relevance filtering

### 3. Gemini Service (`app/services/gemini_service.py`)
- **Purpose**: Generates search queries and provides final legal analysis
- **Technology**: Google Gemini AI API
- **Functions**:
  - Query generation from case facts
  - Final verdict evaluation with legal reasoning
  - Explanation generation in natural language

### 4. API Layer (`app/api/routes.py`)
- **Endpoints**:
  - `POST /api/v1/analyze-case` - Main analysis endpoint
  - `GET /api/v1/health` - Service health monitoring
- **Features**: Error handling, logging, service orchestration

## Data Flow

1. **Request Processing**: Case text received via POST request
2. **Initial Analysis**: LegalBERT processes text and returns preliminary verdict
3. **Query Generation**: Gemini generates optimized search query from case facts
4. **Knowledge Retrieval**: RAG system searches multiple legal document indexes
5. **Final Evaluation**: Gemini analyzes initial verdict against retrieved legal context
6. **Response Assembly**: Combined results with explanations returned to client

## External Dependencies

### AI/ML Models
- **LegalBERT Model**: Custom fine-tuned model for legal verdict prediction
- **Sentence Transformer**: BAAI/bge-large-en-v1.5 for text embeddings
- **Gemini AI**: Google's generative AI for natural language processing

### Vector Databases
- **FAISS Indexes**: Multiple pre-built indexes for different legal document types:
  - Constitution documents
  - Indian Penal Code (IPC)
  - Case law precedents
  - Statutes and regulations
  - Q&A legal content

### Python Libraries
- FastAPI, Uvicorn (web framework)
- Transformers, PyTorch (ML models)
- Sentence-Transformers (embeddings)
- FAISS (vector search)
- Google GenAI (external API)

## Deployment Strategy

### Development Setup
- Local development with hot reload enabled
- Model files and indexes loaded from configurable paths
- Environment-based configuration management

### Configuration Management
- Centralized settings in `app/core/config.py`
- Environment variable support for sensitive data (API keys)
- Flexible path configuration for model and index files

### Health Monitoring
- Service-level health checks for all components
- Graceful degradation when external services are unavailable
- Comprehensive logging for debugging and monitoring

### CORS Configuration
- Permissive CORS setup for development
- Can be restricted for production deployment

## Current Development Status (January 2025)

### ✅ Completed
- FastAPI backend structure with proper routing and middleware
- Placeholder implementations for all services (LegalBERT, RAG, Gemini)
- Full Gemini AI integration for query generation and case evaluation
- Health monitoring endpoints for all components
- CORS configuration for frontend integration
- API documentation with comprehensive endpoints
- Camel case naming conventions as per user preference

### 🔄 Ready for Model Integration
- LegalBERT service structure ready for torch/transformers integration
- RAG service prepared for FAISS indexes and sentence-transformers
- Configuration paths set for model files and indexes
- Graceful degradation when model files are missing

### 📁 Directory Structure for Model Files
- `./models/legalbert_model/` - LegalBERT model files (to be added)
- `./faiss_indexes/` - FAISS vector indexes and chunks (to be added)

### 🔗 API Endpoints Working
- `POST /api/v1/analyze-case` - Full case analysis with Gemini evaluation
- `GET /api/v1/health` - Service health monitoring  
- `GET /api/v1/models/status` - Model loading status
- `GET /` - Basic API info

## Next Steps for Full Functionality

1. Add LegalBERT model files to `./models/legalbert_model/`
2. Install ML dependencies: `torch`, `transformers`, `sentence-transformers`, `faiss-cpu`
3. Add FAISS indexes and chunk files to `./faiss_indexes/`
4. All placeholder implementations will automatically switch to real ML models

## Notes

- The system gracefully handles missing model files during development
- All services include health check mechanisms for monitoring
- The RAG system supports parallel querying of multiple legal document indexes
- Query generation is optimized for Indian criminal law terminology
- The architecture supports easy addition of new legal document indexes
- API follows camelCase conventions and clean code principles as requested