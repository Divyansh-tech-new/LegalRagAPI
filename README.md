# Legal RAG Analysis API

A FastAPI backend for legal case analysis using Retrieval-Augmented Generation (RAG) system with LegalBERT predictions and Gemini AI evaluation.

## Overview

This API provides comprehensive legal case analysis by combining:
- LegalBERT model for initial verdict predictions
- RAG system with FAISS indexes for retrieving relevant legal documents
- Gemini AI for final evaluation and detailed explanations

## Features

- **Case Analysis**: Analyze legal cases and predict verdicts
- **RAG Integration**: Retrieve relevant legal documents from multiple sources
- **AI Evaluation**: Get detailed legal explanations from Gemini AI
- **Health Monitoring**: Check system status across all components
- **Model Status**: Monitor loading status of ML models and indexes

## API Endpoints

### Core Endpoints

#### `POST /api/v1/analyze-case`
Analyze a legal case and provide verdict prediction with detailed explanation.

**Request Body:**
```json
{
  "caseText": "The accused was found in possession of stolen property...",
  "useQueryGeneration": true
}
```

**Response:**
```json
{
  "initialVerdict": "guilty",
  "initialConfidence": 0.85,
  "finalVerdict": "guilty", 
  "verdictChanged": false,
  "searchQuery": "stolen property, IPC section 411, criminal breach of trust",
  "geminiExplanation": "Based on the legal analysis...",
  "supportingSources": {...},
  "analysisLogs": {...}
}
```

#### `GET /api/v1/health`
Check the health status of all system components.

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "legal_bert": true,
    "rag": true,
    "gemini": true
  },
  "error": null
}
```

#### `GET /api/v1/models/status`
Get detailed status of all models and indexes.

**Response:**
```json
{
  "legalBert": {
    "loaded": false,
    "device": "cpu"
  },
  "ragIndexes": {
    "loaded": false,
    "indexCount": 0
  },
  "gemini": {
    "configured": true
  }
}
```

## Setup Instructions

### Prerequisites

1. **Gemini API Key**: Required for AI analysis
   - Get from [Google AI Studio](https://aistudio.google.com/)
   - Add as `GEMINI_API_KEY` environment variable

2. **Model Files** (Optional for development):
   - LegalBERT model files in `./models/legalbert_model/`
   - FAISS indexes in `./faiss_indexes/`

### Installation

1. **Install Dependencies:**
   ```bash
   pip install fastapi uvicorn pydantic pydantic-settings google-genai
   ```

2. **For Full Functionality (ML Models):**
   ```bash
   pip install torch transformers sentence-transformers faiss-cpu numpy
   ```

3. **Run the Server:**
   ```bash
   python -m uvicorn main:app --host 0.0.0.0 --port 5000 --reload
   ```

## Project Structure

```
├── main.py                 # FastAPI application entry point
├── app/
│   ├── api/
│   │   └── routes.py       # API route definitions
│   ├── core/
│   │   └── config.py       # Configuration settings
│   ├── models/
│   │   └── schemas.py      # Pydantic models
│   └── services/
│       ├── legal_bert.py   # LegalBERT service
│       ├── rag_service.py  # RAG retrieval service
│       └── gemini_service.py # Gemini AI service
├── models/                 # LegalBERT model files (to be added)
└── faiss_indexes/          # FAISS indexes (to be added)
```

## Development Mode

The API works in development mode without ML dependencies:
- Uses placeholder predictions for LegalBERT
- Provides mock RAG retrieval
- Full Gemini AI integration for analysis

## Adding Model Files

To enable full functionality:

1. **LegalBERT Model:**
   - Place model files in `./models/legalbert_model/`
   - Install torch and transformers

2. **FAISS Indexes:**
   - Add indexes to `./faiss_indexes/`
   - Install faiss-cpu and sentence-transformers

## Configuration

Key settings in `app/core/config.py`:
- Model paths
- FAISS index locations  
- API configuration
- RAG parameters

## Environment Variables

- `GEMINI_API_KEY`: Required for Gemini AI integration
- `LEGAL_BERT_MODEL_PATH`: Path to LegalBERT model
- `FAISS_INDEXES_PATH`: Base path for FAISS indexes

## Usage Examples

### Basic Case Analysis
```python
import requests

response = requests.post('http://localhost:5000/api/v1/analyze-case', json={
    'caseText': 'The accused was caught stealing from a shop.',
    'useQueryGeneration': True
})

result = response.json()
print(f"Verdict: {result['finalVerdict']}")
print(f"Explanation: {result['geminiExplanation']}")
```

### Health Check
```python
import requests

health = requests.get('http://localhost:5000/api/v1/health')
print(health.json())
```

## API Documentation

Once running, visit:
- **Interactive API Docs**: http://localhost:5000/docs
- **OpenAPI Schema**: http://localhost:5000/openapi.json

## Legal Document Sources

The RAG system retrieves from:
- Indian Constitution articles
- IPC sections
- Case law precedents  
- Legal statutes
- Q&A legal content

## Notes

- The system is designed for Indian criminal law cases
- Placeholder implementations allow development without full ML setup
- All services include health monitoring for production deployment
- CORS is configured for frontend integration