from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional

class CaseAnalysisRequest(BaseModel):
    caseText: str = Field(..., description="The legal case text to analyze", min_length=10)
    useQueryGeneration: bool = Field(default=True, description="Whether to use Gemini for query generation in RAG")

class CaseAnalysisResponse(BaseModel):
    initialVerdict: str = Field(..., description="Initial verdict from LegalBERT model")
    initialConfidence: float = Field(..., description="Confidence score of initial verdict")
    finalVerdict: Optional[str] = Field(None, description="Final verdict after Gemini evaluation")
    verdictChanged: bool = Field(default=False, description="Whether the verdict was changed by Gemini")
    searchQuery: str = Field(..., description="Query used for RAG retrieval")
    geminiExplanation: Optional[str] = Field(None, description="Detailed explanation from Gemini AI")
    supportingSources: Dict[str, List[Any]] = Field(default_factory=dict, description="Retrieved supporting legal documents")
    analysisLogs: Dict[str, Any] = Field(default_factory=dict, description="Detailed analysis logs")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall health status")
    services: Dict[str, bool] = Field(default_factory=dict, description="Status of individual services")
    error: Optional[str] = Field(None, description="Error message if unhealthy")

class VerdictPrediction(BaseModel):
    verdict: str = Field(..., description="Predicted verdict (guilty/not guilty)")
    confidence: float = Field(..., description="Confidence score between 0 and 1")

class RAGRetrievalResult(BaseModel):
    query: str = Field(..., description="Query used for retrieval")
    supportChunks: Dict[str, List[Any]] = Field(..., description="Retrieved chunks by category")
    logs: Dict[str, Any] = Field(default_factory=dict, description="Retrieval logs")

class GeminiEvaluationRequest(BaseModel):
    inputText: str = Field(..., description="Original case text")
    modelVerdict: str = Field(..., description="Initial model verdict")
    confidence: float = Field(..., description="Confidence of initial verdict")
    support: Dict[str, List[Any]] = Field(..., description="Supporting legal documents")
    searchQuery: Optional[str] = Field(None, description="Search query used")

class GeminiEvaluationResponse(BaseModel):
    finalVerdict: Optional[str] = Field(None, description="Final verdict from Gemini")
    verdictChanged: str = Field(..., description="Whether verdict was changed")
    explanation: str = Field(..., description="Detailed legal explanation")
    relevantLaws: List[str] = Field(default_factory=list, description="Relevant laws identified")
