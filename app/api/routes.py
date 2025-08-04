from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import CaseAnalysisRequest, CaseAnalysisResponse, HealthResponse
from app.services.legal_bert import LegalBertService
from app.services.rag_service import RAGService
from app.services.gemini_service import GeminiService
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

legal_bert_service = LegalBertService()
rag_service = RAGService()
gemini_service = GeminiService()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        services_status = {
            "legal_bert": legal_bert_service.is_healthy(),
            "rag": rag_service.is_healthy(),
            "gemini": gemini_service.is_healthy()
        }
        
        all_healthy = all(services_status.values())
        
        return HealthResponse(
            status="healthy" if all_healthy else "degraded",
            services=services_status,
            error=None
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            services={},
            error=str(e)
        )

@router.post("/analyze-case", response_model=CaseAnalysisResponse)
async def analyze_case(request: CaseAnalysisRequest):
    try:
        logger.info(f"Analyzing case with text length: {len(request.caseText)}")
        
        initial_verdict = legal_bert_service.predictVerdict(request.caseText)
        confidence = legal_bert_service.getConfidence(request.caseText)
        
        logger.info(f"Initial verdict: {initial_verdict}, confidence: {confidence}")
        
        evaluation_result = gemini_service.evaluateCaseWithGemini(
            inputText=request.caseText,
            modelVerdict=initial_verdict,
            confidence=confidence,
            retrieveFn=rag_service,
            geminiQueryModel=gemini_service if request.useQueryGeneration else None
        )
        
        logger.info(f"Retrieved support chunks from RAG system")
        search_query = evaluation_result.get("ragSearchQuery", request.caseText)
        
        logger.info(f"Gemini evaluation completed. Final verdict: {evaluation_result.get('finalVerdictByGemini')}")
        
        support_chunks = evaluation_result.get("support", {})
        return CaseAnalysisResponse(
            initialVerdict=initial_verdict,
            initialConfidence=confidence,
            finalVerdict=evaluation_result.get("finalVerdictByGemini"),
            verdictChanged=evaluation_result.get("verdictChanged") == "changed",
            searchQuery=search_query,
            geminiExplanation=evaluation_result.get("geminiOutput"),
            supportingSources=support_chunks,
            analysisLogs=evaluation_result
        )
        
    except Exception as e:
        logger.error(f"Error analyzing case: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/models/status")
async def get_models_status():
    try:
        status = {
            "legalBert": {
                "loaded": legal_bert_service.is_model_loaded(),
                "device": legal_bert_service.get_device()
            },
            "ragIndexes": {
                "loaded": rag_service.areIndexesLoaded(),
                "indexCount": len(rag_service.getLoadedIndexes())
            },
            "gemini": {
                "configured": gemini_service.is_configured()
            }
        }
        return status
    except Exception as e:
        logger.error(f"Error getting models status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get models status: {str(e)}")
