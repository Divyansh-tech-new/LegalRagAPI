import re
from typing import Dict, List, Any, Optional
from google import genai
from google.genai import types
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        try:
            if settings.gemini_api_key:
                self.client = genai.Client(api_key=settings.gemini_api_key)
                logger.info("Gemini client initialized successfully")
            else:
                logger.warning("Gemini API key not provided")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
    
    def generateSearchQueryFromCase(self, caseFacts: str, verbose: bool = False) -> str:
        if not self.client:
            raise ValueError("Gemini client not initialized")
        
        prompt = f"""
You are a legal assistant for a retrieval system based on Indian criminal law.

Given the case facts below, generate a **concise and focused search query** with **only the most relevant legal keywords**. These should include:

- Specific **IPC sections**
- Core **legal concepts** (e.g., "right of private defence", "criminal breach of trust")
- **Crime type** (e.g., "assault", "corruption")
- Any relevant **procedural issue** (e.g., "absence of intent", "lack of evidence")

Do **not** include:
- Full sentences
- Personal names
- Generic or vague words (e.g., "man", "incident", "case", "situation")

Keep the query under **20 words**. Separate terms by commas if needed. Optimize for legal document search.

Case Facts:
\"\"\"{caseFacts}\"\"\"

Return only the search query, no explanation or prefix:
"""
        
        try:
            response = self.client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt
            )
            
            if response.text:
                query = response.text.replace("Search Query:", "").strip().strip('"').replace("\n", "")
            else:
                query = caseFacts[:50]  # Fallback to first 50 chars
            
            if verbose:
                logger.info(f"Generated RAG Query: {query}")
            
            return query
        except Exception as e:
            logger.error(f"Error generating search query: {str(e)}")
            raise ValueError(f"Search query generation failed: {str(e)}")
    
    def _build_gemini_prompt(self, input_text: str, model_verdict: str, confidence: float, 
                           support: Dict[str, List], query: Optional[str] = None) -> str:
        verdict_outcome = "a loss for the person" if model_verdict.lower() == "guilty" else "in favor of the person"
        
        prompt = f"""You are a judge evaluating a legal dispute under Indian law.

### Case Facts:
{input_text}

### Initial Model Verdict:
{model_verdict.upper()} (Confidence: {confidence * 100:.2f}%)
This verdict is interpreted as {verdict_outcome}.
"""
        
        if query:
            prompt += f"\n### Legal Query Used:\n{query}\n"
        
        prompt += "\n---\n\n### Legal References Retrieved:\n\n#### Constitution Articles (Top 5):\n"
        for i, art in enumerate(support.get("constitution", [])):
            prompt += f"- {i+1}. {str(art)}\n"
        
        prompt += "\n#### IPC Sections (Top 5):\n"
        for i, sec in enumerate(support.get("ipcSections", [])):
            prompt += f"- {i+1}. {str(sec)}\n"
        
        prompt += "\n#### IPC Case Law (Top 5):\n"
        for i, case in enumerate(support.get("ipcCase", [])):
            prompt += f"- {i+1}. {str(case)}\n"
        
        prompt += "\n#### Statutes (Top 5):\n"
        for i, stat in enumerate(support.get("statutes", [])):
            prompt += f"- {i+1}. {str(stat)}\n"
        
        prompt += "\n#### QA Texts (Top 5):\n"
        for i, qa in enumerate(support.get("qaTexts", [])):
            prompt += f"- {i+1}. {str(qa)}\n"
        
        prompt += "\n#### General Case Law (Top 5):\n"
        for i, gcase in enumerate(support.get("caseLaw", [])):
            prompt += f"- {i+1}. {str(gcase)}\n"
        
        prompt += f"""

---

### Instructions to the Judge (You):

1. Review the legal materials provided:
   - Identify which Constitution articles, IPC sections, statutes, and case laws are relevant to the facts.
   - Also note and explain which retrieved references are **not applicable** or irrelevant.

2. If relevant past cases appear in the retrieved materials, summarize them and analyze whether they support or contradict the model's verdict.

3. Using the above, assess the model's prediction:
   - If confidence is below {settings.confidence_threshold * 100}%, you may revise or retain it.
   - If confidence is {settings.confidence_threshold * 100}% or higher, retain unless clear legal grounds exist to challenge it.

4. Provide a thorough and formal legal explanation that:
   - Justifies the final decision using legal logic
   - Cites relevant IPCs, constitutional provisions, statutes, and precedents
   - Explains any reasoning for overriding the model's prediction, if applicable

5. Conclude with the following lines, formatted as shown:

Final Verdict: Guilty or Not Guilty
Verdict Changed: Yes or No

Respond in the tone of a formal Indian judge. Your explanation should reflect reasoning, neutrality, and respect for legal procedure.
"""
        return prompt
    
    def _extract_final_verdict(self, gemini_output: str) -> tuple[Optional[str], str]:
        verdict_match = re.search(r"final verdict\s*[:\-]\s*(guilty|not guilty)", gemini_output, re.IGNORECASE)
        changed_match = re.search(r"verdict changed\s*[:\-]\s*(yes|no)", gemini_output, re.IGNORECASE)
        
        final_verdict = verdict_match.group(1).lower() if verdict_match else None
        verdict_changed = "changed" if changed_match and changed_match.group(1).lower() == "yes" else "not changed"
        
        return final_verdict, verdict_changed
    
    def evaluateCaseWithGemini(self, inputText: str, modelVerdict: str, confidence: float,
                                support: Dict[str, List], searchQuery: str) -> Dict[str, Any]:
        if not self.client:
            raise ValueError("Gemini client not initialized")
        
        try:
            prompt = self._build_gemini_prompt(inputText, modelVerdict, confidence, support, searchQuery)
            
            response = self.client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt
            )
            
            geminiOutput = response.text if response.text else "No response from Gemini"
            finalVerdict, verdictChanged = self._extract_final_verdict(geminiOutput)
            
            logs = {
                "inputText": inputText,
                "modelVerdict": modelVerdict,
                "confidence": confidence,
                "support": support,
                "promptToGemini": prompt,
                "geminiOutput": geminiOutput,
                "finalVerdictByGemini": finalVerdict,
                "verdictChanged": verdictChanged,
                "ragSearchQuery": searchQuery
            }
            
            return logs
        except Exception as e:
            logger.error(f"Error in Gemini evaluation: {str(e)}")
            return {
                "error": str(e),
                "inputText": inputText,
                "modelVerdict": modelVerdict,
                "confidence": confidence,
                "ragSearchQuery": searchQuery,
                "support": None,
                "promptToGemini": None,
                "geminiOutput": None,
                "finalVerdictByGemini": None,
                "verdictChanged": None
            }
    
    def is_configured(self) -> bool:
        return self.client is not None
    
    def is_healthy(self) -> bool:
        return self.is_configured()
