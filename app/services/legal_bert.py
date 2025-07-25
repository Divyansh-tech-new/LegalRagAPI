from app.core.config import settings
import logging
import os
import zipfile
import hashlib

logger = logging.getLogger(__name__)

class LegalBertService:
    def __init__(self):
        self.device = "cpu"
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _extract_model_from_zip(self, zipPath: str, extractPath: str):
        """Extract LegalBERT model from zip file"""
        try:
            if not os.path.exists(zipPath):
                logger.warning(f"Model zip file not found: {zipPath}")
                return False
            
            if not os.path.exists(extractPath):
                os.makedirs(extractPath)
                logger.info(f"Created model directory: {extractPath}")
            
            # Check if model is already extracted
            if os.path.exists(os.path.join(extractPath, "config.json")):
                logger.info("Model already extracted")
                return True
            
            logger.info(f"Extracting model from {zipPath} to {extractPath}")
            with zipfile.ZipFile(zipPath, 'r') as zipRef:
                zipRef.extractall(extractPath)
            
            logger.info("Model extraction completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract model: {str(e)}")
            return False
    
    def _load_model(self):
        try:
            # Check for zip file first
            zipPath = os.path.join("./models", "legalbert_epoch4.zip")
            
            if os.path.exists(zipPath):
                if self._extract_model_from_zip(zipPath, settings.legal_bert_model_path):
                    logger.info("Model zip file found and extracted")
            
            # Try to load the actual model
            if os.path.exists(settings.legal_bert_model_path) and os.path.exists(os.path.join(settings.legal_bert_model_path, "config.json")):
                try:
                    import torch
                    import torch.nn.functional as F
                    from transformers import AutoTokenizer, AutoModelForSequenceClassification
                    
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    logger.info(f"Loading LegalBERT model from {settings.legal_bert_model_path}")
                    
                    self.tokenizer = AutoTokenizer.from_pretrained(settings.legal_bert_model_path)
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        settings.legal_bert_model_path
                    ).to(self.device)
                    
                    logger.info(f"LegalBERT model loaded successfully on {self.device}")
                    
                except ImportError:
                    logger.warning("torch/transformers not installed - using placeholder mode")
                except Exception as e:
                    logger.error(f"Failed to load actual model: {str(e)}")
            else:
                logger.warning(f"LegalBERT model files not found in: {settings.legal_bert_model_path}")
                logger.info("Place your legalbert_epoch4.zip in ./models/ or model files directly in ./models/legalbert_model/")
                
        except Exception as e:
            logger.error(f"Failed to initialize LegalBERT service: {str(e)}")
    
    def predictVerdict(self, inputText: str) -> str:
        if not self.is_model_loaded():
            logger.info("Using placeholder verdict prediction")
            textHash = int(hashlib.md5(inputText.encode()).hexdigest(), 16)
            return "guilty" if textHash % 2 == 1 else "not guilty"
        
        try:
            import torch
            import torch.nn.functional as F
            
            inputs = self.tokenizer(
                inputText, 
                return_tensors="pt", 
                truncation=True, 
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probabilities = F.softmax(logits, dim=1)
                predictedLabel = torch.argmax(probabilities, dim=1).item()
            
            return "guilty" if predictedLabel == 1 else "not guilty"
            
        except Exception as e:
            logger.error(f"Error predicting verdict: {str(e)}")
            return "not guilty"
    
    def getConfidence(self, inputText: str) -> float:
        if not self.is_model_loaded():
            logger.info("Using placeholder confidence score")
            textHash = int(hashlib.md5(inputText.encode()).hexdigest(), 16)
            return 0.5 + (textHash % 100) / 200.0
        
        try:
            import torch
            import torch.nn.functional as F
            
            inputs = self.tokenizer(
                inputText, 
                return_tensors="pt", 
                truncation=True, 
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probabilities = F.softmax(logits, dim=1)
            
            return float(torch.max(probabilities).item())
            
        except Exception as e:
            logger.error(f"Error getting confidence: {str(e)}")
            return 0.5
    
    def is_model_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None
    
    def get_device(self) -> str:
        return str(self.device)
    
    def is_healthy(self) -> bool:
        return True
