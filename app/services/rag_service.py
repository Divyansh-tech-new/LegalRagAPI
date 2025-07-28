import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Tuple
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.encoder = None
        self.preloadedIndexes = {}
        self._initialize_encoder()
        self._load_indexes()
    
    def _initialize_encoder(self):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence transformer: {settings.sentence_transformer_model}")
            self.encoder = SentenceTransformer(settings.sentence_transformer_model)
            logger.info("Sentence transformer loaded successfully")
        except ImportError:
            logger.warning("sentence-transformers not installed - using placeholder mode")
            self.encoder = "placeholder"
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {str(e)}")
            self.encoder = "placeholder"
    
    def loadFaissIndexAndChunks(self, indexPath: str, chunkPath: str) -> Tuple[Any, List]:
        try:
            if not os.path.exists(indexPath) or not os.path.exists(chunkPath):
                logger.warning(f"Missing files: {indexPath} or {chunkPath}")
                return None, []
            
            try:
                import faiss
                index = faiss.read_index(indexPath)
            except ImportError:
                logger.warning("faiss-cpu not installed - returning placeholder")
                return "placeholder_index", []
            
            if chunkPath.endswith('.pkl'):
                with open(chunkPath, 'rb') as f:
                    chunks = pickle.load(f)
            else:
                with open(chunkPath, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
            
            logger.info(f"Loaded index from {indexPath} with {len(chunks)} chunks")
            return index, chunks
        except Exception as e:
            logger.error(f"Failed to load index {indexPath}: {str(e)}")
            return None, []
    
    def _load_indexes(self):
        basePath = settings.faiss_indexes_base_path
        self.preloadedIndexes = {
            # "constitution": self.loadFaissIndexAndChunks(f"{basePath}/constitution_bgeLarge.index", f"{basePath}/constitution_chunks.json"),
            "ipcSections": self.loadFaissIndexAndChunks(f"{basePath}/ipc_bgeLarge.index", f"{basePath}/ipc_chunks.json"),
            "ipcCase": self.loadFaissIndexAndChunks(f"{basePath}/ipc_case_flat.index", f"{basePath}/ipc_case_chunks.json"),
            # "statutes": self.loadFaissIndexAndChunks(f"{basePath}/statute_index.faiss", f"{basePath}/statute_chunks.pkl"),
            "qaTexts": self.loadFaissIndexAndChunks(f"{basePath}/qa_faiss_index.idx", f"{basePath}/qa_text_chunks.json"),
            "caseLaw": self.loadFaissIndexAndChunks(f"{basePath}/case_faiss.index", f"{basePath}/case_chunks.pkl")
        }
        
        # Remove failed loads
        self.preloadedIndexes = {k: v for k, v in self.preloadedIndexes.items() if v[0] is not None}
        logger.info(f"Successfully loaded {len(self.preloadedIndexes)} indexes")
    
    def search(self, index: Any, chunks: List, queryEmbedding, topK: int) -> List[Tuple[float, Any]]:
        try:
            if index == "placeholder_index":
                return [(0.5, chunk) for chunk in chunks[:topK]]
            
            import faiss
            D, I = index.search(queryEmbedding, topK)
            results = []
            for score, idx in zip(D[0], I[0]):
                if idx < len(chunks):
                    results.append((score, chunks[idx]))
            return results
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
    
    def retrieveSupportChunksParallel(self, inputText: str) -> Tuple[Dict[str, List], Dict]:
        if self.encoder == "placeholder":
            logger.info("Using placeholder RAG retrieval")
            logs = {"query": inputText}
            support = {}
            for name in ["constitution", "ipcSections", "ipcCase", "statutes", "qaTexts", "caseLaw"]:
                if name in self.preloadedIndexes:
                    _, chunks = self.preloadedIndexes[name]
                    support[name] = chunks[:5] if chunks else []
                else:
                    support[name] = []
            logs["supportChunksUsed"] = support
            return support, logs
        
        try:
            import faiss
            queryEmbedding = self.encoder.encode([inputText], normalize_embeddings=True).astype('float32')
            faiss.normalize_L2(queryEmbedding)
            
            logs = {"query": inputText}
            
            def retrieve(name):
                if name not in self.preloadedIndexes:
                    return name, []
                idx, chunks = self.preloadedIndexes[name]
                results = self.search(idx, chunks, queryEmbedding, 5)
                return name, [c[1] for c in results]
            
            support = {}
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = [executor.submit(retrieve, name) for name in self.preloadedIndexes.keys()]
                for f in futures:
                    name, topChunks = f.result()
                    support[name] = topChunks
            
            logs["supportChunksUsed"] = support
            return support, logs
            
        except Exception as e:
            logger.error(f"Error retrieving support chunks: {str(e)}")
            raise ValueError(f"Support chunk retrieval failed: {str(e)}")
    
    def retrieveDualSupportChunks(self, inputText: str, geminiQueryModel):
        try:
            geminiQuery = geminiQueryModel.generateSearchQueryFromCase(inputText, geminiQueryModel)
        except:
            geminiQuery = None

        supportFromCase, _ = self.retrieveSupportChunksParallel(inputText)
        supportFromQuery, _ = self.retrieveSupportChunksParallel(geminiQuery or inputText)

        combinedSupport = {}
        for key in supportFromCase:
            combined = supportFromCase[key] + supportFromQuery[key]
            seen = set()
            unique = []
            for chunk in combined:
                if isinstance(chunk, str):
                    rep = chunk
                else:
                    rep = chunk.get("text") or chunk.get("description") or chunk.get("section_desc") or str(chunk)
                if rep not in seen:
                    seen.add(rep)
                    unique.append(chunk)
                if len(unique) == 10:
                    break
            combinedSupport[key] = unique

        return combinedSupport, geminiQuery
    
    def areIndexesLoaded(self) -> bool:
        return len(self.preloadedIndexes) > 0
    
    def getLoadedIndexes(self) -> List[str]:
        return list(self.preloadedIndexes.keys())
    
    def is_healthy(self) -> bool:
        return self.encoder is not None
