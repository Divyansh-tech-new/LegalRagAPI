# ⚖️ LegalLens-API: Judge-as-a-Service using Gemini 2.5 + LegalBERT

> 🔍 Verdict Prediction + Dual RAG + Verdict Justification for Indian Supreme Court Cases

---

## 🚀 Overview

LegalLens-API is a **hybrid legal reasoning engine** that simulates how a judge reads, retrieves, and rules. It uses a **fine-tuned LegalBERT** classifier to predict verdicts, and **Gemini 2.5 Flash** to either **defend or override** them using a **two-level FAISS-based retrieval system**.

---

## 📊 Data Used

| Dataset              | Description |
|----------------------|-------------|
| **Fine-tuning Data** | 12,000 manually labeled Supreme Court judgments (ILDC format) for `guilty` / `not guilty` classification |
| **RAG Retrieval Base** | ~5,000 chunks across 6 domains: Constitution, IPC sections, IPC case laws, Statutes, QA-style texts, and General Case Law |
| **Embedding Model** | [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) for chunk embedding |
| **Indexing Strategy** | FAISS with cosine similarity, L2-normalized dense vectors (not HNSW due to small size) |

---

## 🧬 How It Works (Pipeline)

### 1. 🧠 Verdict Prediction via LegalBERT

- Model: Fine-tuned LegalBERT (epoch=4)
- Output: `"guilty"` or `"not guilty"` + confidence score (softmax)

```python
verdict = predictVerdict(caseText)
confidence = getConfidence(caseText)
2. 🔍 Search Query Generation using Gemini 2.5
Input: Case text

Output: Legal search query (e.g., IPC 420, breach of trust, absence of intent)

3. 🧭 Dual-Stage Retrieval (Parallel RAG)
Both run in parallel over same FAISS-indexed corpus:

✅ RAG #1 (Gemini Query)
Retrieves top 5 chunks per domain using Gemini’s legal keywords.

✅ RAG #2 (Original Case Text)
Retrieves top 5 chunks per domain using raw case text.

🔄 Merge Strategy
Combined → deduplicated → top 10 support chunks per domain selected.

4. ⚖️ Verdict Evaluation by Gemini 2.5
Condition	Gemini Behavior
Confidence ≥ 60%	Justify LegalBERT’s verdict using legal chunks
Confidence < 60%	Re-evaluate case using laws + precedent

Gemini Prompt Includes:

Case text

Model’s prediction + confidence

Top 10 retrieved chunks

Gemini’s search query (if available)

Gemini Responds With:

Legal analysis like a judge

Final verdict

Verdict change status

5. 📤 Final Output (Sample)
json
Copy
Edit
{
  "verdict": "not guilty",
  "confidence": 0.71,
  "geminiVerdict": "not guilty",
  "verdictChanged": "No",
  "reasoning": "Given the absence of mens rea and legal precedent ABC vs State..."
}
🧠 Gemini Prompting Strategy
Role: Act like a judge

Consider: IPC, Constitution, precedent, statutes

Include final lines:

yaml
Copy
Edit
Final Verdict: Guilty or Not Guilty  
Verdict Changed: Yes or No
🗂️ File/Code Structure
bash
Copy
Edit
LegalLens-API/
├── app/
│   ├── services/               # RAG + Gemini handlers
│   ├── models/                 # LegalBERT loading
│   ├── api/                    # FastAPI routes
├── faiss_indexes/              # Chunk indexes (.faiss, .json, .pkl)
├── main.py                     # Entrypoint
├── raggy.ipynb                 # Full Colab notebook
└── README.md                   # This file
🧪 How to Run
🔌 Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
⚡ Use LegalBERT
python
Copy
Edit
verdict = predictVerdict(caseText)
confidence = getConfidence(caseText)
🔎 Run Dual RAG + Gemini
python
Copy
Edit
logs = evaluateCaseWithGemini(
  inputText=caseText,
  modelVerdict=verdict,
  confidence=confidence,
  retrieveFn=retrieveSupportChunksParallel,
  geminiQueryModel=model
)
📊 Evaluation Modes
Metric	Description
Verdict Accuracy	Based on Gemini's final verdict
Verdict Stability	Whether Gemini changes prediction
Explanation Depth	Manual judgment of legal reasoning richness

📌 Notable Insights
Gemini is highly accurate at generating legal search queries

Dual RAG improves factual/legal correctness over single retrieval

Confidence gating helps Gemini behave conservatively when uncertain

🌐 Future Roadmap
Add NyayaAnumana dataset for local court decisions

Enable Gemini's long-context mode for detailed case files

Retrieval from scanned PDFs using OCR

Explainable AI layer for why a chunk was retrieved

🧾 Example Output (YAML)
yaml
Copy
Edit
Final Verdict: Not Guilty  
Verdict Changed: No

Reason: The retrieved IPC sections and prior judgments do not establish criminal intent conclusively.
diff
Copy
Edit

Let me know if you'd like:
- a condensed version,
- Hindi/localized version,
- LaTeX version for academic use,
- or a version styled for Hugging Face Spaces or GitHub Pages.
