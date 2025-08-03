# LegalLens-API: Legal Case Analysis & RAG Architecture

A robust backend API leveraging Dual-Stage Retrieval-Augmented Generation (RAG) and advanced Gemini 2.5 prompting for Indian legal case analysis. Combines LegalBERT predictions with relevant law/precedent retrieval and judge-style generative reasoning.

### 🔗 Hosted API
Access the running instance of LegalLens-API here:  
👉 [LegalLens-API on Hugging Face Spaces](https://huggingface.co/spaces/negi2725/LegalLens-API)

## 🧾 Example Input & Output

### ✅ Input (Case Summary)

The Petitioner, a small business owner, entered into a contract with the Respondent—a freelance developer—for the creation of an e-commerce website. The contract, signed on March 5, 2023, stipulated delivery by April 15, 2023, against an advance payment of ₹1,00,000. The Respondent repeatedly delayed delivery, eventually submitting an incomplete, non-functional version on May 10, 2023.

The Petitioner claims losses due to missed business opportunities and reputational harm, alleging criminal wrongdoing under IPC Sections 420 (cheating) and 406 (criminal breach of trust). Legal action is sought for refund, damages, and criminal conviction.

Query by tags for 1st retrival : IPC 420, IPC 406, cheating, criminal breach of trust, dishonest intention, fraud

### ✅ Output(Verdict)
```
{
  "verdict": "guilty",
  "confidence": 0.6841,
  "geminiVerdict": "not guilty",
  "verdictChanged": "Yes",
  "reasoning": "While the Respondent failed to deliver as per the contractual terms, the evidence does not support a finding of dishonest or fraudulent intent at the time the agreement was made. As highlighted in G. Sagar Suri v. State of U.P., mere failure to fulfill a contract or repay money is insufficient to establish criminal liability unless 'mens rea' (guilty mind) is proven at inception.

The dispute, as presented, is civil in nature—pertaining to breach of contract and deficiency in service—best addressed through civil remedies such as refund or compensation under the Consumer Protection Act or Contract Act. No criminal offense under IPC 420 or 406 is made out."
}
```
Final Verdict: Not Guilty  
Verdict Changed: Yes

Reason: Absence of mens rea at the inception of the agreement and the availability of civil remedies make this a contractual dispute rather than a criminal offense under IPC 420/406.



---

## 🔗 **Pipeline Overview**

1. **LegalBERT Prediction**
   - `verdict = predictVerdict(caseText)`
   - `confidence = getConfidence(caseText)`

2. **🔍 Search Query Generation (Gemini 2.5)**
   - **Input:** Raw case text
   - **Output:** Optimized legal search query  
     *(e.g., "IPC 420, breach of trust, absence of intent")*

3. **🧭 Dual-Stage Retrieval (Parallel RAG)**
   - **Both below run on same FAISS-indexed legal corpus:**
     - **RAG #1 (Gemini Query):** Top 5 chunks/domain using Gemini's generated legal keywords.
     - **RAG #2 (Original Case Text):** Top 5 chunks/domain using the raw input.
   - **Merge:** Deduplicate results, select top 10 support chunks/domain.

4. **⚖️ Verdict Evaluation (Gemini 2.5)**
   - **If Confidence ≥ 60%:**  
     Gemini justifies LegalBERT's verdict using support chunks.
   - **If Confidence < 60%:**  
     Gemini fully re-evaluates the case based on law & precedent.
   - **Prompt Components for Gemini:**
     - Full case text
     - Model's prediction + confidence
     - Top 10 retrieved legal chunks
     - Gemini’s generated search query (if present)
   - **Gemini Returns:**  
     Legal reasoning, final verdict, and verdict change status.

5. **📤 Final Sample Output (JSON)**
{
"verdict": "not guilty",
"confidence": 0.71,
"geminiVerdict": "not guilty",
"verdictChanged": "No",
"reasoning": "Given the absence of mens rea and legal precedent ABC vs State..."
}

text

---

## 🧠 **Gemini Prompting Strategy**

- **Role:** Act like an Indian judge.
- **References:** IPC, Constitution, statutes, precedent.
- **Formatting:** Final lines in prompt:
Final Verdict: Guilty or Not Guilty
Verdict Changed: Yes or No

text

---

## 🗂️ **File/Code Structure**

LegalLens-API/
├── app/
│ ├── services/ # RAG + Gemini logic
│ ├── models/ # LegalBERT model loading
│ ├── api/ # FastAPI route handlers
├── faiss_indexes/ # FAISS chunk indexes (.faiss, .json, .pkl)
├── main.py # Entrypoint / startup
├── raggy.ipynb # Full pipeline Colab/Jupyter notebook
└── README.md # Documentation (this file)

text

---

## 🧪 **How to Run**

### 🔌 Install Dependencies

pip install -r requirements.txt

text

### ⚡ Use LegalBERT

verdict = predictVerdict(caseText)
confidence = getConfidence(caseText)

text

### 🔎 Run Dual RAG + Gemini

logs = evaluateCaseWithGemini(
inputText=caseText,
modelVerdict=verdict,
confidence=confidence,
retrieveFn=retrieveSupportChunksParallel,
geminiQueryModel=model
)

text

---

## 📊 **Evaluation Modes**

| Metric             | Description                                    |
|--------------------|------------------------------------------------|
| Verdict Accuracy   | Whether Gemini's final verdict is correct      |
| Verdict Stability  | Change in verdict pre/post Gemini evaluation   |
| Explanation Depth  | Manual analysis of Gemini's legal reasoning    |

---

## 📌 **Notable Insights**

- Gemini 2.5 **excels** at generating legal search queries from raw case text.
- **Dual-Stage RAG** increases factual/legal correctness versus single retrieval.
- **Confidence-based gating** improves Gemini’s conservativeness when model is uncertain.

---

## 🌐 **Future Roadmap**

- Add **NyayaAnumana**: Dataset for local court verdicts.
- Enable **Gemini's long-context mode** (for lengthy case files).
- Retrieval from **scanned PDFs using OCR**.
- Add explainability for *why* a chunk was retrieved per verdict.

---

## 🧾 **Example Output (YAML)**

Final Verdict: Not Guilty
Verdict Changed: No
Reason: The retrieved IPC sections and prior judgments do not establish criminal intent conclusively.

text

---

## 📝 **Let Me Know If You Want:**

- a **condensed version**
- a **Hindi/local language version**
- a **LaTeX version** for academic publication
- a **Hugging Face Spaces**/**GitHub Pages**-formatted version

---

*For more details, see code comments and docstrings within each module!*
# ⚖️ LegalLens-API: Judge-as-a-Service using Gemini 2.5 + LegalBERT

> 🔍 Verdict Prediction + Dual RAG + Verdict Justification for Indian Supreme Court Cases

---

## 🚀 Overview

LegalLens-API is a **hybrid legal reasoning engine** that simulates how a judge reads, retrieves, and rules. It uses a **fine-tuned LegalBERT** classifier to predict verdicts, and **Gemini 2.5 Flash** to either **defend or override** them using a **two-level FAISS-based retrieval system**.

---

## 📊 Data Used

| Dataset | Description |
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

python

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
ConditionGemini Behavior
Confidence ≥ 60%Justify LegalBERT’s verdict using legal chunks
Confidence < 60%Re-evaluate case using laws + precedent

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
│ ├── services/ # RAG + Gemini handlers
│ ├── models/ # LegalBERT loading
│ ├── api/ # FastAPI routes
├── faiss_indexes/ # Chunk indexes (.faiss, .json, .pkl)
├── main.py # Entrypoint
├── raggy.ipynb # Full Colab notebook
└── README.md # This file
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
MetricDescription
Verdict AccuracyBased on Gemini's final verdict
Verdict StabilityWhether Gemini changes prediction
Explanation DepthManual judgment of legal reasoning richness

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

```
all in .md format
# ⚖️ LegalLens-API: Judge-as-a-Service using Gemini 2.5 + LegalBERT > 🔍 Verdict Prediction + Dual RAG + Verdict Justification for Indian Supreme Court Cases --- ## 🚀 Overview LegalLens-API is a **hybrid legal reasoning engine** that simulates how a judge reads, retrieves, and rules. It uses a **fine-tuned LegalBERT** classifier to predict verdicts, and **Gemini 2.5 Flash** to either **defend or override** them using a **two-level FAISS-based retrieval system**. --- ## 📊 Data Used | Dataset | Description | |----------------------|-------------| | **Fine-tuning Data** | 12,000 manually labeled Supreme Court judgments (ILDC format) for `guilty` / `not guilty` classification | | **RAG Retrieval Base** | ~5,000 chunks across 6 domains: Constitution, IPC sections, IPC case laws, Statutes, QA-style texts, and General Case Law | | **Embedding Model** | [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) for chunk embedding | | **Indexing Strategy** | FAISS with cosine similarity, L2-normalized dense vectors (not HNSW due to small size) | --- ## 🧬 How It Works (Pipeline) ### 1. 🧠 Verdict Prediction via LegalBERT - Model: Fine-tuned LegalBERT (epoch=4) - Output: `"guilty"` or `"not guilty"` + confidence score (softmax) ```python verdict = predictVerdict(caseText) confidence = getConfidence(caseText) 2. 🔍 Search Query Generation using Gemini 2.5 Input: Case text Output: Legal search query (e.g., IPC 420, breach of trust, absence of intent) 3. 🧭 Dual-Stage Retrieval (Parallel RAG) Both run in parallel over same FAISS-indexed corpus: ✅ RAG #1 (Gemini Query) Retrieves top 5 chunks per domain using Gemini’s legal keywords. ✅ RAG #2 (Original Case Text) Retrieves top 5 chunks per domain using raw case text. 🔄 Merge Strategy Combined → deduplicated → top 10 support chunks per domain selected. 4. ⚖️ Verdict Evaluation by Gemini 2.5 ConditionGemini Behavior Confidence ≥ 60%Justify LegalBERT’s verdict using legal chunks Confidence < 60%Re-evaluate case using laws + precedent Gemini Prompt Includes: Case text Model’s prediction + confidence Top 10 retrieved chunks Gemini’s search query (if available) Gemini Responds With: Legal analysis like a judge Final verdict Verdict change status 5. 📤 Final Output (Sample) json Copy Edit { "verdict": "not guilty", "confidence": 0.71, "geminiVerdict": "not guilty", "verdictChanged": "No", "reasoning": "Given the absence of mens rea and legal precedent ABC vs State..." } 🧠 Gemini Prompting Strategy Role: Act like a judge Consider: IPC, Constitution, precedent, statutes Include final lines: yaml Copy Edit Final Verdict: Guilty or Not Guilty Verdict Changed: Yes or No 🗂️ File/Code Structure bash Copy Edit LegalLens-API/ ├── app/ │ ├── services/ # RAG + Gemini handlers │ ├── models/ # LegalBERT loading │ ├── api/ # FastAPI routes ├── faiss_indexes/ # Chunk indexes (.faiss, .json, .pkl) ├── main.py # Entrypoint ├── raggy.ipynb # Full Colab notebook └── README.md # This file 🧪 How to Run 🔌 Install Dependencies bash Copy Edit pip install -r requirements.txt ⚡ Use LegalBERT python Copy Edit verdict = predictVerdict(caseText) confidence = getConfidence(caseText) 🔎 Run Dual RAG + Gemini python Copy Edit logs = evaluateCaseWithGemini( inputText=caseText, modelVerdict=verdict, confidence=confidence, retrieveFn=retrieveSupportChunksParallel, geminiQueryModel=model ) 📊 Evaluation Modes MetricDescription Verdict AccuracyBased on Gemini's final verdict Verdict StabilityWhether Gemini changes prediction Explanation DepthManual judgment of legal reasoning richness 📌 Notable Insights Gemini is highly accurate at generating legal search queries Dual RAG improves factual/legal correctness over single retrieval Confidence gating helps Gemini behave conservatively when uncertain 🌐 Future Roadmap Add NyayaAnumana dataset for local court decisions Enable Gemini's long-context mode for detailed case files Retrieval from scanned PDFs using OCR Explainable AI layer for why a chunk was retrieved 🧾 Example Output (YAML) yaml Copy Edit Final Verdict: Not Guilty Verdict Changed: No Reason: The retrieved IPC sections and prior judgments do not establish criminal intent conclusively. diff Copy Edit Let me know if you'd like: - a condensed version, - Hindi/localized version, - LaTeX version for academic use, - or a version styled for Hugging Face Spaces or GitHub Pages. all in .md format
text
# ⚖️ LegalLens-API: Judge-as-a-Service using Gemini 2.5 + LegalBERT

> 🔍 Verdict Prediction + Dual Retrieval-Augmented Generation (RAG) + Verdict Justification for Indian Supreme Court Cases

---

## 🚀 Overview

LegalLens-API is a **hybrid legal reasoning engine** simulating how a judge reads, retrieves, and rules. It employs a fine-tuned **LegalBERT** classifier to predict verdicts and **Gemini 2.5 Flash** to defend or override them using a two-level FAISS-based retrieval system.

---

## 📊 Data Used

| Dataset                 | Description                                                                                                    |
|-------------------------|----------------------------------------------------------------------------------------------------------------|
| **Fine-tuning Data**    | 12,000 manually labeled Supreme Court judgments (ILDC format) for `guilty` / `not guilty` classification       |
| **RAG Retrieval Base**  | ~5,000 chunks, 6 domains: Constitution, IPC, IPC case laws, Statutes, QA texts, General Case Law               |
| **Embedding Model**     | [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) for chunk embedding                   |
| **Indexing Strategy**   | FAISS, cosine similarity, L2-normalized dense vectors (not HNSW due to small size)                             |

---

## 🧬 How It Works (Pipeline)

### 1. 🧠 Verdict Prediction (LegalBERT)
- Model: Fine-tuned LegalBERT (epoch=4)
- Output: `"guilty"` or `"not guilty"` and confidence score (softmax)

verdict = predictVerdict(caseText)
confidence = getConfidence(caseText)

text

---

### 2. 🔍 Search Query Generation (Gemini 2.5)
- **Input:** Case text
- **Output:** Legal search query
  - _e.g.: "IPC 420, breach of trust, absence of intent"_

---

### 3. 🧭 Dual-Stage Retrieval (Parallel RAG)
- Both run in parallel over the same FAISS-indexed corpus:
  - **RAG #1 (Gemini Query):** Top 5 chunks/domain using Gemini’s legal keywords.
  - **RAG #2 (Original Case Text):** Top 5 chunks/domain using raw case text.
- **Merge:** Combine & deduplicate → Select top 10 supporting chunks/domain

---

### 4. ⚖️ Verdict Evaluation (Gemini 2.5)
- **Condition:**  
  - **Confidence ≥ 60%:** Gemini justifies LegalBERT’s verdict  
  - **Confidence < 60%:** Gemini fully re-evaluates using laws & precedent  
- **Gemini Prompt includes:**
  - Case text
  - Model’s prediction + confidence
  - Top 10 retrieved chunks
  - Gemini’s search query (if available)
- **Gemini Responds with:**
  - Legal analysis (judge-style reasoning)
  - Final verdict
  - Whether verdict changed

---

### 5. 📤 Final Output (Sample)

{
"verdict": "not guilty",
"confidence": 0.71,
"geminiVerdict": "not guilty",
"verdictChanged": "No",
"reasoning": "Given the absence of mens rea and legal precedent ABC vs State..."
}

text

---

## 🧠 Gemini Prompting Strategy

- **Role:** Act like an Indian judge
- **Consider:** IPC, Constitution, statutes, precedent
- **Prompt format:**
Final Verdict: Guilty or Not Guilty
Verdict Changed: Yes or No

text

---

## 🗂️ File/Code Structure

LegalLens-API/
├── app/
│ ├── services/ # RAG + Gemini handlers
│ ├── models/ # LegalBERT loading
│ ├── api/ # FastAPI routes
├── faiss_indexes/ # Chunk indexes (.faiss, .json, .pkl)
├── main.py # Entrypoint
├── raggy.ipynb # Full Colab notebook
└── README.md # This file

text

---

## 🧪 How To Run

### 🔌 Install Dependencies

pip install -r requirements.txt

text

### ⚡ Use LegalBERT

verdict = predictVerdict(caseText)
confidence = getConfidence(caseText)

text

### 🔎 Run Dual RAG + Gemini

logs = evaluateCaseWithGemini(
inputText=caseText,
modelVerdict=verdict,
confidence=confidence,
retrieveFn=retrieveSupportChunksParallel,
geminiQueryModel=model
)

text

---

## 📊 Evaluation Modes

| Metric             | Description                                 |
|--------------------|---------------------------------------------|
| Verdict Accuracy   | Consistency of Gemini’s final verdict        |
| Verdict Stability  | Whether Gemini changes the original verdict |
| Explanation Depth  | Manual assessment of legal reasoning        |

---

## 📌 Notable Insights

- Gemini is highly accurate at generating legal search queries.
- Dual RAG improves factual/legal correctness versus single retrieval.
- Confidence gating helps Gemini behave conservatively when uncertain.

---

## 🌐 Future Roadmap

- Add NyayaAnumana dataset for local court verdicts.
- Enable Gemini's long-context mode for detailed case files.
- Retrieval from scanned PDFs via OCR.
- Explainable AI layer: why a chunk was retrieved.

---

## 🧾 Example Output (YAML)

Final Verdict: Not Guilty
Verdict Changed: No

Reason: The retrieved IPC sections and prior judgments do not establish criminal intent conclusively.

text

---

## ✨ Variants Available

Let me know if you'd like:
- a **condensed version**,
- a **Hindi/localized version**,
- a **LaTeX version** for academic use,
- or a version styled for **Hugging Face Spaces** or **GitHub Pages**.

---

_For more details, see `raggy.ipynb` and in-code documentation._
