# 💊 DRUGVISTA

**AI co-pilot for molecular, clinical, and market intelligence.**

Built for **AWS ImpactX Challenge Finals @ IIT Bombay** — A working GenAI prototype demonstrating RAG, multi-step reasoning, and decision-oriented outputs in healthcare/pharma.

---

## 🎯 What This Project Does

DRUGVISTA is an intelligent analysis system that helps pharmaceutical researchers, investors, and decision-makers evaluate:

- **Molecules** (e.g., "Semaglutide", "Lecanemab")
- **Diseases** (e.g., "Alzheimer's disease", "Type 2 Diabetes")
- **Research Abstracts** (paste any biomedical abstract for analysis)

The system provides:
1. **Clinical Viability Assessment** — Efficacy, safety, regulatory status
2. **Risk Analysis** — Safety concerns, trial failures, contraindications
3. **Market Intelligence** — Commercial potential, competitive landscape
4. **Actionable Recommendation** — Proceed / Investigate Further / Drop

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DRUGVISTA ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐         ┌──────────────────────────────────────────────┐  │
│   │             │         │              BACKEND (FastAPI)               │  │
│   │  FRONTEND   │  HTTP   │                                              │  │
│   │ (Streamlit) │◄───────►│  ┌──────────┐    ┌─────────────────────────┐ │  │
│   │             │         │  │ /analyze │───►│    RAG PIPELINE         │ │  │
│   │  - Query    │         │  └──────────┘    │                         │ │  │
│   │    Input    │         │                  │  1. Query → Embedding   │ │  │
│   │  - Results  │         │                  │  2. Vector Search       │ │  │
│   │    Display  │         │                  │  3. Context Retrieval   │ │  │
│   │             │         │                  │  4. Multi-step LLM      │ │  │
│   └─────────────┘         │                  │     Reasoning           │ │  │
│                           │                  │  5. JSON Output         │ │  │
│   [AWS: Amplify/          │                  └─────────────────────────┘ │  │
│    CloudFront]            │                            │                 │  │
│                           │                  ┌─────────▼─────────┐       │  │
│                           │                  │   VECTOR STORE    │       │  │
│                           │                  │     (FAISS)       │       │  │
│                           │                  │   20 Documents    │       │  │
│                           │                  └───────────────────┘       │  │
│                           │  [AWS: Lambda + API Gateway]                 │  │
│                           └──────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        DATA LAYER                                   │   │
│   │  ┌─────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │   │
│   │  │  10 Papers  │  │ 5 Clinical      │  │ 5 Market News           │  │   │
│   │  │  (Abstracts)│  │ Trial Summaries │  │ Snippets                │  │   │
│   │  └─────────────┘  └─────────────────┘  └─────────────────────────┘  │   │
│   │  [AWS: S3 Bucket]                                                   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        LLM LAYER                                    │   │
│   │                                                                     │   │
│   │    ┌─────────────────────────────────────────────────────────┐      │   │
│   │    │              MULTI-STEP REASONING CHAIN                 │      │   │
│   │    │                                                         │      │   │
│   │    │  Step 1: Context Understanding                          │      │   │
│   │    │     └──► Parse & categorize retrieved documents         │      │   │
│   │    │                                                         │      │   │
│   │    │  Step 2: Clinical Reasoning                             │      │   │
│   │    │     └──► Analyze efficacy, safety, viability            │      │   │
│   │    │                                                         │      │   │
│   │    │  Step 3: Market Reasoning                               │      │   │
│   │    │     └──► Assess commercial potential                    │      │   │
│   │    │                                                         │      │   │
│   │    │  Step 4: Decision Synthesis                             │      │   │
│   │    │     └──► Generate structured recommendation             │      │   │
│   │    └─────────────────────────────────────────────────────────┘      │   │
│   │    [AWS: Amazon Bedrock - Claude/Titan]                             │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 How GenAI is Used

### 1. Retrieval-Augmented Generation (RAG)

```
User Query ──► Sentence Transformer ──► Query Embedding
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │  FAISS Index    │
                                    │  (Cosine Sim)   │
                                    └────────┬────────┘
                                              │
                                              ▼
                              Top-K Relevant Documents Retrieved
                                              │
                                              ▼
                                    Context for LLM Reasoning
```

- **Embeddings**: `all-MiniLM-L6-v2` (384 dimensions)
- **Vector Store**: FAISS with Inner Product (cosine similarity)
- **Retrieval**: Top-5 documents, filtered by category

### 2. Multi-Step Chain-of-Thought Reasoning

Unlike simple single-prompt systems, DRUGVISTA uses **4-step reasoning**:

| Step | Purpose | Output |
|------|---------|--------|
| **1. Context Understanding** | Parse retrieved docs, identify gaps | Structured summary |
| **2. Clinical Reasoning** | Analyze efficacy, safety, viability | Clinical assessment |
| **3. Market Reasoning** | Evaluate commercial potential | Market assessment |
| **4. Decision Synthesis** | Combine into recommendation | Structured JSON |

This approach:
- ✅ Reduces hallucination by grounding in evidence
- ✅ Provides transparent reasoning chain
- ✅ Enables nuanced multi-factor analysis
- ✅ Produces consistent, structured outputs

---

## 🚀 Quick Start (< 5 minutes)

### Prerequisites
- Python 3.9+
- OpenAI API key

### Step 1: Install Dependencies

```bash
cd Drugvista
pip install -r requirements.txt
```

### Step 2: Set API Key

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your-openai-api-key"
```

**Windows (CMD):**
```cmd
set OPENAI_API_KEY=your-openai-api-key
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### Step 3: Start Backend

```bash
cd backend
python main.py
```

Backend runs at `http://localhost:8000`

### Step 4: Start Frontend (new terminal)

```bash
cd frontend
streamlit run app.py
```

Frontend runs at `http://localhost:8501`

### Step 5: Try It!

Enter queries like:
- "Semaglutide for cardiovascular protection in diabetes"
- "CAR-T therapy for lymphoma"
- "Alzheimer's disease treatment with amyloid antibodies"

---

## 📁 Project Structure

```
drugvista/
├── backend/
│   ├── main.py              # FastAPI server + endpoints
│   ├── rag_pipeline.py      # Core RAG + reasoning logic
│   ├── prompts.py           # Multi-step prompt templates
│   ├── embeddings.py        # Sentence transformer embeddings
│   ├── vector_store.py      # FAISS vector store
│   ├── faiss_index.bin      # [Generated] FAISS index
│   └── documents_metadata.json  # [Generated] Doc metadata
├── frontend/
│   └── app.py               # Streamlit UI
├── data/
│   ├── papers/              # 10 biomedical abstracts
│   ├── clinical_trials/     # 5 trial summaries
│   └── market/              # 5 market intelligence snippets
├── requirements.txt
└── README.md
```

---

## 📊 Output Format

### Structured JSON Response

```json
{
  "clinical_viability": "High | Medium | Low",
  "key_evidence": ["paper_001", "trial_002"],
  "major_risks": ["ARIA events", "anticoagulant interaction"],
  "market_signal": "Strong | Moderate | Weak",
  "recommendation": "Proceed | Investigate Further | Drop",
  "confidence_score": 0.78,
  "explanation": "Based on Phase 3 trial data showing 24% reduction in cardiovascular events..."
}
```

### Human-Readable Explanation

A 3-5 sentence summary suitable for stakeholder presentations, citing specific evidence and highlighting key decision factors.

---

## ☁️ AWS ImpactX Alignment

| Local Component | AWS Production Equivalent |
|-----------------|---------------------------|
| FAISS vector store | Amazon OpenSearch Serverless |
| Sentence Transformers | Amazon Bedrock Embeddings / SageMaker |
| OpenAI GPT | Amazon Bedrock (Claude, Titan, Llama) |
| Text files in `/data` | Amazon S3 bucket |
| FastAPI server | AWS Lambda + API Gateway |
| Streamlit frontend | AWS Amplify / CloudFront + S3 |
| Pipeline orchestration | AWS Step Functions |

### AWS Architecture (Production)

```
CloudFront ──► API Gateway ──► Lambda (RAG Pipeline)
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              S3 (Docs)     OpenSearch (Vectors)  Bedrock (LLM)
```

---

## 🎖️ Judging Criteria Alignment

| Criterion | How DRUGVISTA Addresses It |
|-----------|---------------------------|
| **Innovation** | Multi-step reasoning chain, not single-prompt; RAG for grounding |
| **Technical Depth** | Real embeddings, vector search, structured prompting |
| **AWS Alignment** | Code structured for Bedrock/Lambda/S3/OpenSearch migration |
| **Impact** | Addresses real pharma decision-making pain points |
| **Completeness** | End-to-end working system, not mockup |
| **Demo Quality** | Clean UI, fast results, clear output format |

---

## 🔧 API Reference

### `POST /analyze`

Analyze a query for pharmaceutical intelligence.

**Request:**
```json
{
  "query": "Semaglutide cardiovascular outcomes",
  "use_chain_of_thought": true
}
```

**Response:** See Output Format above.

### `GET /health`

Health check endpoint.

### `GET /documents`

List all indexed documents.

### `GET /search?query=...&top_k=5`

Search documents without LLM analysis (for debugging).

---

## 🛠️ Development Notes

### Adding New Documents

1. Add `.txt` files to `data/papers/`, `data/clinical_trials/`, or `data/market/`
2. Include metadata header:
```
---
id: unique_id
type: paper | clinical_trial | market
topic: Disease or molecule name
---
Content here...
```
3. Delete `faiss_index.bin` and `documents_metadata.json`
4. Restart backend (index rebuilds automatically)

### Switching to AWS Bedrock

Replace OpenAI calls in `rag_pipeline.py`:

```python
import boto3

bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

response = bedrock.invoke_model(
    modelId='anthropic.claude-3-sonnet-20240229-v1:0',
    body=json.dumps({
        'anthropic_version': 'bedrock-2023-05-31',
        'max_tokens': 2048,
        'messages': [{'role': 'user', 'content': prompt}],
        'system': system_prompt
    })
)
```

---

## 👥 Team

Built for AWS ImpactX Challenge @ IIT Bombay

---

## 📜 License

MIT License — Free to use and modify.

