# DermAI Pro — LangChain Multi-Agent Architecture

## Stack Overview

```
Language:    Python 3.11
Framework:   FastAPI + Uvicorn (async)
AI:          LangChain 0.3 (LCEL + ReAct Agents)
Models:      HuggingFace ViT + OpenAI GPT-4o
Embeddings:  sentence-transformers/all-MiniLM-L6-v2
Vector DB:   FAISS (persisted to disk)
Database:    PostgreSQL + SQLAlchemy Async
Dataset:     DermNet (10 conditions, 100+ routines)
Streaming:   SSE via FastAPI StreamingResponse
```

---

## Agent Architecture

```
Image Input
    │
    ▼
┌────────────────────────────────────────────────┐
│              ORCHESTRATOR                       │
│          (agents/orchestrator.py)               │
└──────┬────────────┬───────────────┬─────────────┘
       │            │               │
       ▼            ▼               ▼
  [Agent 1]    [Agent 2]       [Agent 3]
  Vision       RAG             Routine
  Classifier   Knowledge       Generator
  ─────────    ─────────       ─────────
  HF ViT       FAISS +         LangChain
  + GPT-4o     MiniLM          ReAct Agent
  Vision       embeddings      + 3 Tools
       │            │               │
       └────────────┴───────────────┘
                    │
                    ▼
              [PostgreSQL]
                    │
                    ▼
              [Agent 4]
              Chat Agent
              ─────────────────
              LangChain LCEL
              + ConvHistory
              + RAG per turn
              + SSE Streaming
```

---

## Agent 1 — Vision Classifier (`agents/vision_agent.py`)

### Strategy: Dual-model with fallback

```
Primary:  HuggingFace Inference API
          Model: Anwarkh1/Skin_Disease-Image_Classification
          Architecture: ViT (Vision Transformer)
          Fine-tuned on: HAM10000 / skin disease datasets
          Output: top-k classification with confidence scores

Fallback: OpenAI GPT-4o Vision
          Mode: high-detail image analysis
          Output: structured JSON (conditions, type, score, analysis)
```

### Output Schema
```python
{
  "conditions": ["Acne Vulgaris", "Hyperpigmentation"],
  "skin_type": "Oily",
  "overall_score": 68,          # 0-100 health score
  "confidence": 87.3,           # HF model confidence %
  "hf_classifications": [       # Raw HF predictions
    {"condition": "Acne", "confidence": 87.3},
    {"condition": "Rosacea", "confidence": 6.2}
  ],
  "summary": "Clinical narrative...",
  "detailed_analysis": "Detailed paragraph...",
  "recommendations": ["...", "..."],
  "severity": "moderate",       # mild|moderate|severe
  "urgency": "routine",         # routine|soon|urgent
  "source": "huggingface_vit"   # which model was used
}
```

---

## Agent 2 — RAG Knowledge Agent (`agents/rag_agent.py`)

### Architecture

```
Knowledge Sources:
  1. DermNet conditions (10 conditions × 3 chunk types = 30+ documents)
  2. Clinical research facts (8 evidence-based facts)
  Total: ~80 documents after recursive text splitting

Embedding Model:
  sentence-transformers/all-MiniLM-L6-v2
  - 384-dim dense embeddings
  - Normalized for cosine similarity
  - Runs on CPU efficiently

Vector Store:
  FAISS (Facebook AI Similarity Search)
  - Persisted to /tmp/dermai_faiss_index
  - IndexFlatL2 for exact search
  - Loaded from disk on restart

Retrieval Strategy:
  MMR (Maximal Marginal Relevance)
  - Balances relevance + diversity
  - fetch_k=15 candidates, return k=5
  - lambda_mult=0.6 (60% relevance, 40% diversity)

Multi-query:
  One retrieval per condition + combined query
  Deduplication by content prefix
  Max 8 unique documents returned
```

### Retrieval Flow
```
conditions: ["Acne", "Hyperpigmentation"]
    │
    ├─── query: "Acne" → FAISS MMR → 3 docs
    ├─── query: "Hyperpigmentation" → FAISS MMR → 3 docs
    └─── query: "Acne Hyperpigmentation" → FAISS MMR → 3 docs
         │
         └── Deduplicate by content hash
             │
             └── Format with source metadata
                 │
                 └── 3500-char context for LLM
```

### Fallback
When embeddings unavailable: `KeywordFallbackRetriever`
- TF-IDF-style term frequency scoring
- Weights longer terms higher (len > 6 → ×3)
- No ML dependency required

---

## Agent 3 — Routine Generator (`agents/routine_agent.py`)

### Architecture: LangChain ReAct Agent

```
LangChain ReAct (Reasoning + Acting):
  LLM: GPT-4o (temp=0.3 for consistency)
  Max iterations: 5
  Tools: 3 custom tools

Tool 1: check_ingredient_compatibility
  Input: comma-separated ingredients
  Logic: checks INCOMPATIBLE_PAIRS lookup table
  Detects: retinol+AHA, BP+retinol, vitC+retinoid, etc.

Tool 2: get_ingredient_info
  Input: ingredient name
  Logic: INGREDIENT_DATABASE lookup (6 key ingredients)
  Returns: mechanism, concentration, pH, best_for, avoid_if

Tool 3: validate_routine_order
  Input: JSON steps array
  Logic: enforces correct layering order
  Order: cleanser → toner → serum → treatment → moisturizer → SPF

ReAct Loop:
  Thought → Action → Observation → Thought → ... → Final Answer

Fallback: Direct LLM call (no ReAct loop)
```

### Output Schema
```python
{
  "morning": [
    {
      "order": 1,
      "step": "Cleanser",
      "product": "CeraVe Foaming Facial Cleanser",
      "active": "ceramides, niacinamide",
      "instruction": "Massage 60s, lukewarm water",
      "why": "Removes sebum without disrupting barrier"
    }
  ],
  "evening": [...],
  "weekly": [...],
  "key_ingredients": ["ceramides", "niacinamide"],
  "ingredient_warnings": ["Retinol + AHA: use alternate nights"],
  "expected_timeline": "8-12 weeks",
  "lifestyle_tips": ["Change pillowcase 2x/week"]
}
```

---

## Agent 4 — Chat Agent (`agents/chat_agent.py`)

### Architecture: LangChain LCEL + Streaming

```
Conversation Memory:
  In-memory sliding window (last 20 turns per diagnosis_id)
  Stored in: in_memory_histories dict
  Key: diagnosis_id (int)

Per-turn RAG:
  Agent 2 retrieves fresh context for each message
  Context: 2000 chars (shorter than diagnosis pipeline)

System Prompt:
  Patient profile (conditions, skin_type, score, severity)
  RAG context (retrieved at each turn)
  Clinical guidelines for response format

Streaming:
  LangChain astream() → async generator
  SSE format: data: {"content": "token"}\n\n
  Final: data: {"done": true, "sources": [...]}\n\n

DB Persistence:
  Saves user + assistant messages AFTER stream completes
  Stored in: dermai_chat table
```

---

## Data Flow — Full Diagnosis

```
POST /api/diagnoses
  │
  ├─1─ VisionAgent.analyze(image_b64)
  │      ├─ HF API call (ViT classifier)
  │      └─ GPT-4o Vision call
  │           └─ returns: conditions, type, score, analysis
  │
  ├─2─ RAGAgent.retrieve_for_conditions(conditions)
  │      ├─ Multi-query FAISS MMR search
  │      └─ returns: 8 documents, formatted context
  │
  ├─3─ RoutineAgent.generate(conditions, skin_type, rag_context)
  │      ├─ ReAct agent: check compatibility tool
  │      ├─ ReAct agent: validate routine order tool
  │      └─ returns: structured morning/evening/weekly routine
  │
  ├─4─ Save to PostgreSQL (DiagnosisModel)
  │      ├─ diagnosis record
  │      └─ welcome chat message
  │
  └─5─ Return JSON response with all fields
```

---

## API Reference

```
GET  /api/healthz                    → system health + agent status
GET  /api/dashboard                  → user stats, score trend, RAG status
POST /api/diagnoses                  → run full 4-agent pipeline
GET  /api/diagnoses                  → list user diagnoses
GET  /api/diagnoses/:id              → diagnosis + routine + chat history
DELETE /api/diagnoses/:id            → delete diagnosis
POST /api/diagnoses/:id/chat         → SSE streaming chat (Agent 4)
GET  /api/diagnoses/:id/chat         → chat history
DELETE /api/diagnoses/:id/chat       → clear chat session
GET  /api/conditions                 → all 10 DermNet conditions
GET  /api/conditions/search?q=...    → semantic RAG search
GET  /api/conditions/:id             → condition detail + full routine
GET  /api/auth/user                  → current user
POST /api/auth/demo                  → create demo session
```

---

## Environment Variables

```bash
# Required for full functionality:
OPENAI_API_KEY=sk-...              # GPT-4o Vision + Chat + Routine
OPENAI_API_BASE=https://...        # Optional: custom base URL

# For HuggingFace ViT classification:
HUGGINGFACE_API_KEY=hf_...         # Free inference API

# Database:
DATABASE_URL=postgresql://...      # PostgreSQL connection string

# Server:
PORT=8090                          # Default port
```

---

## Kaggle Dataset Integration

To use real HAM10000 or DermNet data:

```python
# 1. Download from Kaggle:
# kaggle datasets download -d shubhamgoel27/dermnet
# kaggle datasets download -d marmal88/skin_cancer (HAM10000)

# 2. Load into knowledge base:
from langchain_community.document_loaders import CSVLoader
loader = CSVLoader("ham10000_metadata.csv",
                   source_column="image_id",
                   metadata_columns=["dx", "dx_type", "age", "sex", "localization"])
docs = loader.load()

# 3. Add to FAISS index:
vectorstore.add_documents(docs)
vectorstore.save_local(FAISS_INDEX_PATH)
```

---

## Why These Choices?

| Decision | Choice | Reason |
|---|---|---|
| Vector DB | FAISS | No external service, runs in-memory, fast |
| Embeddings | MiniLM-L6-v2 | 384-dim, CPU-fast, strong semantic performance |
| RAG Strategy | MMR | Avoids redundant chunks, improves diversity |
| Agent Pattern | ReAct | Explainable, tool-augmented, debuggable |
| Streaming | SSE | Native HTTP, no WebSocket complexity |
| Async | Full async/await | FastAPI + SQLAlchemy async for scalability |
| Memory | Sliding window (20) | Cost-effective, recency-biased |
