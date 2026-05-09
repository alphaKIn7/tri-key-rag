# Three-Architecture RAG Comparison System — Implementation Plan

## Goal

Build **three RAG architectures** from simple to advanced, compare their retrieval accuracy, and prove why each improvement matters — all while learning AI engineering from the ground up.

| Method | Name | Signals | Your Idea? |
|--------|------|---------|------------|
| 🟢 Method 1 | Normal RAG | Dense embeddings only | Baseline |
| 🟡 Method 2 | Hybrid RAG | Dense + Lexical (TF-IDF/BM25) | Industry standard |
| 🔵 Method 3 | **Tri-Key RAG** | Dense + Lexical + Summary embeddings | ✅ Your innovation |

The key insight: **each method builds on the previous one**, so you'll see exactly what each addition contributes.

---

## The Three Architectures — Visual Comparison

### 🟢 Method 1 — Normal RAG (Dense Only)

```
Document → Chunks → Embeddings → FAISS Index
                                      ↓
                    Query → Embed → Vector Search → Top-K → LLM → Answer
```

**One signal**: Semantic similarity only.

### 🟡 Method 2 — Hybrid RAG (Lexical + Dense)

```
Document → Chunks → Embeddings → FAISS Index
                  → TF-IDF Vectors → BM25 Index
                                          ↓
                    Query → Vector Search + BM25 Search
                                          ↓
                              RRF Fusion → Re-rank → LLM → Answer
```

**Two signals**: Semantic similarity + keyword matching.

### 🔵 Method 3 — Tri-Key RAG (Your Idea)

```
Document → Chunks → Embeddings ──────────→ FAISS Index (semantic key)
                  → TF-IDF Vectors ──────→ BM25 Index (lexical key)
                  → LLM Summary → Embed → FAISS Index (intent key)
                                                ↓
                    Query → Vector Search + BM25 Search + Summary Search
                                                ↓
                              RRF Fusion → Re-rank → LLM → Answer
```

**Three signals**: Semantic + keyword + intent. Every chunk has three "keys" to be found by.

---

## Why Method 3 Is Clever (Theory Preview)

> [!TIP]
> **The vocabulary mismatch problem**: A user asks "How do I fix a slow database?" but the chunk says "Optimize SQL query performance using indexing." Dense search might catch this — but what if the chunk is buried in a long paragraph about database administration? The raw embedding gets diluted.
>
> **Method 3's solution**: The LLM summary distills the chunk to its core intent: "Techniques for improving database query speed." Now the summary embedding captures pure intent without noise. That's the **intent key**.

| Problem | Method 1 | Method 2 | Method 3 |
|---------|----------|----------|----------|
| User asks with different words than document | ⚠️ Sometimes works | ⚠️ Sometimes works | ✅ Summary captures intent |
| User asks with exact technical terms | ❌ Misses keywords | ✅ BM25 catches it | ✅ BM25 catches it |
| Long noisy chunk dilutes meaning | ❌ Embedding is noisy | ❌ Embedding still noisy | ✅ Summary is clean |
| User asks a question the chunk answers | ⚠️ Depends on phrasing | ⚠️ Depends on phrasing | ✅ Summary matches questions |

---

## Project Structure

```
/Volumes/MainDrive/RAG New Architecture/
│
├── chunking/                    # Step 1: Shared across all methods
│   ├── __init__.py
│   └── semantic_chunker.py
│
├── enrichment/                  # Step 2: Used by Method 3 only
│   ├── __init__.py
│   └── enricher.py             # LLM summary + key generation
│
├── indexing/                    # Step 3: All index types
│   ├── __init__.py
│   ├── vector_index.py         # FAISS index (Methods 1, 2, 3)
│   └── bm25_index.py           # BM25/TF-IDF index (Methods 2, 3)
│
├── retrieval/                   # Step 4: Retrieval per method
│   ├── __init__.py
│   └── retriever.py            # Dense, hybrid, and tri-key retrieval
│
├── ranking/                     # Steps 5 & 6: Fusion + re-ranking
│   ├── __init__.py
│   ├── rrf_fusion.py           # Reciprocal Rank Fusion (Methods 2, 3)
│   └── reranker.py             # Cross-encoder re-ranking (Methods 2, 3)
│
├── llm/                         # Steps 7 & 8: Query + compression
│   ├── __init__.py
│   ├── query_rewriter.py       # Query understanding
│   ├── context_compressor.py   # Context compression
│   └── generator.py            # Final answer generation
│
├── pipeline/                    # Step 9: Orchestration
│   ├── __init__.py
│   ├── method1_pipeline.py     # 🟢 Normal RAG pipeline
│   ├── method2_pipeline.py     # 🟡 Hybrid RAG pipeline
│   └── method3_pipeline.py     # 🔵 Tri-Key RAG pipeline
│
├── evaluation/                  # Step 10: Compare all three
│   ├── __init__.py
│   ├── evaluator.py            # Run all 3 methods, collect metrics
│   └── comparison_report.py    # Generate comparison tables/charts
│
├── observability/               # Logging across all methods
│   ├── __init__.py
│   └── logger.py
│
├── data/                        # Test documents + eval questions
│   ├── sample_docs/
│   └── eval_questions.json     # Questions with expected answers
│
├── config.py                    # Central configuration
├── requirements.txt             # All dependencies
├── main.py                      # Entry point — run any/all methods
└── README.md                    # Project documentation
```

### Why This Structure Matters

| Folder | Engineering Principle |
|--------|----------------------|
| `chunking/` | **Shared foundation** — All 3 methods use the same chunks (fair comparison) |
| `enrichment/` | **Method 3 exclusive** — Shows exactly what the extra LLM step adds |
| `indexing/` | **Abstraction** — Same interface, different index types |
| `retrieval/` | **Strategy pattern** — Swap retrieval strategies without changing other code |
| `ranking/` | **Composability** — Plug in/out fusion and re-ranking |
| `pipeline/` | **Separation** — Each method is its own pipeline, easy to compare |
| `evaluation/` | **Scientific rigor** — Can't claim "better" without measuring |

---

## Tech Stack

| Tool | Purpose | Used By |
|------|---------|---------|
| Python 3.10+ | Language | All methods |
| FAISS | Vector similarity search | Methods 1, 2, 3 |
| rank_bm25 | BM25 keyword search | Methods 2, 3 |
| sentence-transformers | Embedding generation | All methods |
| cross-encoder | Re-ranking candidates | Methods 2, 3 |
| OpenAI API (or Ollama) | Summaries, query rewrite, answers | Method 3 (enrichment), all (answers) |
| scikit-learn | TF-IDF vectorization | Methods 2, 3 |
| python-dotenv | API key management | All methods |
| matplotlib / rich | Results visualization | Evaluation |

---

## Step-by-Step Build Order

### Phase A — Shared Foundation (All Methods Need This)

#### Step 1 — Semantic Chunking
- **Theory**: Why naive chunking fails, sentence boundaries, overlap, metadata
- **Build**: `chunking/semantic_chunker.py`
- **Learn**: Chunking strategies, why chunk quality is the #1 factor in RAG
- **Output**: List of chunks with metadata

#### Step 2 — Vector Indexing (FAISS)
- **Theory**: What embeddings are, how FAISS works, cosine vs L2 distance
- **Build**: `indexing/vector_index.py`
- **Learn**: Embedding models, vector databases, similarity search

#### Step 3 — Basic LLM Answer Generation
- **Theory**: How LLMs use context, prompt engineering for RAG
- **Build**: `llm/generator.py`
- **Learn**: Prompt templates, context injection, temperature settings

---

### Phase B — 🟢 Method 1: Normal RAG

#### Step 4 — Build Method 1 Pipeline
- **Build**: `pipeline/method1_pipeline.py`
- **Flow**: Chunks → Embed → FAISS → Top-K → LLM → Answer
- **Test**: Run queries, see results, observe limitations
- **Learn**: Where dense-only retrieval fails

---

### Phase C — Add Lexical Search (Building Toward Method 2)

#### Step 5 — BM25 / TF-IDF Indexing
- **Theory**: How BM25 works, TF-IDF scoring, why keywords matter
- **Build**: `indexing/bm25_index.py`
- **Learn**: Sparse vs dense retrieval, when BM25 beats vectors

#### Step 6 — RRF Score Fusion
- **Theory**: Why you can't just merge scores, rank-based fusion
- **Build**: `ranking/rrf_fusion.py`
- **Learn**: Reciprocal Rank Fusion algorithm, the k parameter

#### Step 7 — Cross-Encoder Re-Ranking
- **Theory**: Bi-encoder vs cross-encoder, why re-ranking is essential
- **Build**: `ranking/reranker.py`
- **Learn**: Pointwise scoring, compute tradeoffs, cascade architecture

---

### Phase D — 🟡 Method 2: Hybrid RAG

#### Step 8 — Build Method 2 Pipeline
- **Build**: `pipeline/method2_pipeline.py`
- **Flow**: Chunks → Embed + TF-IDF → FAISS + BM25 → RRF → Re-rank → LLM → Answer
- **Test**: Run same queries, compare with Method 1
- **Learn**: How hybrid retrieval catches what dense misses

---

### Phase E — Add Enrichment (Building Toward Method 3)

#### Step 9 — Chunk Enrichment (Your Core Idea)
- **Theory**: Vocabulary mismatch, intent distillation, why summaries help
- **Build**: `enrichment/enricher.py`
- **Learn**: LLM-as-a-tool, structured output, prompt engineering

#### Step 10 — Query Understanding
- **Theory**: Why user queries are messy, multi-representation queries
- **Build**: `llm/query_rewriter.py`
- **Learn**: Query rewriting, keyword extraction, semantic expansion

#### Step 11 — Context Compression
- **Theory**: Lost-in-the-middle problem, context window management
- **Build**: `llm/context_compressor.py`
- **Learn**: Extractive vs abstractive compression

---

### Phase F — 🔵 Method 3: Tri-Key RAG

#### Step 12 — Build Method 3 Pipeline
- **Build**: `pipeline/method3_pipeline.py`
- **Flow**: Full tri-key retrieval with all enhancements
- **Test**: Run same queries, compare with Methods 1 & 2
- **Learn**: How three signals together outperform two

---

### Phase G — Evaluation & Comparison

#### Step 13 — Observability Layer
- **Build**: `observability/logger.py`
- **Learn**: Why you need to see inside the pipeline

#### Step 14 — Evaluation Framework
- **Build**: `evaluation/evaluator.py`, `evaluation/comparison_report.py`
- **Metrics**: Retrieval accuracy, answer quality, latency
- **Output**: Side-by-side comparison table + analysis
- **Learn**: How to evaluate RAG systems, what metrics matter

---

## Evaluation Framework

### What We'll Measure

| Metric | What It Tells Us | How We Measure |
|--------|-----------------|----------------|
| **Retrieval Recall@K** | Did the right chunks make it to top-K? | Check if ground-truth chunk is in retrieved set |
| **Retrieval Precision@K** | How many retrieved chunks were relevant? | Relevant chunks / total retrieved |
| **Answer Accuracy** | Did the LLM give the correct answer? | Manual check or LLM-as-judge |
| **Answer Faithfulness** | Is the answer grounded in the context? | Check for hallucination |
| **Latency** | How fast is each method? | Time per query |
| **Cost** | How expensive per query? | LLM API calls count |

### Comparison Table (What We'll Build)

| Metric | 🟢 Method 1 | 🟡 Method 2 | 🔵 Method 3 |
|--------|-------------|-------------|-------------|
| Recall@5 | ? | ? | ? |
| Precision@5 | ? | ? | ? |
| Answer Accuracy | ? | ? | ? |
| Faithfulness | ? | ? | ? |
| Avg Latency | ? | ? | ? |
| LLM Calls/Query | 1 | 1 | 2-3 |

### Test Dataset

We'll create **20-30 evaluation questions** across different difficulty levels:
- **Easy**: Answer is a direct quote from a chunk (favors BM25)
- **Medium**: Answer requires paraphrasing (favors dense search)
- **Hard**: Answer requires synthesizing across vocabulary gaps (favors Tri-Key)
- **Adversarial**: Misleading keyword overlap (tests re-ranking)

---

## Verification Plan

### Per-Step Testing
- Each module tested independently with sample data
- Print intermediate outputs to verify correctness

### Cross-Method Testing
- All 3 methods run on identical chunks and questions
- Side-by-side output comparison

### Final Evaluation
- Run full evaluation suite
- Generate comparison report with metrics table
- Analyze where each method wins and why

---

## Open Questions

> [!IMPORTANT]
> **OpenAI API Key**: Methods 2 & 3 enrichment + all methods' answer generation need an LLM. Do you have an OpenAI API key, or should we use a free local alternative (Ollama + Llama 3)?

> [!NOTE]
> **Sample Data**: I'll create test documents. Do you have a specific domain preference (e.g., technical docs, aviation, general knowledge)?

> [!NOTE]
> **Pace**: I'll go one step at a time, explain everything, and wait for your confirmation. Ready to start with Step 1?

---

## Teaching Approach

For every step, I will:
1. 📖 **Explain the theory** — What is this concept? Why does it matter?
2. 🔍 **Show the problem** — What goes wrong without this?
3. 🛠️ **Write small code** — Incremental, every line explained
4. 🧪 **Tell you what to run** — You execute and share output
5. 📊 **Compare** — After Methods 1, 2, 3 are built, we compare them head-to-head
6. ✅ **Confirm before moving on** — No rushing ahead
