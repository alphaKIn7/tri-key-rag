# Tri-Key RAG — A Three-Architecture Comparison

A scientific comparison of three RAG (Retrieval-Augmented Generation) architectures, built from scratch as a learning project. The core contribution is **Tri-Key RAG** — an original retrieval architecture that adds a third retrieval signal (LLM-generated intent summaries) on top of the standard dense + keyword hybrid.

---

## What is RAG?

RAG stands for Retrieval-Augmented Generation. Instead of asking an AI to answer from memory, you:
1. **Retrieve** the most relevant passages from your documents
2. **Feed** those passages to an LLM as context
3. **Generate** a grounded answer based only on what was retrieved

This project builds and scientifically compares three progressively more sophisticated ways to do step 1.

---

## The Three Architectures

| Method | Name | Retrieval Signals |
|--------|------|-------------------|
| 🟢 Method 1 | Normal RAG | Dense embeddings only |
| 🟡 Method 2 | Hybrid RAG | Dense + BM25 keyword search |
| 🔵 Method 3 | **Tri-Key RAG** | Dense + BM25 + LLM summary embeddings |

### The Core Innovation — Tri-Key RAG

The key insight: a chunk can be found in three different ways.

```
Every chunk gets three "keys":

  Semantic key  →  embed the raw chunk text           → FAISS index 1
  Lexical key   →  BM25 keyword matching              → BM25 index
  Intent key    →  embed an LLM summary of the chunk  → FAISS index 2
```

The **intent key** solves the vocabulary mismatch problem: a user asks *"what happens when pilots get too tired"* but the document says *"Flight Duty Period limit"*. The LLM-generated summary writes the chunk's meaning in plain human language — closer to how a user would phrase the question — creating a bridge between the two.

At query time, all three indexes are searched and results are fused using Reciprocal Rank Fusion (RRF).

---

## Project Structure

```
├── chunking/              # Sentence-boundary chunker with overlap
├── enrichment/            # LLM summary generation (intent key)
├── indexing/              # FAISS vector index + BM25 index
├── ranking/               # RRF fusion + cross-encoder reranker
├── llm/                   # Answer generator
├── pipeline/              # method1_pipeline.py, method2_pipeline.py, method3_pipeline.py
├── evaluation/            # All evaluation scripts
│   ├── ragbench_eval.py          # Retrieval-only Recall@K
│   ├── ragbench_e2e_eval.py      # End-to-end with answer accuracy
│   ├── ragbench_grounded_eval.py # Grounded answer fusion experiment
│   └── debug_retrieval.py        # Per-signal retrieval inspector
└── data/
    ├── sample_docs/       # Aviation operations manual (prototype corpus)
    └── ragbench/          # vectara/open_ragbench subset (30 arXiv papers, 74 queries)
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| `sentence-transformers` (`all-MiniLM-L6-v2`) | Dense embeddings |
| `faiss-cpu` | Vector similarity search |
| `rank-bm25` | BM25 keyword search |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Re-ranking |
| `openai` (`gpt-4o-mini`) | Chunk summarisation + answer generation |
| `python-dotenv` | API key management |

---

## Evaluation Dataset

**vectara/open_ragbench** — a benchmark of 3,045 question-answer pairs over 1,000 arXiv papers.

Subset used: 30 papers → 7,294 chunks, 74 queries (40 abstractive, 34 extractive).

Ground truth: `qrels.json` maps each query to the exact `(doc_id, section_id)` that contains the answer. No key-phrase heuristics — pure chunk-level relevance judgements.

Query types:
- **Extractive**: answer uses exact words from the document (favours BM25)
- **Abstractive**: answer paraphrases the content (favours dense + intent key)

---

## Results

### Experiment 1 — Retrieval-only (Recall@K)

Does Method 3 retrieve the correct chunk more often?

| Metric | 🟢 M1 | 🟡 M2 | 🔵 M3 |
|--------|-------|-------|-------|
| Recall@5 — All | 89.2% | 93.2% | **95.9%** |
| Recall@5 — Abstractive | 87.5% | 92.5% | **95.0%** |
| Recall@5 — Extractive | 91.2% | 94.1% | **97.1%** |
| Recall@10 — All | 98.6% | 97.3% | **98.6%** |

**Finding:** Method 3 retrieves the correct chunk more reliably on both query types. The intent key rescues 2 queries that neither dense nor BM25 could find. M3 wins on 2 queries where M2 fails; M2 never beats M3.

---

### Experiment 2 — End-to-end answer accuracy (CONTEXT_K = 3)

Does better retrieval translate to better answers?

| Metric | 🟢 M1 | 🟡 M2 | 🔵 M3 |
|--------|-------|-------|-------|
| Answer Accuracy — All | 51.4% | 55.4% | 55.4% |

**Finding:** Better retrieval (M3) does not produce better answers at this context size. M2 and M3 are tied.

---

### Experiment 3 — End-to-end answer accuracy (CONTEXT_K = 5)

What if the generator sees more context?

| Metric | 🟢 M1 | 🟡 M2 | 🔵 M3 |
|--------|-------|-------|-------|
| Answer Accuracy — All | 64.9% | **70.3%** | **70.3%** |
| Answer Accuracy — Abstractive | 62.5% | **70.0%** | 65.0% |
| Answer Accuracy — Extractive | 67.6% | 70.6% | **76.5%** |

**Finding:** Increasing context from 3 to 5 chunks improved all methods by ~13 percentage points — a larger gain than any retrieval strategy change. M2 and M3 tied overall, but with an interesting split: M3 outperforms on extractive questions (+5.9pp) while M2 outperforms on abstractive (+5pp). The RRF noise from a third signal occasionally dilutes abstractive context but reinforces extractive results.

---

### Experiment 4 — Grounded Answer Fusion

**Hypothesis:** Each retrieval index generates its own answer independently. A synthesiser receives all three answers alongside their source chunks and verifies each claim against evidence before producing a final answer. This eliminates the "confidently wrong answer" failure mode.

```
Dense index   → top 5 chunks → Generator → Answer A + evidence
Summary index → top 5 chunks → Generator → Answer B + evidence
BM25 index    → top 5 chunks → Generator → Answer C + evidence
                                                    ↓
                        Synthesiser verifies each answer against its source
                                                    ↓
                                          Final grounded answer
```

| Configuration | Answer Accuracy | Wins over M3 | Losses to M3 |
|---------------|-----------------|--------------|--------------|
| gpt-4o-mini (strict prompt) | 66.2% | 6 | 9 |
| gpt-4o (strict prompt) | 50.0% | 3 | 18 |
| gpt-4o (softened prompt) | 67.6% | 7 | 9 |
| **M3 baseline** | **70.3%** | — | — |

**Finding:** Grounded synthesis did not outperform M3 in any configuration. The architecture has genuine value — 6–7 questions answered correctly that M3 missed — but introduces more errors than it fixes. Two causes identified:

1. **Synthesis noise**: Three intermediate generation steps accumulate more error than one
2. **Model-prompt mismatch**: GPT-4o followed the strict verification instruction literally, discarding unverifiable answers and collapsing to 50% accuracy. The softened prompt recovered to 67.6%

**Lesson:** Prompt design is model-specific. An instruction that produces reasonable behaviour in a weaker model can produce overly cautious behaviour in a stronger one.

---

## Key Findings

### 1. Method 3 improves retrieval
Recall@5 increased from 93.2% (M2) to 95.9% (M3) on a 7,294-chunk corpus. The intent key rescued 2 queries that neither dense embeddings nor BM25 could find. M3 never underperformed M2 on retrieval.

### 2. The retrieval-to-answer gap is large
M3 achieved 95.9% Recall@5 but only 70.3% answer accuracy — a 25-point gap. Even when the correct chunk is retrieved, the generator fails to produce a correct answer roughly 1 in 4 times. This gap is where most 2024–2025 RAG research is focused.

### 3. Context window size matters more than retrieval strategy
Going from CONTEXT_K=3 to CONTEXT_K=5 improved all methods by ~13pp. No single retrieval strategy improvement came close to that magnitude. Giving the generator more relevant context is more impactful than optimising which chunks rank first.

### 4. Complex pipelines don't outperform simple ones at this scale
A 4-call grounded synthesis pipeline (3 generators + 1 synthesiser) consistently underperformed a 1-call M3 pipeline. More steps mean more accumulated error. On datasets of this size (74 queries), the simpler approach wins.

### 5. M3 splits on query type
Method 3 outperforms Method 2 on extractive questions (+5.9pp) but underperforms on abstractive questions (-5pp). The third RRF signal reinforces keyword-confident results but introduces noise in meaning-sensitive contexts.

---

## Related Work

| Technique | Relation to this project |
|-----------|--------------------------|
| **Contextual Retrieval** (Anthropic 2024) | Similar to Tri-Key: adds LLM-generated context to chunks at index time. Difference: prepends to chunk text (one embedding) vs. separate intent index |
| **RAPTOR** (Stanford 2024) | Summarises *clusters* of chunks into a tree. Tri-Key summarises *individual* chunks — different granularity and purpose |
| **Lost in the Middle** (Stanford 2023) | Explains why Recall@5 ≠ Answer Accuracy: LLMs ignore middle-position context. Directly relevant to the 25pp gap observed |
| **CRAG** (2024) | Evaluates retrieved documents before generation. Would address the noise problem observed in Experiment 4 |
| **Self-RAG** (2023) | LLM decides when to retrieve and critiques its own output. Addresses the same gap this project observed |

---

## How to Run

### Setup

```bash
git clone <repo>
cd RAG-New-Architecture
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add your OpenAI API key to .env
```

### Download the evaluation corpus

```bash
python data/download_ragbench.py  # downloads 30 arXiv papers + pre-generates summaries
```

### Run retrieval-only evaluation

```bash
python evaluation/ragbench_eval.py
```

### Run end-to-end evaluation

```bash
python evaluation/ragbench_e2e_eval.py
```

### Run grounded synthesis experiment

```bash
python evaluation/ragbench_grounded_eval.py
```

### Inspect per-signal retrieval

```bash
python evaluation/debug_retrieval.py  # shows what each index retrieves per query
```

---

## Design Decisions

**Why not use the query rewriter?** The query rewriter was built (`llm/query_rewriter.py`) but excluded from all pipeline comparisons. Including it only in Method 3 would make better performance attributable to query quality rather than the Tri-Key retrieval strategy. Scientific validity requires controlling variables.

**Why sentence-boundary chunking?** More sophisticated methods (embedding-based split detection, proposition chunking) exist but are held constant across all three methods. Chunking quality affects all methods equally — it is a controlled variable, not a comparison variable.

**Why Recall@K without a reranker?** The reranker is excluded from the retrieval-only evaluation to isolate each method's raw retrieval quality. Including it would conflate retrieval and reranking performance.

---

## Cost

All LLM calls use `gpt-4o-mini`. Total API cost for the full experimental pipeline (corpus enrichment + all evaluations):

| Step | Approximate Cost |
|------|-----------------|
| Corpus enrichment (7,294 chunks) | $0.63 |
| End-to-end evaluations (3 runs) | $0.15 |
| Grounded synthesis (3 variants) | $1.20 |
| **Total** | **~$2.00** |
