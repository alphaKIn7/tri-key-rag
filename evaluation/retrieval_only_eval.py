"""
Retrieval-only evaluation — no reranker, no generator.

Measures Recall@3: does the correct chunk appear in the top 3 retrieved chunks?
This isolates each method's retrieval quality and removes the reranker as a variable.
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunking.semantic_chunker import SemanticChunker
from enrichment.enricher import Enricher
from indexing.vector_index import VectorIndex
from indexing.bm25_index import BM25Index
from ranking.rrf_fusion import RRFFusion
from typing import List, Dict


def recall_at_k(chunks: List[Dict], key_phrases: List[str]) -> bool:
    for chunk in chunks:
        text = chunk["text"].lower()
        if all(phrase.lower() in text for phrase in key_phrases):
            return True
    return False


def retrieve_method1(query, vi, top_k=3):
    return vi.search(query, top_k=top_k)


def retrieve_method2(query, vi, bm25, rrf, top_k=3):
    dense  = vi.search(query, top_k=top_k)
    lexical = bm25.search(query, top_k=top_k)
    return rrf.fuse(dense, lexical, top_k=top_k)


def retrieve_method3(query, vi, si, bm25, rrf, top_k=3):
    dense   = vi.search(query, top_k=top_k)
    summary = si.search(query, top_k=top_k)
    for r in summary:
        if "original_text" in r:
            r["text"] = r["original_text"]
    lexical = bm25.search(query, top_k=top_k)
    return rrf.fuse(dense, summary, lexical, top_k=top_k)


def build_all(doc_path):
    with open(doc_path) as f:
        text = f.read()

    chunks = SemanticChunker(max_chunk_size=400, overlap=50).chunk_document(
        text, metadata={"source": "aviation_manual.txt"}
    )

    print("Enriching chunks for Method 3...")
    enriched = Enricher().enrich(chunks)

    vi   = VectorIndex(); vi.build(enriched)
    bm25 = BM25Index();   bm25.build(enriched)

    summary_chunks = []
    for c in enriched:
        s = c.copy(); s["original_text"] = c["text"]; s["text"] = c["summary"]
        summary_chunks.append(s)
    si = VectorIndex(); si.build(summary_chunks)

    rrf = RRFFusion(k=60)
    return vi, si, bm25, rrf


if __name__ == "__main__":
    vi, si, bm25, rrf = build_all("data/sample_docs/aviation_manual.txt")

    with open("data/eval_questions.json") as f:
        questions = json.load(f)

    results = {"Method1": [], "Method2": [], "Method3": []}

    for q in questions:
        r1 = retrieve_method1(q["question"], vi)
        r2 = retrieve_method2(q["question"], vi, bm25, rrf)
        r3 = retrieve_method3(q["question"], vi, si, bm25, rrf)

        results["Method1"].append({**q, "recall": recall_at_k(r1, q["key_phrases"]), "chunks": r1})
        results["Method2"].append({**q, "recall": recall_at_k(r2, q["key_phrases"]), "chunks": r2})
        results["Method3"].append({**q, "recall": recall_at_k(r3, q["key_phrases"]), "chunks": r3})

    # --- Print report ---
    methods = ["Method1", "Method2", "Method3"]
    labels  = {"Method1": "🟢 M1", "Method2": "🟡 M2", "Method3": "🔵 M3"}

    print("\n" + "=" * 65)
    print("  RETRIEVAL-ONLY EVALUATION  (no reranker, no generator)")
    print("  Metric: Recall@3 — correct chunk in top 3?")
    print("=" * 65)

    def pct(vals): return f"{sum(vals)/len(vals)*100:.0f}%"
    def diff(m_vals, base_vals):
        d = sum(m_vals) - sum(base_vals)
        return f"(+{d})" if d > 0 else f"({d})" if d < 0 else "(=)"

    m1r = [r["recall"] for r in results["Method1"]]
    m2r = [r["recall"] for r in results["Method2"]]
    m3r = [r["recall"] for r in results["Method3"]]

    print(f"\n{'Metric':<30} {'🟢 M1':>8} {'🟡 M2':>8} {'🔵 M3':>8}")
    print("-" * 58)
    print(f"{'Recall@3 (all 20 questions)':<30} {pct(m1r):>8} {pct(m2r):>8} {pct(m3r):>8}")

    for diff_label, key in [("  easy", "easy"), ("  medium", "medium"), ("  hard", "hard")]:
        d1 = [r["recall"] for r in results["Method1"] if r["difficulty"] == key]
        d2 = [r["recall"] for r in results["Method2"] if r["difficulty"] == key]
        d3 = [r["recall"] for r in results["Method3"] if r["difficulty"] == key]
        print(f"{'Recall@3 ' + diff_label:<30} {pct(d1):>8} {pct(d2):>8} {pct(d3):>8}")

    print(f"\n{'Per-question breakdown':}")
    print("-" * 65)
    print(f"{'Q':>3}  {'Diff':<8}  {'M1':>5}  {'M2':>5}  {'M3':>5}  Question")
    print("-" * 65)

    for i in range(len(questions)):
        r1 = results["Method1"][i]
        r2 = results["Method2"][i]
        r3 = results["Method3"][i]
        m  = lambda r: "  ✓  " if r["recall"] else "  ✗  "
        # flag where M3 beats M2
        flag = " ◄" if r3["recall"] and not r2["recall"] else ""
        print(f"Q{r1['id']:02d}  {r1['difficulty']:<8}  {m(r1)}  {m(r2)}  {m(r3)}  "
              f"{r1['question'][:40]}{flag}")

    print("\n" + "=" * 65)
    print("  KEY FINDING")
    print("=" * 65)
    m3_beats_m2 = sum(1 for i in range(len(questions))
                      if results["Method3"][i]["recall"] and not results["Method2"][i]["recall"])
    m2_beats_m3 = sum(1 for i in range(len(questions))
                      if results["Method2"][i]["recall"] and not results["Method3"][i]["recall"])

    print(f"\n  Questions where M3 retrieves correctly but M2 does not: {m3_beats_m2}")
    print(f"  Questions where M2 retrieves correctly but M3 does not: {m2_beats_m3}")
    print(f"\n  M1 total: {sum(m1r)}/20   M2 total: {sum(m2r)}/20   M3 total: {sum(m3r)}/20")
    if sum(m3r) > sum(m2r):
        print(f"\n  ✓ Intent key adds {sum(m3r) - sum(m2r)} retrieval wins over Method 2.")
    elif sum(m3r) == sum(m2r):
        print(f"\n  ~ Intent key matches Method 2 on retrieval — no gain, no loss.")
    else:
        print(f"\n  ✗ Intent key loses {sum(m2r) - sum(m3r)} retrieval questions vs Method 2.")
    print()
