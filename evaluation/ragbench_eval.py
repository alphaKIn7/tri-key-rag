"""
Ragbench evaluation — no key phrases, no generator.

Ground truth: qrels.json maps each query to the exact (doc_id, section_id)
that contains the answer.

Recall@K = did any top-K retrieved chunk come from the correct section?

Reports overall Recall@5 and Recall@10, split by query type
(abstractive vs extractive) — this is where Method 3's intent key should show.
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indexing.vector_index import VectorIndex
from indexing.bm25_index import BM25Index
from ranking.rrf_fusion import RRFFusion
from typing import List, Dict


CACHE_DIR  = "data/cache"
INDEX_DIR  = os.path.join(CACHE_DIR, "ragbench_indexes")
TOP_K_LIST = [5, 10]


def is_correct(chunk: Dict, doc_id: str, section_id: int) -> bool:
    m = chunk["metadata"]
    return m.get("source") == doc_id and m.get("section_id") == section_id


def recall_at_k(chunks: List[Dict], doc_id: str, section_id: int, k: int) -> bool:
    return any(is_correct(c, doc_id, section_id) for c in chunks[:k])


def build_or_load_indexes(enriched_chunks: List[Dict]) -> tuple:
    os.makedirs(INDEX_DIR, exist_ok=True)

    vi_path = os.path.join(INDEX_DIR, "dense")
    si_path = os.path.join(INDEX_DIR, "summary")

    vi   = VectorIndex()
    si   = VectorIndex()
    bm25 = BM25Index()

    if os.path.exists(vi_path + ".faiss"):
        print("Loading saved dense index...")
        vi.load(vi_path)
    else:
        print("Building dense index (raw text)...")
        vi.build(enriched_chunks)
        vi.save(vi_path)

    if os.path.exists(si_path + ".faiss"):
        print("Loading saved summary index...")
        si.load(si_path)
    else:
        print("Building summary index (intent key)...")
        summary_chunks = []
        for c in enriched_chunks:
            s = c.copy()
            s["original_text"] = c["text"]
            s["text"] = c["summary"]
            summary_chunks.append(s)
        si.build(summary_chunks)
        si.save(si_path)

    print("Building BM25 index...")
    bm25.build(enriched_chunks)

    return vi, si, bm25


def retrieve(method: str, query: str, vi, si, bm25, rrf, top_k: int) -> List[Dict]:
    if method == "Method1":
        return vi.search(query, top_k=top_k)

    dense   = vi.search(query, top_k=top_k)
    lexical = bm25.search(query, top_k=top_k)

    if method == "Method2":
        return rrf.fuse(dense, lexical, top_k=top_k)

    # Method3 — restore original text on summary results
    summary = si.search(query, top_k=top_k)
    for r in summary:
        if "original_text" in r:
            r["text"] = r["original_text"]
    return rrf.fuse(dense, summary, lexical, top_k=top_k)


def run():
    print("Loading data...")
    with open("data/cache/ragbench_enriched.json") as f:
        enriched_chunks = json.load(f)
    with open("data/ragbench/queries.json") as f:
        queries = json.load(f)
    with open("data/ragbench/qrels.json") as f:
        qrels = json.load(f)

    print(f"Corpus: {len(enriched_chunks)} chunks | Queries: {len(queries)}\n")

    vi, si, bm25 = build_or_load_indexes(enriched_chunks)
    rrf = RRFFusion(k=60)

    methods  = ["Method1", "Method2", "Method3"]
    results  = {m: [] for m in methods}
    max_k    = max(TOP_K_LIST)

    print(f"\nEvaluating {len(queries)} queries across 3 methods...\n")

    for qid, q in queries.items():
        qrel       = qrels[qid]
        doc_id     = qrel["doc_id"]
        section_id = qrel["section_id"]

        for method in methods:
            chunks = retrieve(method, q["query"], vi, si, bm25, rrf, top_k=max_k)
            results[method].append({
                "qid":        qid,
                "query":      q["query"],
                "type":       q["type"],
                "doc_id":     doc_id,
                "section_id": section_id,
                **{f"recall@{k}": recall_at_k(chunks, doc_id, section_id, k) for k in TOP_K_LIST},
            })

    _print_report(results)


def _print_report(results: Dict):
    methods = ["Method1", "Method2", "Method3"]

    def pct(vals): return f"{sum(vals)/len(vals)*100:.1f}%" if vals else "—"
    def by_type(method, q_type, key):
        return [r[key] for r in results[method] if r["type"] == q_type]

    print("\n" + "=" * 68)
    print("  RAGBENCH EVALUATION — Recall@K  (7,294 chunks, ground-truth qrels)")
    print("=" * 68)

    for k in TOP_K_LIST:
        key = f"recall@{k}"
        print(f"\n{'Recall@'+str(k):<30} {'🟢 M1':>10} {'🟡 M2':>10} {'🔵 M3':>10}")
        print("-" * 63)

        rows = [
            ("All queries",   [[r[key] for r in results[m]] for m in methods]),
            ("  Abstractive", [by_type(m, "abstractive", key) for m in methods]),
            ("  Extractive",  [by_type(m, "extractive",  key) for m in methods]),
        ]
        for label, row in rows:
            print(f"{label:<30} {pct(row[0]):>10} {pct(row[1]):>10} {pct(row[2]):>10}")

    # where does M3 beat M2?
    print(f"\n\n{'Per-query: M3 vs M2 differences (Recall@5)'}")
    print("-" * 68)
    m3_wins = m2_wins = 0
    for i in range(len(results["Method1"])):
        r2 = results["Method2"][i]
        r3 = results["Method3"][i]
        if r3["recall@5"] and not r2["recall@5"]:
            m3_wins += 1
            print(f"  ✓ M3 wins [{r3['type']:12s}] {r3['query'][:60]}")
        elif r2["recall@5"] and not r3["recall@5"]:
            m2_wins += 1
            print(f"  ✗ M2 wins [{r2['type']:12s}] {r2['query'][:60]}")

    print(f"\n  M3 retrieves correctly where M2 fails: {m3_wins}")
    print(f"  M2 retrieves correctly where M3 fails: {m2_wins}")

    # summary
    m1 = sum(r["recall@5"] for r in results["Method1"])
    m2 = sum(r["recall@5"] for r in results["Method2"])
    m3 = sum(r["recall@5"] for r in results["Method3"])
    n  = len(results["Method1"])
    print(f"\n  Recall@5 totals — M1: {m1}/{n}   M2: {m2}/{n}   M3: {m3}/{n}")
    print()


if __name__ == "__main__":
    run()
