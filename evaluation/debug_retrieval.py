import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunking.semantic_chunker import SemanticChunker
from enrichment.enricher import Enricher
from indexing.vector_index import VectorIndex
from indexing.bm25_index import BM25Index
from ranking.rrf_fusion import RRFFusion


def build_indexes(doc_path):
    with open(doc_path) as f:
        text = f.read()

    chunks = SemanticChunker(max_chunk_size=400, overlap=50).chunk_document(
        text, metadata={"source": "aviation_manual.txt"}
    )
    enriched = Enricher().enrich(chunks)

    vi = VectorIndex(); vi.build(enriched)

    summary_chunks = []
    for c in enriched:
        s = c.copy(); s["original_text"] = c["text"]; s["text"] = c["summary"]
        summary_chunks.append(s)
    si = VectorIndex(); si.build(summary_chunks)

    bm25 = BM25Index(); bm25.build(enriched)

    return vi, si, bm25, enriched


def show_retrieval(query, vi, si, bm25, enriched, top_k=3):
    dense_results   = vi.search(query, top_k=top_k)
    summary_results = si.search(query, top_k=top_k)
    bm25_results    = bm25.search(query, top_k=top_k)

    # restore original text on summary results
    for r in summary_results:
        if "original_text" in r:
            r["text"] = r["original_text"]

    rrf = RRFFusion(k=60)
    fused = rrf.fuse(dense_results, summary_results, bm25_results, top_k=top_k)

    # build a lookup: chunk_id → summary
    summary_map = {c["metadata"]["chunk_id"]: c["summary"] for c in enriched}

    print(f"\n{'─'*65}")
    print(f"QUERY: {query}")
    print(f"{'─'*65}")

    for label, results in [("🔵 SUMMARY KEY", summary_results),
                           ("🟢 DENSE KEY",   dense_results),
                           ("🟡 BM25 KEY",    bm25_results)]:
        print(f"\n  {label}:")
        if not results:
            print("    (no results)")
        for i, r in enumerate(results):
            cid = r["metadata"]["chunk_id"]
            summary = summary_map.get(cid, "—")
            print(f"    #{i+1} chunk_id={cid}  score={r['score']}")
            print(f"         SUMMARY: {summary}")
            print(f"         RAW:     {r['text'][:90]}...")

    print(f"\n  ⚡ AFTER RRF FUSION (what goes to reranker):")
    for i, r in enumerate(fused):
        cid = r["metadata"]["chunk_id"]
        print(f"    #{i+1} chunk_id={cid}  rrf_score={r['score']}")
        print(f"         {r['text'][:90]}...")


if __name__ == "__main__":
    print("Building indexes...")
    vi, si, bm25, enriched = build_indexes("data/sample_docs/aviation_manual.txt")

    with open("data/eval_questions.json") as f:
        questions = json.load(f)

    # show all questions, or filter by difficulty
    filter_difficulty = None  # change to "hard" / "medium" / "easy" to filter

    for q in questions:
        if filter_difficulty and q["difficulty"] != filter_difficulty:
            continue
        show_retrieval(q["question"], vi, si, bm25, enriched)
