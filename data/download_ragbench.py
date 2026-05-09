"""
Downloads a subset of vectara/open_ragbench.

Structure:
  - Each corpus file = one arXiv paper with a list of sections
  - queries.json = {query_id: {query, type, source}}
  - qrels.json   = {query_id: {doc_id, section_id}}  ← exact ground truth

We download TARGET_PAPERS papers, extract their sections as chunks,
keep only text-only queries referencing those papers, and save everything.
"""
import os
import json
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import hf_hub_download, list_repo_files

TARGET_PAPERS = 30
MIN_SECTION_CHARS = 200  # skip very short sections (headers, captions, etc.)
SAVE_DIR = os.path.join(os.path.dirname(__file__), "ragbench")
os.makedirs(SAVE_DIR, exist_ok=True)


def main():
    # ── 1. Download queries and qrels ─────────────────────────────────────────
    print("Downloading queries and qrels...")
    q_path = hf_hub_download("vectara/open_ragbench", "pdf/arxiv/queries.json", repo_type="dataset")
    r_path = hf_hub_download("vectara/open_ragbench", "pdf/arxiv/qrels.json",   repo_type="dataset")

    with open(q_path) as f: all_queries = json.load(f)
    with open(r_path) as f: all_qrels   = json.load(f)

    # ── 2. Get the list of corpus filenames ───────────────────────────────────
    print("Listing corpus files...")
    all_files = list(list_repo_files("vectara/open_ragbench", repo_type="dataset"))
    corpus_files = sorted([f for f in all_files if f.startswith("pdf/arxiv/corpus/")])
    print(f"  Total papers available: {len(corpus_files)}")

    # ── 3. Download TARGET_PAPERS papers and extract their sections ──────────
    print(f"\nDownloading {TARGET_PAPERS} papers...")
    corpus = {}  # {doc_id: {title, sections: [{section_id, text}]}}

    for cf in corpus_files[:TARGET_PAPERS]:
        doc_id = os.path.basename(cf).replace(".json", "")
        local  = hf_hub_download("vectara/open_ragbench", cf, repo_type="dataset")

        with open(local) as f:
            doc = json.load(f)

        sections = [
            {"section_id": s["section_id"], "text": s["text"].strip()}
            for s in doc.get("sections", [])
            if len(s.get("text", "").strip()) >= MIN_SECTION_CHARS
        ]

        if not sections:
            continue

        corpus[doc_id] = {
            "title":    doc.get("title", "").replace("\n", " ").strip(),
            "sections": sections,
        }
        print(f"  {doc_id}  {len(sections):3d} sections  \"{corpus[doc_id]['title'][:60]}\"")

    selected_doc_ids = set(corpus.keys())

    # ── 4. Filter to text-only queries referencing our papers ─────────────────
    queries = {}
    qrels   = {}

    for qid, qrel in all_qrels.items():
        doc_id     = qrel["doc_id"]
        section_id = qrel["section_id"]
        if doc_id not in selected_doc_ids:
            continue
        q = all_queries.get(qid, {})
        if q.get("source", "") != "text":   # skip multimodal (image/table)
            continue
        queries[qid] = {
            "query": q["query"],
            "type":  q.get("type", "unknown"),
        }
        qrels[qid] = {
            "doc_id":     doc_id,
            "section_id": section_id,
        }

    # ── 5. Save ───────────────────────────────────────────────────────────────
    with open(os.path.join(SAVE_DIR, "corpus.json"),  "w") as f: json.dump(corpus,   f, indent=2)
    with open(os.path.join(SAVE_DIR, "queries.json"), "w") as f: json.dump(queries,  f, indent=2)
    with open(os.path.join(SAVE_DIR, "qrels.json"),   "w") as f: json.dump(qrels,    f, indent=2)

    # ── 6. Cost estimate ──────────────────────────────────────────────────────
    total_sections = sum(len(d["sections"]) for d in corpus.values())
    est_chunks     = total_sections  # sections ≈ chunks at this stage
    enrich_cost    = est_chunks * 0.000086
    eval_cost      = len(queries) * 3 * 0.00018

    abs_count = sum(1 for q in queries.values() if q["type"] == "abstractive")
    ext_count = sum(1 for q in queries.values() if q["type"] == "extractive")

    print(f"\n{'='*55}")
    print(f"  Papers downloaded:   {len(corpus)}")
    print(f"  Sections (chunks):   {total_sections}")
    print(f"  Queries:             {len(queries)}  (abstractive: {abs_count}, extractive: {ext_count})")
    print(f"{'='*55}")
    print(f"  Enrichment cost:   ${enrich_cost:.4f}")
    print(f"  Evaluation cost:   ${eval_cost:.4f}")
    print(f"  Total:             ${enrich_cost + eval_cost:.4f}")

    print("\nSample queries:")
    for qid, q in list(queries.items())[:4]:
        doc_id     = qrels[qid]["doc_id"]
        section_id = qrels[qid]["section_id"]
        print(f"  [{q['type']:12s}] {q['query'][:65]}")
        print(f"               answer in: {doc_id}  section {section_id}")

    # ── 7. Pre-generate and cache summaries ───────────────────────────────────
    ans = input("\nPre-generate Method 3 summaries now and cache them? (y/n): ").strip().lower()
    if ans == "y":
        from chunking.semantic_chunker import SemanticChunker
        from enrichment.enricher import Enricher

        chunker = SemanticChunker(max_chunk_size=400, overlap=50)
        all_chunks = []
        for doc_id, doc in corpus.items():
            for section in doc["sections"]:
                chunks = chunker.chunk_document(
                    section["text"],
                    metadata={
                        "source":     doc_id,
                        "section_id": section["section_id"],
                        "title":      doc["title"],
                    }
                )
                all_chunks.extend(chunks)

        # fix chunk_ids to be globally unique across all sections
        for i, chunk in enumerate(all_chunks):
            chunk["metadata"]["chunk_id"] = i

        print(f"\nTotal chunks across all papers: {len(all_chunks)}")
        print(f"Actual enrichment cost: ${len(all_chunks) * 0.000086:.4f}")

        cache_path = os.path.join("data", "cache", "ragbench_enriched")
        os.makedirs(os.path.join("data", "cache"), exist_ok=True)
        Enricher().enrich(all_chunks, cache_path=cache_path)
        print(f"Summaries saved. Future pipeline builds will load from data/cache/ragbench_enriched.json")
    else:
        print("Skipped. Summaries will be generated on the first pipeline build.")


if __name__ == "__main__":
    main()
