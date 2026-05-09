from typing import List, Dict


class RRFFusion:
    def __init__(self, k: int = 60):
        """
        k: the dampening constant. 60 is the standard value from the original RRF paper.
        It prevents rank-1 results from dominating over rank-2 results too heavily.
        Higher k = more equal weight across ranks. Lower k = rank-1 matters much more.
        """
        self.k = k

    def fuse(self, *ranked_lists: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Merges any number of ranked lists into one combined ranking.

        Each list must be sorted best-first (index 0 = most relevant).
        Each chunk dict must have metadata["chunk_id"] so duplicates can be detected.

        Returns top_k results sorted by combined RRF score, highest first.
        """
        scores = {}    # chunk_id → accumulated RRF score
        chunk_map = {} # chunk_id → chunk dict (to reconstruct full result)

        for ranked_list in ranked_lists:
            for rank, chunk in enumerate(ranked_list):
                chunk_id = chunk["metadata"]["chunk_id"]
                # rank is 0-indexed so we add 1 to match the formula (rank starts at 1)
                rrf_score = 1.0 / (self.k + rank + 1)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_score
                chunk_map[chunk_id] = chunk

        sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

        results = []
        for cid in sorted_ids[:top_k]:
            result = chunk_map[cid].copy()
            result["score"] = round(scores[cid], 6)
            results.append(result)

        return results


# --- See RRF fix what neither FAISS nor BM25 could alone ---
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from chunking.semantic_chunker import SemanticChunker
    from indexing.vector_index import VectorIndex
    from indexing.bm25_index import BM25Index

    with open("data/sample_docs/aviation_manual.txt", "r") as f:
        text = f.read()

    chunks = SemanticChunker(max_chunk_size=400, overlap=50).chunk_document(
        text, metadata={"source": "aviation_manual.txt"}
    )

    vi = VectorIndex()
    vi.build(chunks)

    bm25 = BM25Index()
    bm25.build(chunks)

    rrf = RRFFusion(k=60)

    queries = [
        "what happens when pilots get too tired",
        "MEL",
    ]

    for query in queries:
        dense_results  = vi.search(query, top_k=5)
        bm25_results   = bm25.search(query, top_k=5)
        fused_results  = rrf.fuse(dense_results, bm25_results, top_k=3)

        print("=" * 60)
        print(f"QUERY: '{query}'")
        print("=" * 60)

        print("\nFAISS top 3:")
        for i, r in enumerate(dense_results[:3]):
            print(f"  #{i+1} score={r['score']}  \"{r['text'][:80]}...\"")

        print("\nBM25 top 3:")
        for i, r in enumerate(bm25_results[:3]):
            print(f"  #{i+1} score={r['score']}  \"{r['text'][:80]}...\"")

        print("\nAfter RRF fusion top 3:")
        for i, r in enumerate(fused_results):
            print(f"  #{i+1} score={r['score']}  \"{r['text'][:80]}...\"")

        print()
