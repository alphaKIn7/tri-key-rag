from sentence_transformers import CrossEncoder
from typing import List, Dict


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        model_name: a cross-encoder model trained on MS MARCO (a large passage retrieval dataset).
        MiniLM-L-6-v2 is the small fast version — good for development.
        Scores are raw logits (can be negative). Higher = more relevant.
        """
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        Takes the query and a list of candidate chunks (from RRF fusion),
        scores every (query, chunk) pair together, and returns top_k reranked results.

        Input chunks can be from any retriever — just needs a "text" key.
        """
        if not chunks:
            return []

        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self.model.predict(pairs)

        scored_chunks = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True
        )

        results = []
        for score, chunk in scored_chunks[:top_k]:
            result = chunk.copy()
            result["score"] = round(float(score), 4)
            results.append(result)

        return results


# --- See the full retrieval cascade: FAISS + BM25 → RRF → Reranker ---
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from chunking.semantic_chunker import SemanticChunker
    from indexing.vector_index import VectorIndex
    from indexing.bm25_index import BM25Index
    from ranking.rrf_fusion import RRFFusion

    with open("data/sample_docs/aviation_manual.txt", "r") as f:
        text = f.read()

    chunks = SemanticChunker(max_chunk_size=400, overlap=50).chunk_document(
        text, metadata={"source": "aviation_manual.txt"}
    )

    vi    = VectorIndex()  ;  vi.build(chunks)
    bm25  = BM25Index()    ;  bm25.build(chunks)
    rrf   = RRFFusion(k=60)
    reranker = Reranker()

    queries = [
        "what happens when pilots get too tired to fly",
        "MEL broken component aircraft",
    ]

    for query in queries:
        dense_results = vi.search(query, top_k=5)
        bm25_results  = bm25.search(query, top_k=5)
        fused         = rrf.fuse(dense_results, bm25_results, top_k=5)
        reranked      = reranker.rerank(query, fused, top_k=3)

        print("=" * 60)
        print(f"QUERY: '{query}'")
        print("=" * 60)

        print("\nAfter RRF (before reranking):")
        for i, r in enumerate(fused[:3]):
            print(f"  #{i+1} rrf_score={r['score']}  \"{r['text'][:80]}...\"")

        print("\nAfter reranking:")
        for i, r in enumerate(reranked):
            print(f"  #{i+1} rerank_score={r['score']}  \"{r['text'][:80]}...\"")

        print()
