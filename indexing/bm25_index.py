import re
import pickle
from rank_bm25 import BM25Okapi
from typing import List, Dict


class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.chunks = []

    def _tokenize(self, text: str) -> List[str]:
        # lowercase + split on anything that isn't a letter or number
        # "FAA Part 117" → ["faa", "part", "117"]
        return re.findall(r"\w+", text.lower())

    def build(self, chunks: List[Dict]) -> None:
        """
        Tokenizes every chunk and builds a BM25 index over them.
        Uses the same chunks list as VectorIndex so both indexes are in sync.
        """
        self.chunks = chunks
        tokenized = [self._tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)
        print(f"BM25 index built. {len(chunks)} documents indexed.")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Tokenizes the query and returns the top_k chunks by BM25 score.
        Chunks with score 0 (no keyword overlap at all) are excluded.
        """
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                result = self.chunks[idx].copy()
                result["score"] = round(float(scores[idx]), 4)
                results.append(result)

        return results

    def save(self, path: str) -> None:
        with open(path + ".bm25", "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)
        print(f"BM25 index saved to {path}.bm25")

    def load(self, path: str) -> None:
        with open(path + ".bm25", "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]
        print(f"BM25 index loaded. {len(self.chunks)} documents ready.")


# --- See the difference between BM25 and dense search ---
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from chunking.semantic_chunker import SemanticChunker
    from indexing.vector_index import VectorIndex

    with open("data/sample_docs/aviation_manual.txt", "r") as f:
        text = f.read()

    chunks = SemanticChunker(max_chunk_size=400, overlap=50).chunk_document(
        text, metadata={"source": "aviation_manual.txt"}
    )

    # Build both indexes on the same chunks
    bm25 = BM25Index()
    bm25.build(chunks)

    vi = VectorIndex()
    vi.build(chunks)

    # Test 1: exact technical acronym — BM25 should win
    print("=" * 60)
    print("TEST 1: Exact acronym — 'MEL'")
    print("=" * 60)
    print("\nBM25 top result:")
    r = bm25.search("MEL", top_k=1)
    if r:
        print(f"  score={r[0]['score']}  \"{r[0]['text'][:120]}...\"")

    print("\nDense top result:")
    r = vi.search("MEL", top_k=1)
    print(f"  score={r[0]['score']}  \"{r[0]['text'][:120]}...\"\n")

    # Test 2: paraphrase — dense should win
    print("=" * 60)
    print("TEST 2: Paraphrase — 'what happens when pilots get too tired'")
    print("=" * 60)
    print("\nBM25 top result:")
    r = bm25.search("what happens when pilots get too tired", top_k=1)
    if r:
        print(f"  score={r[0]['score']}  \"{r[0]['text'][:120]}...\"")
    else:
        print("  No results (score was 0 — no keyword overlap)")

    print("\nDense top result:")
    r = vi.search("what happens when pilots get too tired", top_k=1)
    print(f"  score={r[0]['score']}  \"{r[0]['text'][:120]}...\"\n")
