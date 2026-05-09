import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict


class VectorIndex:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        model_name: which sentence-transformer model to use for embeddings.
        all-MiniLM-L6-v2 is fast, small (80MB), and produces 384-dimensional vectors.
        Good default for experimentation.
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []  # keeps original chunk dicts alongside the index so search results carry full metadata

    def build(self, chunks: List[Dict]) -> None:
        """
        Takes the output of SemanticChunker.chunk_document() and builds a FAISS index.
        After this, you can call .search() to find similar chunks for any query.
        """
        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]

        print(f"Embedding {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        # Normalize to unit length so that inner product == cosine similarity
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]  # 384 for all-MiniLM-L6-v2
        self.index = faiss.IndexFlatIP(dimension)  # IP = inner product (cosine after normalization)
        self.index.add(embeddings)

        print(f"Index built. {self.index.ntotal} vectors stored.")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Find the top_k chunks most semantically similar to the query.
        Returns a list of chunk dicts, each with an added "score" key (0.0 to 1.0).
        Higher score = more similar.
        """
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS pads with -1 when fewer than top_k results exist
                continue
            result = self.chunks[idx].copy()
            result["score"] = round(float(score), 4)
            results.append(result)

        return results

    def save(self, path: str) -> None:
        """
        Saves index to disk so you don't have to rebuild it every run.
        Creates two files: <path>.faiss and <path>.chunks
        """
        faiss.write_index(self.index, path + ".faiss")
        with open(path + ".chunks", "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"Index saved to {path}.faiss / {path}.chunks")

    def load(self, path: str) -> None:
        """Load a previously saved index back into memory."""
        self.index = faiss.read_index(path + ".faiss")
        with open(path + ".chunks", "rb") as f:
            self.chunks = pickle.load(f)
        print(f"Index loaded. {self.index.ntotal} vectors ready.")


# --- Test it ---
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from chunking.semantic_chunker import SemanticChunker

    # Step 1: chunk the aviation manual
    with open("data/sample_docs/aviation_manual.txt", "r") as f:
        text = f.read()

    chunker = SemanticChunker(max_chunk_size=400, overlap=50)
    chunks = chunker.chunk_document(text, metadata={"source": "aviation_manual.txt"})
    print(f"Created {len(chunks)} chunks.\n")

    # Step 2: build the vector index
    vi = VectorIndex()
    vi.build(chunks)

    # Step 3: run a test query
    query = "What happens when a pilot exceeds their duty hours?"
    print(f"\nQuery: '{query}'\n")
    results = vi.search(query, top_k=3)

    for i, result in enumerate(results):
        print(f"--- Result {i+1} (score: {result['score']}) ---")
        print(f"Chunk ID: {result['metadata']['chunk_id']}")
        print(f"Text: {result['text']}\n")

    # Step 4: save and reload to prove persistence works
    vi.save("data/aviation_index")
    vi2 = VectorIndex()
    vi2.load("data/aviation_index")
    print("\nReloaded index. Running same query from reloaded index:")
    results2 = vi2.search(query, top_k=1)
    print(f"Top result score: {results2[0]['score']}")
    print(f"Top result text: {results2[0]['text']}")
