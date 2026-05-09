import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunking.semantic_chunker import SemanticChunker
from indexing.vector_index import VectorIndex
from indexing.bm25_index import BM25Index
from ranking.rrf_fusion import RRFFusion
from ranking.reranker import Reranker
from llm.generator import Generator
from typing import Dict


class Method2Pipeline:
    """
    🟡 Method 2 — Hybrid RAG (Dense + Lexical)

    Flow: Document → Chunks → FAISS + BM25 → RRF → Reranker → LLM → Answer
    Two retrieval signals: semantic similarity + keyword matching.
    """

    def __init__(self, top_k_retrieve: int = 5, top_k_rerank: int = 3, model: str = "gpt-4o-mini"):
        self.chunker      = SemanticChunker(max_chunk_size=400, overlap=50)
        self.vector_index = VectorIndex()
        self.bm25_index   = BM25Index()
        self.rrf          = RRFFusion(k=60)
        self.reranker     = Reranker()
        self.generator    = Generator(model=model)
        self.top_k_retrieve = top_k_retrieve  # how many each retriever returns before fusion
        self.top_k_rerank   = top_k_rerank    # how many the reranker keeps after fusion
        self._built = False

    def build(self, doc_path: str) -> None:
        """
        Reads a document, chunks it, and builds both the FAISS and BM25 indexes.
        Call this once before asking questions.
        """
        print(f"[Method 2] Building indexes from: {doc_path}")

        with open(doc_path, "r") as f:
            text = f.read()

        source_name = os.path.basename(doc_path)
        chunks = self.chunker.chunk_document(text, metadata={"source": source_name})
        print(f"[Method 2] Created {len(chunks)} chunks.")

        self.vector_index.build(chunks)
        self.bm25_index.build(chunks)
        self._built = True
        print("[Method 2] Ready.\n")

    def ask(self, query: str) -> Dict:
        """
        Ask a question. Returns the answer plus the full retrieval trail
        so you can compare against Method 1 later.
        """
        if not self._built:
            raise RuntimeError("Call build() before ask().")

        dense_results = self.vector_index.search(query, top_k=self.top_k_retrieve)
        bm25_results  = self.bm25_index.search(query,  top_k=self.top_k_retrieve)
        fused         = self.rrf.fuse(dense_results, bm25_results, top_k=self.top_k_retrieve)
        reranked      = self.reranker.rerank(query, fused, top_k=self.top_k_rerank)
        result        = self.generator.generate(query, reranked)

        return {
            "query":            query,
            "answer":           result["answer"],
            "retrieved_chunks": reranked,
            "model":            result["model"],
        }


# --- Run Method 1 and Method 2 side by side on the same questions ---
if __name__ == "__main__":
    from pipeline.method1_pipeline import Method1Pipeline

    doc = "data/sample_docs/aviation_manual.txt"

    m1 = Method1Pipeline(top_k=3)
    m2 = Method2Pipeline(top_k_retrieve=5, top_k_rerank=3)

    print("Building Method 1...")
    m1.build(doc)
    print("Building Method 2...")
    m2.build(doc)

    questions = [
        "What happens when a pilot exceeds their duty hours?",
        "What is the Minimum Equipment List?",
        "How does a Ground Delay Program work?",
        "What is block time padding?",
    ]

    for question in questions:
        r1 = m1.ask(question)
        r2 = m2.ask(question)

        print("=" * 60)
        print(f"Q: {question}")
        print("=" * 60)
        print(f"\n🟢 Method 1: {r1['answer']}")
        print(f"\n🟡 Method 2: {r2['answer']}")

        print("\n  Method 1 retrieved chunks:")
        for i, c in enumerate(r1["retrieved_chunks"]):
            print(f"  [{i+1}] score={c['score']}")
            print(f"       {c['text']}")

        print("\n  Method 2 retrieved chunks:")
        for i, c in enumerate(r2["retrieved_chunks"]):
            print(f"  [{i+1}] score={c['score']}")
            print(f"       {c['text']}")

        print()
