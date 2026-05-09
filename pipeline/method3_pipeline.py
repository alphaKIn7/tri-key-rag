import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunking.semantic_chunker import SemanticChunker
from enrichment.enricher import Enricher
from indexing.vector_index import VectorIndex
from indexing.bm25_index import BM25Index
from ranking.rrf_fusion import RRFFusion
from ranking.reranker import Reranker
from llm.generator import Generator
from typing import Dict, List


class Method3Pipeline:
    """
    🔵 Method 3 — Tri-Key RAG (Dense + Lexical + Intent)

    Flow: Document → Chunks → Enricher → FAISS (raw) + BM25 + FAISS (summary)
                                                    ↓
                              Query → 3-way search → RRF → Reranker → LLM → Answer

    Three retrieval signals:
      1. Semantic key  — embedding of raw chunk text
      2. Lexical key   — BM25 keyword match
      3. Intent key    — embedding of LLM-generated summary  ← the Tri-Key innovation
    """

    def __init__(self, top_k_retrieve: int = 5, top_k_rerank: int = 3, model: str = "gpt-4o-mini"):
        self.chunker        = SemanticChunker(max_chunk_size=400, overlap=50)
        self.enricher       = Enricher(model=model)
        self.vector_index   = VectorIndex()   # embeds raw chunk text
        self.summary_index  = VectorIndex()   # embeds LLM summaries
        self.bm25_index     = BM25Index()
        self.rrf            = RRFFusion(k=60)
        self.reranker       = Reranker()
        self.generator      = Generator(model=model)
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank   = top_k_rerank
        self._built         = False

    def build(self, doc_path: str) -> None:
        """
        Reads a document, chunks it, enriches every chunk with an LLM summary,
        then builds all three indexes.
        """
        print(f"[Method 3] Building indexes from: {doc_path}")

        with open(doc_path, "r") as f:
            text = f.read()

        source_name = os.path.basename(doc_path)
        chunks = self.chunker.chunk_document(text, metadata={"source": source_name})
        print(f"[Method 3] Created {len(chunks)} chunks.")

        cache_path = os.path.join("data", "cache", source_name.replace(".", "_") + "_enriched")
        enriched_chunks = self.enricher.enrich(chunks, cache_path=cache_path)

        # Index 1: semantic key — embed raw chunk text
        self.vector_index.build(enriched_chunks)

        # Index 2: intent key — embed summaries
        # We swap "text" → "summary" so VectorIndex embeds the summary, not the raw text.
        # The chunk metadata (including chunk_id) is preserved so RRF can deduplicate correctly.
        summary_chunks = self._swap_text_for_summary(enriched_chunks)
        self.summary_index.build(summary_chunks)

        # Index 3: lexical key — BM25 over raw text
        self.bm25_index.build(enriched_chunks)

        self._built = True
        print("[Method 3] Ready.\n")

    def ask(self, query: str) -> Dict:
        """
        Searches all three indexes, fuses results with RRF, reranks, and generates an answer.
        """
        if not self._built:
            raise RuntimeError("Call build() before ask().")

        dense_results   = self.vector_index.search(query,  top_k=self.top_k_retrieve)
        summary_results = self.summary_index.search(query, top_k=self.top_k_retrieve)
        summary_results = self._restore_original_text(summary_results)
        bm25_results    = self.bm25_index.search(query,    top_k=self.top_k_retrieve)

        fused    = self.rrf.fuse(dense_results, summary_results, bm25_results, top_k=self.top_k_retrieve)
        reranked = self.reranker.rerank(query, fused, top_k=self.top_k_rerank)
        result   = self.generator.generate(query, reranked)

        return {
            "query":            query,
            "answer":           result["answer"],
            "retrieved_chunks": reranked,
            "model":            result["model"],
        }

    def _swap_text_for_summary(self, enriched_chunks: List[Dict]) -> List[Dict]:
        """
        Returns a copy of the chunks list where "text" is replaced by "summary".
        Preserves the original text in "original_text" so it can be restored
        after retrieval — the generator must always receive the original chunk text.
        """
        swapped = []
        for chunk in enriched_chunks:
            c = chunk.copy()
            c["original_text"] = chunk["text"]  # preserve so we can restore later
            c["text"] = chunk["summary"]
            swapped.append(c)
        return swapped

    def _restore_original_text(self, results: List[Dict]) -> List[Dict]:
        """
        After summary_index.search() returns chunks, swap the text back to
        the original so the reranker and generator receive real content.
        """
        for r in results:
            if "original_text" in r:
                r["text"] = r["original_text"]
        return results


# --- Run all three methods side by side ---
if __name__ == "__main__":
    from pipeline.method1_pipeline import Method1Pipeline
    from pipeline.method2_pipeline import Method2Pipeline

    doc = "data/sample_docs/aviation_manual.txt"

    m1 = Method1Pipeline(top_k=3)
    m2 = Method2Pipeline(top_k_retrieve=5, top_k_rerank=3)
    m3 = Method3Pipeline(top_k_retrieve=5, top_k_rerank=3)

    print("Building all three methods...\n")
    m1.build(doc)
    m2.build(doc)
    m3.build(doc)

    questions = [
        "What happens when a pilot exceeds their duty hours?",
        "What is the Minimum Equipment List?",
        "What is block time padding?",
        "what happens when pilots get too tired to fly",
    ]

    for question in questions:
        r1 = m1.ask(question)
        r2 = m2.ask(question)
        r3 = m3.ask(question)

        print("=" * 60)
        print(f"Q: {question}")
        print("=" * 60)
        print(f"\n🟢 Method 1: {r1['answer']}")
        print(f"\n🟡 Method 2: {r2['answer']}")
        print(f"\n🔵 Method 3: {r3['answer']}")

        print("\n  Top chunk per method:")
        for label, r in [("M1", r1), ("M2", r2), ("M3", r3)]:
            c = r["retrieved_chunks"][0]
            print(f"  {label}: score={c['score']}  \"{c['text'][:100]}...\"")
        print()
