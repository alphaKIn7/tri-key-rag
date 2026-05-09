import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunking.semantic_chunker import SemanticChunker
from indexing.vector_index import VectorIndex
from llm.generator import Generator
from typing import Dict


class Method1Pipeline:
    """
    🟢 Method 1 — Normal RAG (Dense Only)

    Flow: Document → Chunks → Embeddings → FAISS → Top-K → LLM → Answer
    One retrieval signal: semantic similarity only.
    """

    def __init__(self, top_k: int = 3, model: str = "gpt-4o-mini"):
        self.chunker = SemanticChunker(max_chunk_size=400, overlap=50)
        self.vector_index = VectorIndex()
        self.generator = Generator(model=model)
        self.top_k = top_k
        self._built = False

    def build(self, doc_path: str) -> None:
        """
        Reads a document, chunks it, embeds the chunks, and builds the FAISS index.
        Call this once before asking questions.
        """
        print(f"[Method 1] Building index from: {doc_path}")

        with open(doc_path, "r") as f:
            text = f.read()

        source_name = os.path.basename(doc_path)
        chunks = self.chunker.chunk_document(text, metadata={"source": source_name})
        print(f"[Method 1] Created {len(chunks)} chunks.")

        self.vector_index.build(chunks)
        self._built = True
        print("[Method 1] Ready.\n")

    def ask(self, query: str) -> Dict:
        """
        Ask a question. Returns the answer plus everything that happened internally
        so you can inspect the retrieval later.
        """
        if not self._built:
            raise RuntimeError("Call build() before ask().")

        retrieved = self.vector_index.search(query, top_k=self.top_k)
        result = self.generator.generate(query, retrieved)

        return {
            "query": query,
            "answer": result["answer"],
            "retrieved_chunks": retrieved,
            "model": result["model"],
        }

    def save(self, path: str) -> None:
        """Save the FAISS index to disk so you don't have to rebuild next time."""
        self.vector_index.save(path)

    def load(self, path: str) -> None:
        """Load a previously saved index."""
        self.vector_index.load(path)
        self._built = True


# --- Run Method 1 end-to-end ---
if __name__ == "__main__":
    pipeline = Method1Pipeline(top_k=3)
    pipeline.build("data/sample_docs/aviation_manual.txt")

    questions = [
        "What happens when a pilot exceeds their duty hours?",
        "How does a Ground Delay Program work?",
        "What is block time padding?",
        "How long is a standard turnaround for a narrow-body aircraft?",
    ]

    for question in questions:
        result = pipeline.ask(question)
        print(f"Q: {result['query']}")
        print(f"A: {result['answer']}")
        print()
        print("  Retrieved chunks:")
        for i, chunk in enumerate(result["retrieved_chunks"]):
            print(f"  [{i+1}] score={chunk['score']}  \"{chunk['text'][:80]}...\"")
        print("-" * 60 + "\n")
