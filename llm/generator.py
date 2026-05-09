import os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class Generator:
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        model: which OpenAI model to use.
        gpt-4o-mini is cheap and fast — good default for development and testing.
        Switch to gpt-4o for higher quality answers in production.
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate(self, query: str, chunks: List[Dict]) -> Dict:
        """
        Takes a user question and a list of retrieved chunks, sends both to the
        LLM, and returns a structured response.

        chunks: output from VectorIndex.search() or any retriever — just needs
                "text" and "metadata" keys on each item.
        """
        context = self._format_context(chunks)
        prompt = self._build_prompt(query, context)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise assistant that answers questions using ONLY "
                        "the provided context. If the answer is not in the context, "
                        "say exactly: 'I don't have enough information to answer this.' "
                        "Never use outside knowledge."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,  # deterministic — no creativity, just facts from context
        )

        return {
            "answer": response.choices[0].message.content,
            "model": self.model,
            "chunks_used": len(chunks),
            "sources": [c["metadata"].get("source", "unknown") for c in chunks],
        }

    def _format_context(self, chunks: List[Dict]) -> str:
        """Formats the list of chunks into a readable block of text for the prompt."""
        parts = []
        for i, chunk in enumerate(chunks):
            parts.append(f"[Chunk {i + 1}]\n{chunk['text']}")
        return "\n\n".join(parts)

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""Use ONLY the context below to answer the question. Do not use any outside knowledge.
If the answer cannot be found in the context, say "I don't have enough information to answer this."

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""


# --- Test it end-to-end: chunker → vector index → generator ---
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from chunking.semantic_chunker import SemanticChunker
    from indexing.vector_index import VectorIndex

    # Step 1: chunk
    with open("data/sample_docs/aviation_manual.txt", "r") as f:
        text = f.read()

    chunks = SemanticChunker(max_chunk_size=400, overlap=50).chunk_document(
        text, metadata={"source": "aviation_manual.txt"}
    )
    print(f"Chunks created: {len(chunks)}\n")

    # Step 2: build index
    vi = VectorIndex()
    vi.build(chunks)

    # Step 3: test a few questions
    questions = [
        "What happens when a pilot exceeds their duty hours?",
        "How does a Ground Delay Program work?",
        "What is the Minimum Equipment List?",
        "What question cannot be answered from this document?",  # tests the "I don't know" case
    ]

    gen = Generator()

    for question in questions:
        print(f"QUESTION: {question}")
        retrieved = vi.search(question, top_k=3)
        result = gen.generate(question, retrieved)
        print(f"ANSWER:   {result['answer']}")
        print(f"Sources:  {result['sources']}")
        print(f"Chunks used: {result['chunks_used']}\n{'-' * 60}\n")
