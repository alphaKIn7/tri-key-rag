import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

SUMMARY_PROMPT = """You are preparing a document chunk for a search system.
Write a single concise sentence starting with "This chunk helps a user find out" that captures the core topic and what question a user would ask to find this information.

Chunk:
{text}

Summary:"""


class Enricher:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def _summarize(self, text: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": SUMMARY_PROMPT.format(text=text)}],
            temperature=0,
            max_tokens=80,  # summaries should be short — one tight sentence
        )
        return response.choices[0].message.content.strip()

    def enrich(self, chunks: List[Dict], cache_path: str = None) -> List[Dict]:
        """
        Takes the output of SemanticChunker.chunk_document() and adds a
        "summary" field to every chunk by calling the LLM once per chunk.

        cache_path: if provided, summaries are saved to <cache_path>.json on
        first run and loaded from disk on every subsequent run — no LLM calls.
        """
        if cache_path:
            full_path = cache_path + ".json"
            if os.path.exists(full_path):
                print(f"Loading cached summaries from {full_path}")
                with open(full_path) as f:
                    return json.load(f)

        print(f"Enriching {len(chunks)} chunks with LLM summaries...")
        enriched = []

        for i, chunk in enumerate(chunks):
            summary = self._summarize(chunk["text"])
            enriched_chunk = chunk.copy()
            enriched_chunk["summary"] = summary
            enriched.append(enriched_chunk)
            print(f"  [{i+1}/{len(chunks)}] {summary}")

        print("Enrichment complete.\n")

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path + ".json", "w") as f:
                json.dump(enriched, f, indent=2)
            print(f"Summaries cached to {cache_path}.json\n")

        return enriched


# --- See what the LLM writes as summaries for each aviation chunk ---
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from chunking.semantic_chunker import SemanticChunker

    with open("data/sample_docs/aviation_manual.txt", "r") as f:
        text = f.read()

    chunks = SemanticChunker(max_chunk_size=400, overlap=50).chunk_document(
        text, metadata={"source": "aviation_manual.txt"}
    )

    enricher = Enricher()
    enriched_chunks = enricher.enrich(chunks)

    print("\n--- Side by side: raw text vs summary ---\n")
    for chunk in enriched_chunks:
        print(f"Chunk {chunk['metadata']['chunk_id']}:")
        print(f"  RAW:     {chunk['text'][:120]}...")
        print(f"  SUMMARY: {chunk['summary']}")
        print()
