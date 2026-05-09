import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict

load_dotenv()

REWRITE_PROMPT = """You are improving a search query for a document retrieval system.

Given the user's query, provide:
REWRITTEN: A single expanded sentence that adds context and uses domain terminology. Better for semantic search.
KEYWORDS: 3-6 key technical terms separated by commas. Better for keyword search.

User query: {query}

REWRITTEN:
KEYWORDS:"""


class QueryRewriter:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def rewrite(self, query: str) -> Dict:
        """
        Takes a raw user query and returns two improved versions:
        - "rewritten": expanded query for dense/semantic search
        - "keywords": key terms for BM25 keyword search
        - "original": the unchanged original query
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": REWRITE_PROMPT.format(query=query)}],
            temperature=0,
            max_tokens=100,
        )

        raw = response.choices[0].message.content.strip()
        rewritten, keywords = self._parse(raw, query)

        return {
            "original":  query,
            "rewritten": rewritten,
            "keywords":  keywords,
        }

    def _parse(self, raw: str, original: str) -> tuple:
        """Extracts REWRITTEN and KEYWORDS lines from the LLM response."""
        rewritten = original  # fallback to original if parsing fails
        keywords  = original

        for line in raw.splitlines():
            if line.upper().startswith("REWRITTEN:"):
                rewritten = line.split(":", 1)[1].strip()
            elif line.upper().startswith("KEYWORDS:"):
                keywords = line.split(":", 1)[1].strip()

        return rewritten, keywords


# --- See how the rewriter improves different query styles ---
if __name__ == "__main__":
    rewriter = QueryRewriter()

    queries = [
        "MEL",                                          # too short, just an acronym
        "what happens when pilots get too tired",        # informal paraphrase
        "ground delay",                                  # vague, two words
        "block time padding",                            # jargon without context
    ]

    for query in queries:
        result = rewriter.rewrite(query)
        print(f"ORIGINAL:  {result['original']}")
        print(f"REWRITTEN: {result['rewritten']}")
        print(f"KEYWORDS:  {result['keywords']}")
        print()
