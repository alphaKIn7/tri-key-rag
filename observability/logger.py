import time
import json
from typing import List, Dict


class Logger:
    def __init__(self):
        self.records = []

    def log(self, method: str, question_id: int, query: str,
            answer: str, chunks: List[Dict], latency: float) -> None:
        self.records.append({
            "method":       method,
            "question_id":  question_id,
            "query":        query,
            "answer":       answer,
            "latency":      round(latency, 3),
            "top_chunks":   [{"score": c["score"], "text": c["text"][:80]} for c in chunks],
        })

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2)
        print(f"Log saved to {path}")
