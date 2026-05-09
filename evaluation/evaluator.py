import sys
import os
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.method1_pipeline import Method1Pipeline
from pipeline.method2_pipeline import Method2Pipeline
from pipeline.method3_pipeline import Method3Pipeline
from observability.logger import Logger
from typing import List, Dict


def answer_contains(answer: str, key_phrases: List[str]) -> bool:
    """
    Returns True if every key phrase appears in the answer (case-insensitive).
    This is our proxy for answer correctness — no manual labelling needed.
    """
    answer_lower = answer.lower()
    return all(phrase.lower() in answer_lower for phrase in key_phrases)


def recall_at_k(chunks: List[Dict], key_phrases: List[str]) -> bool:
    """
    Returns True if at least one of the retrieved chunks contains all key phrases.
    Measures whether the right information was retrieved, regardless of answer quality.
    """
    for chunk in chunks:
        chunk_lower = chunk["text"].lower()
        if all(phrase.lower() in chunk_lower for phrase in key_phrases):
            return True
    return False


def run_evaluation(doc_path: str, questions_path: str) -> Dict:
    with open(questions_path) as f:
        questions = json.load(f)

    logger = Logger()

    print("Building pipelines (this takes a moment for Method 3)...\n")
    m1 = Method1Pipeline(top_k=3)
    m2 = Method2Pipeline(top_k_retrieve=5, top_k_rerank=3)
    m3 = Method3Pipeline(top_k_retrieve=5, top_k_rerank=3)

    m1.build(doc_path)
    m2.build(doc_path)
    m3.build(doc_path)

    results = {
        "Method1": [],
        "Method2": [],
        "Method3": [],
    }

    print(f"\nRunning {len(questions)} questions across all 3 methods...\n")

    for q in questions:
        for label, pipeline in [("Method1", m1), ("Method2", m2), ("Method3", m3)]:
            start = time.time()
            r = pipeline.ask(q["question"])
            latency = time.time() - start

            correct  = answer_contains(r["answer"], q["key_phrases"])
            recalled = recall_at_k(r["retrieved_chunks"], q["key_phrases"])

            logger.log(label, q["id"], q["question"], r["answer"],
                       r["retrieved_chunks"], latency)

            results[label].append({
                "question_id":    q["id"],
                "difficulty":     q["difficulty"],
                "question":       q["question"],
                "answer":         r["answer"],
                "answer_correct": correct,
                "recall_at_k":    recalled,
                "latency":        round(latency, 3),
            })

        print(f"  Q{q['id']:02d} [{q['difficulty']:6s}] done")

    logger.save("data/eval_log.json")
    return results


if __name__ == "__main__":
    from evaluation.comparison_report import print_report

    results = run_evaluation(
        doc_path="data/sample_docs/aviation_manual.txt",
        questions_path="data/eval_questions.json",
    )
    print_report(results)
