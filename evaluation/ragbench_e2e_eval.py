"""
End-to-end ragbench evaluation.

For each query, all three methods:
  1. Retrieve top-3 chunks
  2. Generate an answer
  3. LLM-as-judge compares generated answer to ground truth

Reports both retrieval recall AND answer accuracy,
split by query type (abstractive vs extractive).

Results are cached to data/cache/e2e_results.json so re-runs are free.
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from dotenv import load_dotenv
from indexing.vector_index import VectorIndex
from indexing.bm25_index import BM25Index
from ranking.rrf_fusion import RRFFusion
from llm.generator import Generator

load_dotenv()

CACHE_DIR    = "data/cache"
INDEX_DIR    = os.path.join(CACHE_DIR, "ragbench_indexes")
TOP_K        = 5   # retrieval candidates
CONTEXT_K    = 5   # chunks actually sent to generator
RESULTS_FILE = os.path.join(CACHE_DIR, f"e2e_results_context{CONTEXT_K}.json")


JUDGE_PROMPT = """You are evaluating whether an AI-generated answer correctly answers a question.

Question: {question}

Ground truth answer: {ground_truth}

Generated answer: {generated}

Does the generated answer correctly answer the question based on the ground truth?
Reply with exactly one word: CORRECT or INCORRECT."""


def judge(client, question: str, ground_truth: str, generated: str) -> bool:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            generated=generated,
        )}],
        temperature=0,
        max_tokens=5,
    )
    verdict = response.choices[0].message.content.strip().upper()
    return verdict.startswith("CORRECT")


def load_indexes():
    vi   = VectorIndex(); vi.load(os.path.join(INDEX_DIR, "dense"))
    si   = VectorIndex(); si.load(os.path.join(INDEX_DIR, "summary"))
    bm25 = BM25Index()
    with open("data/cache/ragbench_enriched.json") as f:
        enriched = json.load(f)
    bm25.build(enriched)
    return vi, si, bm25, enriched


def retrieve(method, query, vi, si, bm25, rrf, top_k):
    if method == "Method1":
        return vi.search(query, top_k=top_k)
    dense   = vi.search(query, top_k=top_k)
    lexical = bm25.search(query, top_k=top_k)
    if method == "Method2":
        return rrf.fuse(dense, lexical, top_k=top_k)
    summary = si.search(query, top_k=top_k)
    for r in summary:
        if "original_text" in r:
            r["text"] = r["original_text"]
    return rrf.fuse(dense, summary, lexical, top_k=top_k)


def recall_at_k(chunks, doc_id, section_id, k):
    return any(
        c["metadata"].get("source") == doc_id and
        c["metadata"].get("section_id") == section_id
        for c in chunks[:k]
    )


def run():
    # load cached results if they exist
    if os.path.exists(RESULTS_FILE):
        print(f"Loading cached results from {RESULTS_FILE}")
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        _print_report(results)
        return

    with open("data/ragbench/queries.json")  as f: queries   = json.load(f)
    with open("data/ragbench/qrels.json")    as f: qrels     = json.load(f)
    with open("data/ragbench/answers.json")  as f: answers   = json.load(f)

    print("Loading indexes...")
    vi, si, bm25, _ = load_indexes()
    rrf = RRFFusion(k=60)
    gen = Generator(model="gpt-4o-mini")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    methods = ["Method1", "Method2", "Method3"]
    results = {m: [] for m in methods}
    total   = len(queries)

    print(f"Running {total} queries × 3 methods (generation + judging)...\n")

    for i, (qid, q) in enumerate(queries.items()):
        qrel          = qrels[qid]
        doc_id        = qrel["doc_id"]
        section_id    = qrel["section_id"]
        ground_truth  = answers.get(qid, "")

        for method in methods:
            chunks    = retrieve(method, q["query"], vi, si, bm25, rrf, TOP_K)
            context   = chunks[:CONTEXT_K]
            gen_out   = gen.generate(q["query"], context)
            generated = gen_out["answer"]
            correct   = judge(client, q["query"], ground_truth, generated)

            results[method].append({
                "qid":          qid,
                "query":        q["query"],
                "type":         q["type"],
                "ground_truth": ground_truth,
                "generated":    generated,
                "answer_correct": correct,
                "recall@5":     recall_at_k(chunks, doc_id, section_id, 5),
            })

        print(f"  [{i+1:02d}/{total}] done — {q['query'][:60]}")

    # cache results
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults cached to {RESULTS_FILE}")

    _print_report(results)


def _print_report(results):
    methods = ["Method1", "Method2", "Method3"]

    def pct(vals): return f"{sum(vals)/len(vals)*100:.1f}%" if vals else "—"
    def by_type(m, q_type, key):
        return [r[key] for r in results[m] if r["type"] == q_type]

    print("\n" + "=" * 68)
    print("  END-TO-END EVALUATION — Retrieval + Answer Quality")
    print("=" * 68)

    for metric_label, key in [("Recall@5 (retrieval)", "recall@5"),
                               ("Answer Accuracy",      "answer_correct")]:
        print(f"\n{metric_label:<30} {'🟢 M1':>10} {'🟡 M2':>10} {'🔵 M3':>10}")
        print("-" * 63)
        rows = [
            ("  All",          [[r[key] for r in results[m]] for m in methods]),
            ("  Abstractive",  [by_type(m, "abstractive", key) for m in methods]),
            ("  Extractive",   [by_type(m, "extractive",  key) for m in methods]),
        ]
        for label, row in rows:
            print(f"{label:<30} {pct(row[0]):>10} {pct(row[1]):>10} {pct(row[2]):>10}")

    # per-query answer comparison where methods disagree
    print(f"\n\nQueries where methods give different answers:")
    print("-" * 68)
    shown = 0
    for i in range(len(results["Method1"])):
        r1 = results["Method1"][i]
        r2 = results["Method2"][i]
        r3 = results["Method3"][i]
        verdicts = [r1["answer_correct"], r2["answer_correct"], r3["answer_correct"]]
        if len(set(verdicts)) > 1:  # not all same
            shown += 1
            m = lambda r: "✓" if r["answer_correct"] else "✗"
            print(f"\n  [{r1['type']:12s}] {r1['query'][:60]}")
            print(f"  M1:{m(r1)} M2:{m(r2)} M3:{m(r3)}")
            print(f"  Ground truth: {r1['ground_truth'][:100]}...")
            print(f"  M3 answer:    {r3['generated'][:100]}...")
    if shown == 0:
        print("  All methods agreed on every query.")

    # summary
    for method, label in [("Method1","M1"), ("Method2","M2"), ("Method3","M3")]:
        acc = sum(r["answer_correct"] for r in results[method])
        n   = len(results[method])
        print(f"\n  {label} answered correctly: {acc}/{n}")


if __name__ == "__main__":
    run()
