"""
Grounded Answer Fusion evaluation.

Each retrieval index generates its own answer independently.
The synthesiser receives (answer + source chunks) bundles — not just answers —
so it can verify each claim against evidence before synthesising.

This solves the core synthesiser failure: confidently wrong answers
are exposed when the synthesiser checks them against their source.

Architecture:
  Dense index   → top 5 chunks → Generator → Answer A + 5 chunks
  Summary index → top 5 chunks → Generator → Answer B + 5 chunks
  BM25 index    → top 5 chunks → Generator → Answer C + 5 chunks
                                                      ↓
                              Synthesiser sees (A+evidence), (B+evidence), (C+evidence)
                              Verifies each answer against its source
                                                      ↓
                                              Final grounded answer

Results cached to data/cache/grounded_results.json.
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from dotenv import load_dotenv
from indexing.vector_index import VectorIndex
from indexing.bm25_index import BM25Index
from llm.generator import Generator

load_dotenv()

CACHE_DIR    = "data/cache"
INDEX_DIR    = os.path.join(CACHE_DIR, "ragbench_indexes")
GEN_MODEL    = "gpt-4o"          # used for generation + synthesis
JUDGE_MODEL  = "gpt-4o-mini"    # judge is yes/no — mini is sufficient
RESULTS_FILE = os.path.join(CACHE_DIR, f"grounded_results_{GEN_MODEL.replace('-','')}_v2.json")
TOP_K        = 5


SYNTHESISER_PROMPT = """You are synthesising three independently generated answers to the same question.
Each answer comes with the source chunks it was based on.

Instructions:
1. Use the source chunks to assess how confident you are in each answer
2. Prefer answers that are clearly grounded in their source chunks over those that are not
3. If answers conflict, favour the one better supported by its evidence
4. If multiple answers agree, synthesise them into one clear response
5. Keep your answer concise — match the length and directness of the question (yes/no questions need a yes/no answer, not a paragraph)

QUESTION: {question}

--- SYSTEM 1: Dense Search ---
Answer: {answer_1}
Source chunks:
{chunks_1}

--- SYSTEM 2: Intent Search ---
Answer: {answer_2}
Source chunks:
{chunks_2}

--- SYSTEM 3: Keyword Search ---
Answer: {answer_3}
Source chunks:
{chunks_3}

Final answer:"""


JUDGE_PROMPT = """You are evaluating whether an AI-generated answer correctly answers a question.

Question: {question}
Ground truth answer: {ground_truth}
Generated answer: {generated}

Does the generated answer correctly answer the question based on the ground truth?
Reply with exactly one word: CORRECT or INCORRECT."""


def format_chunks(chunks):
    return "\n\n".join(
        f"[{i+1}] {c['text'][:300]}" for i, c in enumerate(chunks)
    )


def load_indexes():
    vi   = VectorIndex(); vi.load(os.path.join(INDEX_DIR, "dense"))
    si   = VectorIndex(); si.load(os.path.join(INDEX_DIR, "summary"))
    bm25 = BM25Index()
    with open("data/cache/ragbench_enriched.json") as f:
        enriched = json.load(f)
    bm25.build(enriched)
    return vi, si, bm25


def judge(client, question, ground_truth, generated):
    r = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, ground_truth=ground_truth, generated=generated
        )}],
        temperature=0, max_tokens=5,
    )
    return r.choices[0].message.content.strip().upper().startswith("CORRECT")


def run():
    if os.path.exists(RESULTS_FILE):
        print(f"Loading cached results from {RESULTS_FILE}")
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        _print_report(results)
        return

    with open("data/ragbench/queries.json")  as f: queries = json.load(f)
    with open("data/ragbench/answers.json")  as f: answers = json.load(f)

    print("Loading indexes...")
    vi, si, bm25 = load_indexes()
    gen    = Generator(model=GEN_MODEL)
    client = OpenAI()

    results = []
    total   = len(queries)

    print(f"Running grounded synthesis on {total} queries...\n")

    for i, (qid, q) in enumerate(queries.items()):
        ground_truth = answers.get(qid, "")

        # ── each index retrieves and generates independently ──────────────
        dense_chunks   = vi.search(q["query"],   top_k=TOP_K)
        summary_chunks = si.search(q["query"],   top_k=TOP_K)
        for r in summary_chunks:
            if "original_text" in r:
                r["text"] = r["original_text"]
        bm25_chunks    = bm25.search(q["query"], top_k=TOP_K)

        answer_1 = gen.generate(q["query"], dense_chunks[:3])["answer"]
        answer_2 = gen.generate(q["query"], summary_chunks[:3])["answer"]
        answer_3 = gen.generate(q["query"], bm25_chunks[:3])["answer"]

        # ── synthesiser sees answers + evidence bundles ───────────────────
        prompt = SYNTHESISER_PROMPT.format(
            question=q["query"],
            answer_1=answer_1,
            chunks_1=format_chunks(dense_chunks[:3]),
            answer_2=answer_2,
            chunks_2=format_chunks(summary_chunks[:3]),
            answer_3=answer_3,
            chunks_3=format_chunks(bm25_chunks[:3]),
        )
        synth_response = client.chat.completions.create(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        final_answer = synth_response.choices[0].message.content.strip()

        correct = judge(client, q["query"], ground_truth, final_answer)

        results.append({
            "qid":            qid,
            "query":          q["query"],
            "type":           q["type"],
            "ground_truth":   ground_truth,
            "answer_1":       answer_1,
            "answer_2":       answer_2,
            "answer_3":       answer_3,
            "final_answer":   final_answer,
            "answer_correct": correct,
        })

        print(f"  [{i+1:02d}/{total}] {'✓' if correct else '✗'}  [{q['type']:12s}]  {q['query'][:55]}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults cached to {RESULTS_FILE}")

    _print_report(results)


def _print_report(results):
    def pct(vals): return f"{sum(vals)/len(vals)*100:.1f}%" if vals else "—"
    def by_type(q_type, key):
        return [r[key] for r in results if r["type"] == q_type]

    total   = len(results)
    correct = sum(r["answer_correct"] for r in results)

    # load M3 context5 results for comparison
    m3_file = os.path.join(CACHE_DIR, "e2e_results_context5.json")
    m3_acc  = m3_abs = m3_ext = None
    if os.path.exists(m3_file):
        with open(m3_file) as f:
            m3_data = json.load(f)
        m3_all = [r["answer_correct"] for r in m3_data["Method3"]]
        m3_abs = [r["answer_correct"] for r in m3_data["Method3"] if r["type"] == "abstractive"]
        m3_ext = [r["answer_correct"] for r in m3_data["Method3"] if r["type"] == "extractive"]
        m3_acc = m3_all

    print("\n" + "=" * 65)
    print("  GROUNDED SYNTHESIS vs METHOD 3 (best existing)")
    print("=" * 65)

    print(f"\n{'Metric':<30} {'🟡 M3 (context5)':>18} {'🔵 Grounded':>12}")
    print("-" * 63)

    rows = [
        ("Answer Accuracy — All",
         pct(m3_acc) if m3_acc else "—",
         pct([r["answer_correct"] for r in results])),
        ("  Abstractive",
         pct(m3_abs) if m3_abs else "—",
         pct(by_type("abstractive", "answer_correct"))),
        ("  Extractive",
         pct(m3_ext) if m3_ext else "—",
         pct(by_type("extractive",  "answer_correct"))),
    ]
    for label, m3_val, gs_val in rows:
        print(f"{label:<30} {m3_val:>18} {gs_val:>12}")

    print(f"\n  Grounded: {correct}/{total} correct")

    # show cases where grounded synthesis changes the outcome vs M3
    if os.path.exists(m3_file):
        with open(m3_file) as f:
            m3_data = json.load(f)
        m3_by_qid = {r["qid"]: r for r in m3_data["Method3"]}

        wins = losses = 0
        print(f"\n{'Cases where Grounded differs from M3':}")
        print("-" * 65)
        for r in results:
            m3_r = m3_by_qid.get(r["qid"])
            if not m3_r:
                continue
            if r["answer_correct"] and not m3_r["answer_correct"]:
                wins += 1
                print(f"  ✓ Grounded wins [{r['type']:12s}] {r['query'][:50]}")
            elif not r["answer_correct"] and m3_r["answer_correct"]:
                losses += 1
                print(f"  ✗ Grounded loses [{r['type']:12s}] {r['query'][:50]}")

        print(f"\n  Grounded wins over M3:  {wins}")
        print(f"  Grounded loses to M3:   {losses}")
    print()


if __name__ == "__main__":
    run()
