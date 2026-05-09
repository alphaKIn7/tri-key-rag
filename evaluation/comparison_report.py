from typing import Dict, List


def _pct(values: List[bool]) -> str:
    return f"{sum(values) / len(values) * 100:.0f}%"

def _avg(values: List[float]) -> str:
    return f"{sum(values) / len(values):.2f}s"

def _by_difficulty(results: List[Dict], difficulty: str, key: str) -> List:
    return [r[key] for r in results if r["difficulty"] == difficulty]


def print_report(results: Dict) -> None:
    methods = ["Method1", "Method2", "Method3"]
    labels  = {"Method1": "🟢 Method 1 (Dense)", "Method2": "🟡 Method 2 (Hybrid)", "Method3": "🔵 Method 3 (Tri-Key)"}

    print("\n" + "=" * 70)
    print("  EVALUATION REPORT — Three-Architecture RAG Comparison")
    print("=" * 70)

    # --- Overall summary table ---
    print(f"\n{'Metric':<28} {'🟢 M1':>10} {'🟡 M2':>10} {'🔵 M3':>10}")
    print("-" * 62)

    for metric_label, key in [("Answer Accuracy (all)", "answer_correct"),
                               ("Recall@K (all)",        "recall_at_k")]:
        row = [_pct([r[key] for r in results[m]]) for m in methods]
        print(f"{metric_label:<28} {row[0]:>10} {row[1]:>10} {row[2]:>10}")

    print(f"{'Avg Latency (all)':<28} {_avg([r['latency'] for r in results['Method1']]):>10} "
          f"{_avg([r['latency'] for r in results['Method2']]):>10} "
          f"{_avg([r['latency'] for r in results['Method3']]):>10}")

    # --- By difficulty ---
    print(f"\n{'By difficulty':<28} {'🟢 M1':>10} {'🟡 M2':>10} {'🔵 M3':>10}")
    print("-" * 62)
    for diff in ["easy", "medium", "hard"]:
        for metric_label, key in [(f"  {diff} — accuracy", "answer_correct"),
                                  (f"  {diff} — recall",   "recall_at_k")]:
            row = [_pct(_by_difficulty(results[m], diff, key)) for m in methods]
            print(f"{metric_label:<28} {row[0]:>10} {row[1]:>10} {row[2]:>10}")

    # --- Per-question breakdown ---
    print(f"\n\n{'Per-question breakdown':}")
    print("-" * 70)
    print(f"{'Q':>3}  {'Difficulty':<8}  {'M1 ans':>6}  {'M2 ans':>6}  {'M3 ans':>6}  Question")
    print("-" * 70)

    for i in range(len(results["Method1"])):
        r1 = results["Method1"][i]
        r2 = results["Method2"][i]
        r3 = results["Method3"][i]

        def mark(correct): return "  ✓  " if correct else "  ✗  "

        print(f"Q{r1['question_id']:02d}  {r1['difficulty']:<8}  "
              f"{mark(r1['answer_correct'])}  "
              f"{mark(r2['answer_correct'])}  "
              f"{mark(r3['answer_correct'])}  "
              f"{r1['question'][:45]}")

    # --- Key findings ---
    print("\n" + "=" * 70)
    print("  KEY FINDINGS")
    print("=" * 70)

    m1_acc = sum(r["answer_correct"] for r in results["Method1"])
    m2_acc = sum(r["answer_correct"] for r in results["Method2"])
    m3_acc = sum(r["answer_correct"] for r in results["Method3"])
    total  = len(results["Method1"])

    print(f"\n  Method 1 answered correctly: {m1_acc}/{total}")
    print(f"  Method 2 answered correctly: {m2_acc}/{total}  (+{m2_acc - m1_acc} over Method 1)")
    print(f"  Method 3 answered correctly: {m3_acc}/{total}  (+{m3_acc - m1_acc} over Method 1, "
          f"{'+' if m3_acc >= m2_acc else ''}{m3_acc - m2_acc} vs Method 2)")

    hard_m1 = sum(r["answer_correct"] for r in results["Method1"] if r["difficulty"] == "hard")
    hard_m3 = sum(r["answer_correct"] for r in results["Method3"] if r["difficulty"] == "hard")
    hard_total = sum(1 for r in results["Method1"] if r["difficulty"] == "hard")
    print(f"\n  Hard questions — Method 1: {hard_m1}/{hard_total},  Method 3: {hard_m3}/{hard_total}")
    print(f"  (Hard questions are where the intent key is most expected to help)")
    print()
