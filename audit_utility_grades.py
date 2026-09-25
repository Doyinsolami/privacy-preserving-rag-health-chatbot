"""Manually audit a small reproducible sample of automatic utility grades.

Pass one or more utility_v2 answer JSON files. Existing audit decisions are
cached, so interruption is safe. The script reports raw agreement and Cohen's
kappa between human and automatic labels.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("answer_files", nargs="+", type=Path)
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=Path("data/results/utility_v2_grading_audit.json"))
    return parser.parse_args()


def kappa(rows):
    if not rows:
        return None
    observed = sum(r["automatic_correct"] == r["human_correct"] for r in rows) / len(rows)
    auto_yes = sum(r["automatic_correct"] for r in rows) / len(rows)
    human_yes = sum(r["human_correct"] for r in rows) / len(rows)
    expected = auto_yes * human_yes + (1 - auto_yes) * (1 - human_yes)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def main():
    args = parse_args()
    pool = []
    for path in args.answer_files:
        for row in json.loads(path.read_text(encoding="utf-8")):
            if row.get("correct") is None:
                continue
            pool.append({**row, "source_file": str(path)})
    if not pool:
        raise SystemExit("No graded answers found")

    rng = random.Random(args.seed)
    strata = {}
    for row in pool:
        key = (row.get("field"), bool(row.get("correct")))
        strata.setdefault(key, []).append(row)
    for rows in strata.values():
        rng.shuffle(rows)

    sample = []
    keys = sorted(strata, key=str)
    while len(sample) < min(args.sample_size, len(pool)):
        added = False
        for key in keys:
            if strata[key] and len(sample) < args.sample_size:
                sample.append(strata[key].pop())
                added = True
        if not added:
            break

    prior = {}
    if args.output.exists():
        prior = {
            (r["source_file"], r["question_id"]): r
            for r in json.loads(args.output.read_text(encoding="utf-8"))
        }
    audited = []
    for index, row in enumerate(sample, 1):
        key = (row["source_file"], row["question_id"])
        if key in prior:
            audited.append(prior[key])
            continue
        separator = " AND " if row.get("expected_answer_match") == "all" else " OR "
        print(f"\n[{index}/{len(sample)}]")
        print("Question:", row["question"])
        print("Expected:", separator.join(row["expected_answers"]))
        print("Answer:", row["generated_answer"])
        print("Automatic grade:", row["correct"], f"({row['grading_method']})")
        while True:
            decision = input("Your independent grade [y/n]: ").strip().lower()
            if decision in {"y", "n"}:
                break
        item = {
            "source_file": row["source_file"],
            "question_id": row["question_id"],
            "field": row["field"],
            "question": row["question"],
            "expected_answers": row["expected_answers"],
            "generated_answer": row["generated_answer"],
            "grading_method": row["grading_method"],
            "automatic_correct": bool(row["correct"]),
            "human_correct": decision == "y",
        }
        audited.append(item)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(audited, indent=2), encoding="utf-8")

    agreement = sum(
        row["automatic_correct"] == row["human_correct"] for row in audited
    ) / len(audited)
    print(f"\nAudited: {len(audited)}")
    print(f"Raw agreement: {agreement:.3f}")
    print(f"Cohen's kappa: {kappa(audited):.3f}")
    print("Methods:", dict(Counter(row["grading_method"] for row in audited)))
    print(f"Saved audit: {args.output}")


if __name__ == "__main__":
    main()
