import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from generate import generate_answer
from ingest import collection as clean_collection, get_collection, search
from results import RESULTS_DIR

DATA_DIR = Path(__file__).resolve().parent / "data"
EVAL_FILE = DATA_DIR / "eval_questions_new.json"
K = 5


def normalize(text):
    text = str(text).lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9.]+", " ", text)
    return " ".join(text.split())


def token_f1(expected, actual):
    e, a = normalize(expected).split(), normalize(actual).split()
    if not e or not a:
        return 0.0
    ec, ac = Counter(e), Counter(a)
    overlap = sum((ec & ac).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(a), overlap / len(e)
    return 2 * precision * recall / (precision + recall)


def automatic_grade(question, expected_answers, answer):
    answer_norm = normalize(answer)
    if question.get("expected_answer_type") == "negative":
        negative_patterns = (
            r"\bno\b.*\b(documented|recorded|listed|prescribed|performed|found)\b",
            r"\b(does not|do not|did not|doesn't|don't|didn't)\b.*\b(contain|show|document|list|mention|indicate)\b",
            r"\brecords do not contain this information\b",
            r"\bnot (explicitly )?(stated|documented|recorded|listed|mentioned|provided)\b",
            r"\b(no|without) (record|evidence|mention|documentation) of\b",
        )
        if any(re.search(pattern, answer_norm) for pattern in negative_patterns):
            return True, "negative_answer_rule", 1.0
        return None, "manual_negative_review_required", 0.0
    if answer_norm == normalize("The records do not contain this information."):
        return False, "explicit_no_information", 0.0
    scores = []
    contained_results = []
    for expected in expected_answers:
        expected_norm = normalize(expected)
        contained = bool(expected_norm) and expected_norm in answer_norm
        contained_results.append(contained)
        score = token_f1(expected, answer)
        scores.append(score)
    if question.get("expected_answer_match", "any") == "all":
        if contained_results and all(contained_results):
            return True, "all_expected_answers_contained", min(scores)
        return None, "manual_multi_answer_review_required", max(scores, default=0.0)
    for contained, score in zip(contained_results, scores):
        if contained:
            return True, "normalized_containment", score
    best = max(scores, default=0.0)
    # High threshold catches minor formatting differences while avoiding broad
    # semantic guesses. Everything else is manually adjudicated once and cached.
    if best >= 0.85:
        return True, "high_token_f1", best
    return None, "manual_review_required", best


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--collection", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--non-interactive", action="store_true",
                   help="Save uncertain items as pending instead of asking for a grade.")
    p.add_argument("--limit", type=int, default=None, help="Smoke-test on the first N questions.")
    return p.parse_args()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    target = get_collection(args.collection) if args.collection else clean_collection
    tag = args.tag or (args.collection if args.collection else "clean")
    questions = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[:args.limit]

    cache_path = DATA_DIR / "results" / f"utility_new_answers_{tag}.json"
    cached = {}
    if cache_path.exists():
        cached = {x["question_id"]: x for x in json.loads(cache_path.read_text(encoding="utf-8"))}

    rows = []
    for number, q in enumerate(questions, 1):
        previous = cached.get(q["question_id"], {})
        results = search(q["question"], n_results=K, patient_id=q["patient_id"], use_collection=target)
        ids = results["ids"][0]
        docs = results["documents"][0]
        relevant = set(q["relevant_encounter_ids"])
        rank = next((i for i, encounter_id in enumerate(ids, 1) if encounter_id in relevant), None)

        answer = previous.get("generated_answer")
        if answer is None:
            print(f"[{number}/{len(questions)}] Generating: {q['question']}")
            answer = generate_answer(q["question"], docs)

        correct, method, similarity = automatic_grade(q, q["expected_answers"], answer)
        if correct is None and previous.get("manual_correct") is not None:
            correct = bool(previous["manual_correct"])
            method = "cached_manual_review"
        elif correct is None and not args.non_interactive:
            print("\nQuestion:", q["question"])
            print("Expected:", " OR ".join(q["expected_answers"]))
            print("Answer:", answer)
            while True:
                decision = input("Factually correct? [y/n]: ").strip().lower()
                if decision in {"y", "n"}:
                    correct = decision == "y"
                    method = "manual_review"
                    break

        row = {
            **q,
            "retrieved_ids": ids,
            "relevant_rank": rank,
            "retrieval_hit_at_1": bool(rank and rank <= 1),
            "retrieval_hit_at_3": bool(rank and rank <= 3),
            "retrieval_hit_at_5": bool(rank and rank <= 5),
            "generated_answer": answer,
            "correct": correct,
            "manual_correct": correct if method in {"manual_review", "cached_manual_review"} else previous.get("manual_correct"),
            "grading_method": method,
            "best_token_f1": similarity,
        }
        rows.append(row)
        save_json(cache_path, rows)

    total = len(rows)
    judged = [r for r in rows if r["correct"] is not None]
    correct_rows = [r for r in judged if r["correct"]]
    hit1 = sum(r["retrieval_hit_at_1"] for r in rows)
    hit3 = sum(r["retrieval_hit_at_3"] for r in rows)
    hit5 = sum(r["retrieval_hit_at_5"] for r in rows)
    rr = sum(1 / r["relevant_rank"] for r in rows if r["relevant_rank"])

    category = defaultdict(lambda: {"n": 0, "hit5": 0, "judged": 0, "correct": 0})
    for r in rows:
        c = category[r["field"]]
        c["n"] += 1
        c["hit5"] += int(r["retrieval_hit_at_5"])
        if r["correct"] is not None:
            c["judged"] += 1
            c["correct"] += int(r["correct"])

    report = {
        "benchmark": "new_v1",
        "collection": target.name,
        "tag": tag,
        "questions": total,
        "retrieval": {
            "recall_at_1": hit1 / total if total else 0,
            "recall_at_3": hit3 / total if total else 0,
            "recall_at_5": hit5 / total if total else 0,
            "mrr_at_5": rr / total if total else 0,
        },
        "answers": {
            "judged": len(judged),
            "pending_manual_review": total - len(judged),
            "end_to_end_accuracy": len(correct_rows) / total if total else 0,
            "accuracy_among_judged": len(correct_rows) / len(judged) if judged else None,
        },
        "by_category": {},
    }
    for name, c in sorted(category.items()):
        report["by_category"][name] = {
            **c,
            "recall_at_5": c["hit5"] / c["n"] if c["n"] else 0,
            "answer_accuracy": c["correct"] / c["judged"] if c["judged"] else None,
        }

    result_path = RESULTS_DIR / f"utility_new_{tag}.json"
    save_json(result_path, report)
    print(json.dumps(report, indent=2))
    print(f"Saved answers: {cache_path}")
    print(f"Saved summary: {result_path}")


if __name__ == "__main__":
    main()
zip -r myproject.zip . -x "__pycache__/*" -x "app/*" -x "chroma_db/*" -x "data/*" -x "models/*" -x "results/*" -x "tests/*" -x "venv/*"