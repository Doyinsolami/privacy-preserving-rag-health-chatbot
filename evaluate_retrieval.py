import json
from pathlib import Path

from ingest import search
from results import save_result

DATA_DIR = Path(__file__).resolve().parent / "data"
EVAL_FILE = DATA_DIR / "eval_questions_retrieval.json"

K_VALUES = [1, 3, 5]
RETRIEVE_K = max(K_VALUES)


def main():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    hits = {k: 0 for k in K_VALUES}
    reciprocal_ranks = []
    total = len(questions)
    misses = []

    for q in questions:
        results = search(
            q["question"],
            n_results=RETRIEVE_K,
            patient_id=q["patient_id"],
        )
        retrieved_ids = results["ids"][0]
        correct = q["correct_encounter_id"]

        for k in K_VALUES:
            if correct in retrieved_ids[:k]:
                hits[k] += 1

        if correct in retrieved_ids:
            rank = retrieved_ids.index(correct) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

        if correct not in retrieved_ids[:max(K_VALUES)]:
            misses.append(q)

    mrr = sum(reciprocal_ranks) / total

    print(f"Evaluated {total} questions\n")
    for k in K_VALUES:
        print(f"Recall@{k}: {hits[k]}/{total} = {hits[k]/total:.3f}")
    print(f"MRR: {mrr:.3f}")

    if misses:
        print(f"\n{len(misses)} questions missed at top-{max(K_VALUES)}:")
        for q in misses:
            print(f"  - {q['fact']}")

    save_result(
        run_id="baseline_no_dp",
        config={"dp_applied": False, "epsilon": None, "mechanism": None},
        utility={
            "recall_at_1": hits[1] / total,
            "recall_at_3": hits[3] / total,
            "recall_at_5": hits[5] / total,
            "mrr": mrr,
            "answer_accuracy": None,
        },
    )


if __name__ == "__main__":
    main()