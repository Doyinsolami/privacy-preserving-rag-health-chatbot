import json
import re
from pathlib import Path

from ingest import search
from generate import generate_answer
from results import RESULTS_DIR

DATA_DIR = Path(__file__).resolve().parent / "data"
EVAL_FILE = DATA_DIR / "eval_questions_accuracy.json"
RETRIEVE_K = 5


def strip_tag(fact):
    return re.sub(r"\s*\([^)]*\)\s*$", "", fact).strip()


def is_correct(fact, answer):
    core = strip_tag(fact).lower()
    return core in answer.lower()


def main():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    total_graded = 0
    correct_count = 0
    skipped = 0
    wrong = []

    for q in questions:
        results = search(
            q["question"],
            n_results=RETRIEVE_K,
            patient_id=q["patient_id"],
        )
        retrieved_ids = results["ids"][0]
        retrieved_docs = results["documents"][0]

        if q["correct_encounter_id"] not in retrieved_ids:
            skipped += 1
            continue

        answer = generate_answer(q["question"], retrieved_docs)
        total_graded += 1

        correct = is_correct(q["fact"], answer)
        if correct:
            correct_count += 1
        else:
            wrong.append({"fact": q["fact"], "answer": answer})

    accuracy = correct_count / total_graded if total_graded else 0.0

    print(f"Total questions: {len(questions)}")
    print(f"Skipped (retrieval missed the note): {skipped}")
    print(f"Graded: {total_graded}")
    print(f"Answer accuracy: {correct_count}/{total_graded} = {accuracy:.3f}")

    if wrong:
        print(f"\n{len(wrong)} incorrect answers:")
        for w in wrong:
            print(f"  - Expected: {w['fact']}")
            print(f"    Got: {w['answer']}")

    result_path = RESULTS_DIR / "baseline_no_dp.json"
    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    result["utility"]["answer_accuracy"] = accuracy
    result["utility"]["answer_accuracy_graded_count"] = total_graded
    result["utility"]["answer_accuracy_skipped_count"] = skipped

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nUpdated {result_path} with answer_accuracy")


if __name__ == "__main__":
    main()