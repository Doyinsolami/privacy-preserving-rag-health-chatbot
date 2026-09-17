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


def manually_grade(question, fact, answer):
    print("\n" + "=" * 70)
    print(f"Question: {question}")
    print(f"Expected fact: {fact}")
    print(f"Generated answer: {answer}")

    while True:
        grade = input(
            "Is the generated answer factually correct? [y/n]: "
        ).strip().lower()

        if grade == "y":
            return True

        if grade == "n":
            return False

        print("Please enter y or n.")


def main():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    total_graded = 0
    correct_count = 0
    skipped = 0
    wrong = []
    graded_answers = []

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
            graded_answers.append({
                "question": q["question"],
                "expected_fact": q["fact"],
                "generated_answer": None,
                "correct": False,
                "grading_method": "retrieval_miss",
                "correct_encounter_id": q["correct_encounter_id"],
            })
            continue

        answer = generate_answer(q["question"], retrieved_docs)
        total_graded += 1

        correct = manually_grade(
            question=q["question"],
            fact=q["fact"],
            answer=answer,
        )

        graded_answers.append({
            "question": q["question"],
            "expected_fact": q["fact"],
            "generated_answer": answer,
            "correct": correct,
            "grading_method": "manual_review",
            "correct_encounter_id": q["correct_encounter_id"],
        })

        if correct:
            correct_count += 1
        else:
            wrong.append({"fact": q["fact"], "answer": answer})

    conditional_accuracy = correct_count / total_graded if total_graded else 0.0
    end_to_end_accuracy = correct_count / len(questions) if questions else 0.0

    print(f"Total questions: {len(questions)}")
    print(f"Skipped (retrieval missed the note): {skipped}")
    print(f"Graded: {total_graded}")
    print(f"Conditional generator accuracy (correct / graded): {correct_count}/{total_graded} = {conditional_accuracy:.3f}")
    print(f"End-to-end answer accuracy (correct / all questions): {correct_count}/{len(questions)} = {end_to_end_accuracy:.3f}")

    total_incorrect = len(questions) - correct_count
    print(f"Total end-to-end incorrect: {total_incorrect}")

    if wrong:
        print(f"\n{len(wrong)} incorrect answers:")
        for w in wrong:
            print(f"  - Expected: {w['fact']}")
            print(f"    Got: {w['answer']}")

    review_path = DATA_DIR / "answer_accuracy_manual_reviews.json"
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(graded_answers, f, indent=2)
    print(f"Saved manual answer reviews to {review_path}")

    result_path = RESULTS_DIR / "baseline_no_dp.json"
    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    result["utility"]["answer_accuracy"] = end_to_end_accuracy
    result["utility"]["conditional_generator_accuracy"] = conditional_accuracy
    result["utility"]["end_to_end_answer_accuracy"] = end_to_end_accuracy
    result["utility"]["answer_accuracy_total_questions"] = len(questions)
    result["utility"]["answer_accuracy_graded_count"] = total_graded
    result["utility"]["answer_accuracy_skipped_count"] = skipped

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nUpdated {result_path} with conditional and end-to-end accuracy")


if __name__ == "__main__":
    main()