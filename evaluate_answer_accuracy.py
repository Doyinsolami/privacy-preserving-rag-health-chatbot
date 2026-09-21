
import json
import re
from pathlib import Path

from generate import generate_answer
from ingest import collection as clean_collection, get_collection, search
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
        grade = input("Is the generated answer factually correct? [y/n]: ").strip().lower()
        if grade == "y":
            return True
        if grade == "n":
            return False
        print("Please enter y or n.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection",
        default=None,
        help="Chroma collection to evaluate. Omit for the clean baseline.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional filename tag. Defaults to the collection name.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.collection:
        target_collection = get_collection(args.collection)
        tag = args.tag if args.tag else args.collection
        result_filename = f"utility_{tag}.json"
        review_filename = f"answer_accuracy_manual_reviews_{tag}.json"
    else:
        target_collection = clean_collection
        result_filename = "baseline_no_dp.json"
        review_filename = "answer_accuracy_manual_reviews.json"

    print(
        f"Selected Chroma collection: {target_collection.name} "
        f"({target_collection.count()} records)"
    )

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
            use_collection=target_collection,
        )
        retrieved_ids = results["ids"][0]
        retrieved_docs = results["documents"][0]

        if q["correct_encounter_id"] not in retrieved_ids:
            skipped += 1
            graded_answers.append(
                {
                    "question": q["question"],
                    "expected_fact": q["fact"],
                    "generated_answer": None,
                    "correct": False,
                    "grading_method": "retrieval_miss",
                    "correct_encounter_id": q["correct_encounter_id"],
                }
            )
            continue

        answer = generate_answer(q["question"], retrieved_docs)
        total_graded += 1
        correct = manually_grade(q["question"], q["fact"], answer)

        graded_answers.append(
            {
                "question": q["question"],
                "expected_fact": q["fact"],
                "generated_answer": answer,
                "correct": correct,
                "grading_method": "manual_review",
                "correct_encounter_id": q["correct_encounter_id"],
            }
        )

        if correct:
            correct_count += 1
        else:
            wrong.append({"fact": q["fact"], "answer": answer})

    conditional_accuracy = correct_count / total_graded if total_graded else 0.0
    end_to_end_accuracy = correct_count / len(questions) if questions else 0.0

    print(f"Total questions: {len(questions)}")
    print(f"Skipped (retrieval missed the note): {skipped}")
    print(f"Graded: {total_graded}")
    print(
        "Conditional generator accuracy (correct / graded): "
        f"{correct_count}/{total_graded} = {conditional_accuracy:.3f}"
    )
    print(
        "End-to-end answer accuracy (correct / all questions): "
        f"{correct_count}/{len(questions)} = {end_to_end_accuracy:.3f}"
    )

    if wrong:
        print(f"\n{len(wrong)} incorrect answers:")
        for item in wrong:
            print(f"  - Expected: {item['fact']}")
            print(f"    Got: {item['answer']}")

    review_path = DATA_DIR / review_filename
    review_path.write_text(json.dumps(graded_answers, indent=2), encoding="utf-8")
    print(f"Saved manual answer reviews to {review_path}")

    result_path = RESULTS_DIR / result_filename
    if not result_path.exists():
        raise FileNotFoundError(
            f"{result_path} does not exist. Run evaluate_retrieval.py against "
            "the same collection first."
        )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["utility"].update(
        {
            "answer_accuracy": end_to_end_accuracy,
            "conditional_generator_accuracy": conditional_accuracy,
            "end_to_end_answer_accuracy": end_to_end_accuracy,
            "answer_accuracy_total_questions": len(questions),
            "answer_accuracy_graded_count": total_graded,
            "answer_accuracy_skipped_count": skipped,
            "answer_accuracy_review_file": review_filename,
        }
    )
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nUpdated {result_path} with conditional and end-to-end accuracy")


if __name__ == "__main__":
    main()
