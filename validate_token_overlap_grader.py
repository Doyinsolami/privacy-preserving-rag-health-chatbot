import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
REVIEW_FILE = DATA_DIR / "answer_accuracy_manual_reviews.json"

# Overlap threshold: fraction of the fact's significant words that
# must appear in the answer for it to be graded correct.
OVERLAP_THRESHOLD = 0.8

STOPWORDS = {
    "a", "an", "the", "of", "with", "in", "to", "and", "or", "for",
    "on", "at", "as", "by", "is", "was", "using", "no", "not",
}


def strip_tag(value):
    return re.sub(r"\s*\([^)]*\)\s*$", "", value).strip()


def significant_words(value):
    core = strip_tag(value).lower()
    # Decimal numbers (e.g. "0.4") are matched as one token first, so
    # a dosage isn't split apart; everything else is matched as plain
    # alphanumeric runs, which also means a sentence-ending period
    # right after a word does NOT get glued onto that word.
    words = re.findall(r"\d+\.\d+|[a-z0-9]+", core)
    return {w for w in words if w not in STOPWORDS and len(w) > 1}


def token_overlap_correct(fact, answer, threshold=OVERLAP_THRESHOLD):
    """
    Grade correct if most of the fact's significant words appear
    somewhere in the answer. Order-independent, tolerant of the
    model rephrasing around the fact, unlike plain substring match.
    """
    fact_words = significant_words(fact)
    if not fact_words:
        return False

    answer_words = significant_words(answer)
    overlap = len(fact_words & answer_words) / len(fact_words)
    return overlap >= threshold


def main():
    with open(REVIEW_FILE, "r", encoding="utf-8") as f:
        reviews = json.load(f)

    manual_reviews = [r for r in reviews if r["grading_method"] == "manual_review"]

    agreements = 0
    disagreements = []

    for r in manual_reviews:
        auto_correct = token_overlap_correct(r["expected_fact"], r["generated_answer"])
        manual_correct = r["correct"]

        if auto_correct == manual_correct:
            agreements += 1
        else:
            disagreements.append({
                "question": r["question"],
                "expected_fact": r["expected_fact"],
                "generated_answer": r["generated_answer"],
                "manual_judgment": manual_correct,
                "automatic_judgment": auto_correct,
            })

    total = len(manual_reviews)
    agreement_rate = agreements / total if total else 0.0

    print(f"Compared automatic grader against {total} manually graded answers")
    print(f"Agreement: {agreements}/{total} = {agreement_rate:.3f}")

    if disagreements:
        print(f"\n{len(disagreements)} disagreements:")
        for d in disagreements:
            print(f"\n  Question: {d['question']}")
            print(f"  Expected: {d['expected_fact']}")
            print(f"  Answer: {d['generated_answer']}")
            print(f"  Manual said: {'correct' if d['manual_judgment'] else 'incorrect'}")
            print(f"  Automatic said: {'correct' if d['automatic_judgment'] else 'incorrect'}")


if __name__ == "__main__":
    main()
