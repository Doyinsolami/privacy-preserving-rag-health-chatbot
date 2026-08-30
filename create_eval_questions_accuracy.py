import json
import random
from collections import defaultdict
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_FILE = DATA_DIR / "ground_truth.json"
OUTPUT_FILE = DATA_DIR / "eval_questions_accuracy.json"

QUESTIONS_PER_PATIENT = 2
MAX_QUESTIONS = 40

FACT_FIELDS = ["conditions", "medications", "procedures", "allergies"]
SKIP_FACTS = {
    "Medication review due (situation)",
    "Medication reconciliation (procedure)",
}

QUESTION_TEMPLATES = {
    "conditions": "What condition was documented during this patient's {encounter_type} on {date}?",
    "medications": "What medication was documented during this patient's {encounter_type} on {date}?",
    "procedures": "What procedure was documented during this patient's {encounter_type} on {date}?",
    "allergies": "What allergy was documented during this patient's {encounter_type} on {date}?",
}


def build_question(field, encounter_type, date):
    return QUESTION_TEMPLATES[field].format(
        encounter_type=encounter_type.lower(), date=date
    )


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    patients = defaultdict(list)
    for enc_id, enc in data.items():
        patients[enc["patient_id"]].append((enc_id, enc))

    eval_set = []

    for pid, encs in patients.items():
        fact_counts = defaultdict(int)
        fact_to_enc = defaultdict(list)
        fact_to_field = {}
        for enc_id, enc in encs:
            seen = set()
            for field in FACT_FIELDS:
                for fact in enc.get(field, []) or []:
                    if fact in SKIP_FACTS:
                        continue
                    if fact not in seen:
                        fact_counts[fact] += 1
                        fact_to_enc[fact].append(enc_id)
                        fact_to_field[fact] = field
                        seen.add(fact)

        uniques = [(fact, fact_to_enc[fact][0])
                   for fact, c in fact_counts.items() if c == 1]

        if not uniques:
            continue

        random.shuffle(uniques)

        seen_combos = set()
        selected = []
        for fact, enc_id in uniques:
            field = fact_to_field[fact]
            combo = (enc_id, field)
            if combo in seen_combos:
                continue
            seen_combos.add(combo)
            selected.append((fact, enc_id))
            if len(selected) >= QUESTIONS_PER_PATIENT:
                break

        for fact, enc_id in selected:
            field = fact_to_field[fact]
            enc = data[enc_id]
            eval_set.append({
                "question": build_question(
                    field, enc["encounter_type"], enc["encounter_date"]
                ),
                "fact": fact,
                "field": field,
                "patient_id": pid,
                "correct_encounter_id": enc_id,
            })

    random.shuffle(eval_set)
    eval_set = eval_set[:MAX_QUESTIONS]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2)

    print(f"Wrote {len(eval_set)} questions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()