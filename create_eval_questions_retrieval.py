import json
import random
import re
from collections import defaultdict
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_FILE = DATA_DIR / "ground_truth.json"
OUTPUT_FILE = DATA_DIR / "eval_questions_retrieval.json"

QUESTIONS_PER_PATIENT = 2
MAX_QUESTIONS = 40

FACT_FIELDS = [
    "conditions",
    "medications",
    "procedures",
    "allergies",
]

CLUE_FIELDS = [
    "conditions",
    "medications",
    "procedures",
    "allergies",
    "reason_for_visit",
]

SKIP_FACTS = {
    "Medication review due (situation)",
    "Medication reconciliation (procedure)",
}

# Target/clue pairs identified during manual review as describing the
# same underlying clinical concept rather than a distinct, clinically
# related one. Checked in both directions since which fact ends up as
# the "target" vs. the "clue" depends on which is unique for a given
# patient.
BLOCKED_TARGET_CLUE_PAIRS = {
    (
        "Concussion with no loss of consciousness (disorder)",
        "Concussion injury of brain (disorder)",
    ),
    (
        "Laceration of foot (disorder)",
        "Laceration - injury (disorder)",
    ),
    (
        "History of seizure (situation)",
        "Seizure disorder (disorder)",
    ),
    (
        "Chronic pain (finding)",
        "Chronic neck pain (finding)",
    ),
    (
        "Chronic pain (finding)",
        "Chronic low back pain (finding)",
    ),
    (
        "Initial patient assessment (procedure)",
        "History AND physical examination (procedure)",
    ),
    (
        "Implantation of subcutaneous contraceptive (procedure)",
        "Etonogestrel 68 MG Drug Implant",
    ),
    (
        "Removal of supragingival plaque and calculus from all teeth using dental instrument (procedure)",
        "Removal of subgingival plaque and calculus from all teeth using dental instrument (procedure)",
    ),
    (
        "Loose dental filling (finding)",
        "Application of composite dental filling material to dentin of tooth following fracture of tooth (procedure)",
    ),
    (
        "Assessment using Car  Relax  Alone  Forget  Friends  Trouble Screening Test (procedure)",
        "Assessment of substance use (procedure)",
    ),
}


def pair_is_blocked(target_fact, clue_value):
    if (
        (target_fact, clue_value) in BLOCKED_TARGET_CLUE_PAIRS
        or (clue_value, target_fact) in BLOCKED_TARGET_CLUE_PAIRS
    ):
        return True

    # "Laceration" pairs keep recurring with different body parts
    # (foot, forearm, and likely others). Rather than adding a new
    # exact pair every time a new body part shows up, block any
    # target/clue pair where both mention laceration.
    target_core = strip_tag(target_fact).lower()
    clue_core = strip_tag(clue_value).lower()
    if "laceration" in target_core and "laceration" in clue_core:
        return True

    return False

QUESTION_TEMPLATES = {
    "conditions": (
        "What condition was documented in the same encounter as {clue}?"
    ),
    "medications": (
        "What medication was documented in the same encounter as {clue}?"
    ),
    "procedures": (
        "What procedure was documented in the same encounter as {clue}?"
    ),
    "allergies": (
        "What allergy was documented in the same encounter as {clue}?"
    ),
}


def strip_tag(value):
    return re.sub(r"\s*\([^)]*\)\s*$", "", value).strip()


def get_field_values(encounter, field):
    if field == "reason_for_visit":
        value = encounter.get("reason_for_visit")
        return [value] if value else []

    return encounter.get(field, []) or []


def find_safe_clues(target_fact, encounter, all_field_counts):
    """
    Return clinical facts that appear in the target encounter and
    nowhere else in this patient's encounters.
    """
    clues = []
    seen = set()
    target_core = strip_tag(target_fact).lower()

    for field in CLUE_FIELDS:
        for value in get_field_values(encounter, field):
            value_core = strip_tag(value).lower()

            if value == target_fact:
                continue

            if value_core == target_core:
                continue

            if value in seen:
                continue

            seen.add(value)

            if pair_is_blocked(target_fact, value):
                continue

            if all_field_counts[value] == 1:
                clues.append({
                    "field": field,
                    "value": value,
                })

    return clues


def build_question(target_field, source_clue):
    return QUESTION_TEMPLATES[target_field].format(
        clue=strip_tag(source_clue)
    )


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    patients = defaultdict(list)

    for encounter_id, encounter in data.items():
        patients[encounter["patient_id"]].append(
            (encounter_id, encounter)
        )

    patient_items = list(patients.items())
    random.shuffle(patient_items)

    eval_set = []
    no_safe_clue_count = 0

    for patient_id, encounters in patient_items:
        if len(eval_set) >= MAX_QUESTIONS:
            break

        fact_counts = defaultdict(int)
        fact_to_encounter = defaultdict(list)
        fact_to_field = {}

        for encounter_id, encounter in encounters:
            seen = set()

            for field in FACT_FIELDS:
                for fact in encounter.get(field, []) or []:
                    if fact in SKIP_FACTS:
                        continue

                    if fact not in seen:
                        fact_counts[fact] += 1
                        fact_to_encounter[fact].append(encounter_id)
                        fact_to_field[fact] = field
                        seen.add(fact)

        all_field_counts = defaultdict(int)

        for encounter_id, encounter in encounters:
            seen = set()

            for field in CLUE_FIELDS:
                for value in get_field_values(encounter, field):
                    if value not in seen:
                        all_field_counts[value] += 1
                        seen.add(value)

        unique_targets = [
            (fact, fact_to_encounter[fact][0])
            for fact, count in fact_counts.items()
            if count == 1
        ]

        random.shuffle(unique_targets)
        selected_for_patient = 0
        used_encounter_ids = set()

        for fact, encounter_id in unique_targets:
            if selected_for_patient >= QUESTIONS_PER_PATIENT:
                break

            if len(eval_set) >= MAX_QUESTIONS:
                break

            # Allow at most one evaluation question per encounter.
            if encounter_id in used_encounter_ids:
                continue

            encounter = data[encounter_id]

            safe_clues = find_safe_clues(
                fact,
                encounter,
                all_field_counts,
            )

            if not safe_clues:
                no_safe_clue_count += 1
                continue

            selected_clue = random.choice(safe_clues)

            question = build_question(
                target_field=fact_to_field[fact],
                source_clue=selected_clue["value"],
            )

            eval_set.append({
                "question": question,
                "fact": fact,
                "field": fact_to_field[fact],
                "source_clue": selected_clue["value"],
                "source_clue_field": selected_clue["field"],
                "patient_id": patient_id,
                "correct_encounter_id": encounter_id,
                "question_source": "controlled_template",
            })

            used_encounter_ids.add(encounter_id)
            selected_for_patient += 1

    random.shuffle(eval_set)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(eval_set, file, indent=2)

    print(f"Wrote {len(eval_set)} questions to {OUTPUT_FILE}")
    print(f"Skipped (no safe second clue): {no_safe_clue_count}")

    print("\nGenerated questions:")

    for number, item in enumerate(eval_set, start=1):
        print(f"\n{number}. {item['question']}")
        print(f"   Expected answer: {item['fact']}")
        print(f"   Verified clue: {item['source_clue']}")
        print(f"   Encounter ID: {item['correct_encounter_id']}")


if __name__ == "__main__":
    main()