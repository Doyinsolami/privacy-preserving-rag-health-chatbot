import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/synthea_csv")
NOTES_DIR = Path("data/notes")
GROUND_TRUTH_PATH = Path("data/ground_truth.json")

patients = pd.read_csv(DATA_DIR / "patients.csv")
encounters = pd.read_csv(DATA_DIR / "encounters.csv")
conditions = pd.read_csv(DATA_DIR / "conditions.csv")
medications = pd.read_csv(DATA_DIR / "medications.csv")
procedures = pd.read_csv(DATA_DIR / "procedures.csv")

def format_name(row):
    parts = [str(row[field]) for field in ("FIRST", "LAST") if pd.notna(row[field])]
    return " ".join(parts) if parts else "Unknown"


patient_names = {row["Id"]: format_name(row) for _, row in patients.iterrows()}


def entities_for_encounter(table, encounter_id):
    rows = table[table["ENCOUNTER"] == encounter_id]
    return rows["DESCRIPTION"].dropna().astype(str).drop_duplicates().tolist()


def build_note(patient_name, encounter, conditions_list, medications_list, procedures_list):
    lines = [
        f"Patient: {patient_name}",
        f"Encounter date: {encounter['START']}",
        f"Encounter type: {encounter['DESCRIPTION']}",
    ]
    if conditions_list:
        lines.append("Assessment: " + "; ".join(conditions_list) + ".")
    if procedures_list:
        lines.append("Procedures performed: " + "; ".join(procedures_list) + ".")
    if medications_list:
        lines.append("Medications prescribed: " + "; ".join(medications_list) + ".")
    return "\n".join(lines)


def main():
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    ground_truth = {}

    for _, encounter in encounters.iterrows():
        encounter_id = encounter["Id"]
        patient_id = encounter["PATIENT"]

        conditions_list = entities_for_encounter(conditions, encounter_id)
        medications_list = entities_for_encounter(medications, encounter_id)
        procedures_list = entities_for_encounter(procedures, encounter_id)

        if not (conditions_list or medications_list or procedures_list):
            continue

        patient_name = patient_names.get(patient_id, "Unknown")
        note = build_note(
            patient_name,
            encounter,
            conditions_list,
            medications_list,
            procedures_list,
        )

        note_path = NOTES_DIR / f"{encounter_id}.txt"
        note_path.write_text(note, encoding="utf-8")

        ground_truth[encounter_id] = {
            "patient_id": patient_id,
            "encounter_date": encounter["START"],
            "encounter_type": encounter["DESCRIPTION"],
            "conditions": conditions_list,
            "medications": medications_list,
            "procedures": procedures_list,
        }

    GROUND_TRUTH_PATH.write_text(
        json.dumps(ground_truth, indent=2), encoding="utf-8"
    )

    print(f"Wrote {len(ground_truth)} notes to {NOTES_DIR}")
    print(f"Wrote ground truth to {GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    main()