import json
import random
import numpy as np
from pathlib import Path
from ingest import collection

random.seed(42)

test_path = Path(__file__).resolve().parent / "data" / "geia" / "test.jsonl"
records = [json.loads(line) for line in open(test_path)]

by_patient = {}
for r in records:
    by_patient.setdefault(r["patient_id"], []).append(r)

patient_ids = list(by_patient.keys())
random.shuffle(patient_ids)
sample_patients = patient_ids[:30]

encounter_ids = [by_patient[p][0]["encounter_id"] for p in sample_patients]

result = collection.get(ids=encounter_ids, include=["embeddings"])

fetched_ids = result["ids"]
embeddings = np.array(result["embeddings"])

print(f"Requested {len(encounter_ids)} encounter embeddings, got {len(fetched_ids)} back")

if len(fetched_ids) < 2:
    print("Could not fetch enough embeddings, check that encounter_id matches the IDs used in Chroma")
else:
    norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sims = norm @ norm.T
    n = len(fetched_ids)
    pairwise = [sims[i][j] for i in range(n) for j in range(i + 1, n)]
    pairwise = np.array(pairwise)
    print(f"Pairwise cosine similarity across {n} different patients' embeddings:")
    print(f"mean: {pairwise.mean():.4f}")
    print(f"min: {pairwise.min():.4f}")
    print(f"max: {pairwise.max():.4f}")
    print(f"std: {pairwise.std():.4f}")