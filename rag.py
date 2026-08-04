from ingest import store_document, search
from generate import generate_answer

if __name__ == "__main__":
    placeholder_records = [
        ("patient_001", "The patient was diagnosed with hypertension and prescribed lisinopril.", {"diagnosis": "hypertension"}),
        ("patient_002", "The patient reported symptoms of type 2 diabetes and was started on metformin.", {"diagnosis": "type 2 diabetes"}),
        ("patient_003", "The patient came in for a routine checkup with no new concerns.", {"diagnosis": "none"}),
    ]

    for doc_id, text, metadata in placeholder_records:
        store_document(doc_id, text, metadata)

    question = "Which patient was prescribed metformin?"
    results = search(question, n_results=2)
    retrieved_docs = results["documents"][0]

    answer = generate_answer(question, retrieved_docs)

    print(f"Question: {question}\n")
    print("Retrieved records:")
    for doc in retrieved_docs:
        print(f"- {doc}")
    print(f"\nAnswer:\n{answer}")