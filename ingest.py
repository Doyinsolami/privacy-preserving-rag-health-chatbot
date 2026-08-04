from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="patient_records")


def embed_text(text):
    return model.encode(text).tolist()


def store_document(doc_id, text, metadata=None):
    collection.add(
        ids=[doc_id],
        embeddings=[embed_text(text)],
        documents=[text],
        metadatas=[metadata or {}],
    )


def search(query_text, n_results=2):
    return collection.query(
        query_embeddings=[embed_text(query_text)],
        n_results=n_results,
    )


if __name__ == "__main__":
    placeholder_records = [
        ("patient_001", "The patient was diagnosed with hypertension and prescribed lisinopril.", {"diagnosis": "hypertension"}),
        ("patient_002", "The patient reported symptoms of type 2 diabetes and was started on metformin.", {"diagnosis": "type 2 diabetes"}),
        ("patient_003", "The patient came in for a routine checkup with no new concerns.", {"diagnosis": "none"}),
    ]

    for doc_id, text, metadata in placeholder_records:
        store_document(doc_id, text, metadata)
        print(f"Stored {doc_id}")

    query = "What condition does the patient have?"
    results = search(query, n_results=2)

    print(f"\nQuery: {query}")
    for i, (doc, doc_id) in enumerate(zip(results["documents"][0], results["ids"][0]), start=1):
        print(f"{i}. [{doc_id}] {doc}")