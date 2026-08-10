from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="encounter_notes")


def embed_text(text):
    return model.encode(text).tolist()


def store_document(doc_id, text, metadata=None):
    collection.upsert(
        ids=[doc_id],
        embeddings=[embed_text(text)],
        documents=[text],
        metadatas=[metadata or {}],
    )


def search(query_text, n_results=2, patient_id=None):
    where = {"patient_id": patient_id} if patient_id else None
    return collection.query(
        query_embeddings=[embed_text(query_text)],
        n_results=n_results,
        where=where,
    )