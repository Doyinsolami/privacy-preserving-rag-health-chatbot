from pathlib import Path

from ingest import model, collection

NOTES_DIR = Path("data/notes")
BATCH_SIZE = 256


def load_notes():
    ids = []
    texts = []
    for note_path in sorted(NOTES_DIR.glob("*.txt")):
        ids.append(note_path.stem)
        texts.append(note_path.read_text(encoding="utf-8"))
    return ids, texts


def main():
    ids, texts = load_notes()
    print(f"Loaded {len(ids)} notes")

    for start in range(0, len(ids), BATCH_SIZE):
        batch_ids = ids[start:start + BATCH_SIZE]
        batch_texts = texts[start:start + BATCH_SIZE]
        embeddings = model.encode(batch_texts).tolist()
        collection.upsert(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
        )
        print(f"Embedded {start + len(batch_ids)} / {len(ids)}")

    print(f"Done. Collection now holds {collection.count()} documents")


if __name__ == "__main__":
    main()