import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "geia"
MODEL_NAME = "all-MiniLM-L6-v2"
SPLITS = ("train", "validation", "test")


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from error


def percentile(sorted_values, percentage):
    if not sorted_values:
        return 0
    index = round((len(sorted_values) - 1) * percentage)
    return sorted_values[index]


def summarize(lengths):
    lengths = sorted(lengths)
    total = len(lengths)
    return {
        "notes": total,
        "minimum": lengths[0],
        "mean": sum(lengths) / total,
        "median": percentile(lengths, 0.50),
        "p90": percentile(lengths, 0.90),
        "p95": percentile(lengths, 0.95),
        "p99": percentile(lengths, 0.99),
        "maximum": lengths[-1],
    }


def get_tokenizer_and_max_length(model):
    try:
        tokenizer = model.tokenizer
    except AttributeError:
        tokenizer = model[0].tokenizer
    max_length = model.max_seq_length
    return tokenizer, max_length


def main():
    model = SentenceTransformer(MODEL_NAME)
    tokenizer, max_length = get_tokenizer_and_max_length(model)

    report = {
        "tokenizer": MODEL_NAME,
        "max_seq_length": max_length,
        "splits": {},
        "truncated_encounters": [],
    }
    all_lengths = []

    for split in SPLITS:
        path = DATA_DIR / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing GEIA split: {path}")

        lengths = []
        for record in load_jsonl(path):
            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"A record in {path} has missing or empty text")
            token_ids = tokenizer.encode(text, add_special_tokens=True)
            length = len(token_ids)
            lengths.append(length)
            if length > max_length:
                report["truncated_encounters"].append({
                    "encounter_id": record["encounter_id"],
                    "patient_id": record.get("patient_id"),
                    "split": split,
                    "token_length": length,
                })

        report["splits"][split] = summarize(lengths)
        all_lengths.extend(lengths)

    report["all_notes"] = summarize(all_lengths)
    n_truncated = len(report["truncated_encounters"])
    report["n_truncated"] = n_truncated
    report["fraction_truncated"] = n_truncated / len(all_lengths)

    output_path = DATA_DIR / "minilm_token_length_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    overall = report["all_notes"]
    print(f"Tokenizer: {MODEL_NAME}")
    print(f"Model max_seq_length: {max_length}")
    print(f"Notes analyzed: {overall['notes']}")
    print(
        "Token lengths (including special tokens): "
        f"min={overall['minimum']}, mean={overall['mean']:.1f}, "
        f"median={overall['median']}, p90={overall['p90']}, "
        f"p95={overall['p95']}, p99={overall['p99']}, "
        f"max={overall['maximum']}"
    )
    print(f"\nNotes exceeding max_seq_length ({max_length}) and therefore truncated "
          f"during embedding: {n_truncated} / {len(all_lengths)} "
          f"({report['fraction_truncated']:.2%})")

    if n_truncated == 0:
        print("\nNo notes are truncated by MiniLM's own tokenizer. Every embedding "
              "in the vector store represents the full note text.")
    else:
        print("\nThe following encounters are truncated when embedded, meaning "
              "both retrieval and the inversion attack are working from a partial "
              "view of these notes:")
        for entry in report["truncated_encounters"][:10]:
            print(f"  {entry['encounter_id']} ({entry['split']}): "
                  f"{entry['token_length']} tokens")
        if n_truncated > 10:
            print(f"  ... and {n_truncated - 10} more, see {output_path}")

    print(f"\nSaved full report to {output_path}")


if __name__ == "__main__":
    main()
