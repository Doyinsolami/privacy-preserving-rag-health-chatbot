import json
from pathlib import Path

from transformers import AutoTokenizer


DATA_DIR = Path(__file__).resolve().parent / "data" / "geia"
MODEL_NAME = "gpt2"
SPLITS = ("train", "validation", "test")
THRESHOLDS = (40, 64, 96, 128, 160, 256)


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
        "coverage": {
            str(limit): sum(length <= limit for length in lengths) / total
            for limit in THRESHOLDS
        },
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    report = {"tokenizer": MODEL_NAME, "splits": {}}
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
            lengths.append(len(tokenizer.encode(text, add_special_tokens=False)))

        report["splits"][split] = summarize(lengths)
        all_lengths.extend(lengths)

    report["all_notes"] = summarize(all_lengths)
    output_path = DATA_DIR / "token_length_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    overall = report["all_notes"]
    print(f"Tokenizer: {MODEL_NAME}")
    print(f"Notes analyzed: {overall['notes']}")
    print(
        "Token lengths: "
        f"min={overall['minimum']}, mean={overall['mean']:.1f}, "
        f"median={overall['median']}, p90={overall['p90']}, "
        f"p95={overall['p95']}, p99={overall['p99']}, "
        f"max={overall['maximum']}"
    )
    print("\nPercentage of notes fully preserved:")
    for limit, fraction in overall["coverage"].items():
        print(f"  {limit} tokens: {fraction:.1%}")
    print(f"\nSaved full report to {output_path}")


if __name__ == "__main__":
    main()
