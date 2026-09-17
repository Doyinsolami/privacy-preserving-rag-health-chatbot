import argparse
import json
from pathlib import Path

import numpy as np
import torch
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from attack_common import fetch_embeddings, load_ground_truth, load_split, note_terms
from results import save_result
from train_attacker import Projection, build_prefix, PREFIX_LENGTH, VICTIM_MODEL

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
NOTES_DIR = DATA_DIR / "notes"
GEIA_DIR = DATA_DIR / "geia"
MODEL_DIR = PROJECT_ROOT / "models" / "attacker"

MAX_NEW_TOKENS = 200
NUM_EXAMPLES_TO_SAVE = 3
NUM_CONSISTENCY_CHECKS = 5
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_decoder():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    attacker = AutoModelForCausalLM.from_pretrained(MODEL_DIR / "gpt2").to(DEVICE).eval()
    projection = Projection().to(DEVICE)
    projection.load_state_dict(torch.load(MODEL_DIR / "projection.pt", map_location=DEVICE))
    projection.eval()
    return tokenizer, attacker, projection


def load_split_records(split):
    records = {}
    with open(GEIA_DIR / f"{split}.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            records[record["encounter_id"]] = record
    return records


@torch.no_grad()
def reconstruct(embedding, tokenizer, attacker, projection):
    embed_t = torch.tensor(embedding, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    prefix_embeds = build_prefix(embed_t, projection, attacker, tokenizer, DEVICE)
    output_ids = attacker.generate(
        inputs_embeds=prefix_embeds,
        attention_mask=torch.ones(
            (1, prefix_embeds.size(1)), dtype=torch.long, device=DEVICE
        ),
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=4,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


@torch.no_grad()
def conditioning_sanity_check(embedding_a, embedding_b, tokenizer, attacker, projection):
    vectors = torch.tensor(
        np.stack([embedding_a, embedding_b]), dtype=torch.float32, device=DEVICE
    )
    prefix_embeds = build_prefix(vectors, projection, attacker, tokenizer, DEVICE)
    logits = attacker(inputs_embeds=prefix_embeds).logits[:, -1, :]
    max_difference = torch.max(torch.abs(logits[0] - logits[1])).item()
    print(f"Embedding-conditioning sanity check, max first-token logit difference: "
          f"{max_difference:.8f}")
    if max_difference <= 1e-7:
        raise RuntimeError(
            "Different embeddings produced indistinguishable first-token logits. "
            "Do not trust reconstruction results until conditioning is fixed."
        )


def embedding_consistency_check(target_ids, embeddings, records):
    victim = SentenceTransformer(VICTIM_MODEL, device=str(DEVICE))
    count = min(NUM_CONSISTENCY_CHECKS, len(target_ids))
    texts = [records[eid]["text"] for eid in target_ids[:count]]
    fresh = np.asarray(victim.encode(texts, convert_to_numpy=True), dtype=np.float32)
    stored = np.asarray(embeddings[:count], dtype=np.float32)
    max_difference = float(np.max(np.abs(fresh - stored)))
    cosines = np.sum(fresh * stored, axis=1) / (
        np.linalg.norm(fresh, axis=1) * np.linalg.norm(stored, axis=1)
    )
    min_cosine = float(np.min(cosines))
    print(f"Embedding consistency check on {count} test notes (fresh MiniLM encoding "
          f"vs vectors stored in Chroma): max abs difference {max_difference:.8f}, "
          f"min cosine similarity {min_cosine:.6f}")
    if min_cosine < 0.999:
        raise RuntimeError(
            "Embeddings stored in Chroma do not match fresh MiniLM encodings of the same "
            "notes. The attacker was trained on fresh encodings, so attack results "
            "would not be trustworthy."
        )
    return max_difference, min_cosine


def cross_patient_pairing(target_ids, patient_of, seed):
    if len({patient_of[eid] for eid in target_ids}) < 2:
        raise ValueError(
            "The shuffled control needs targets from at least 2 patients, use a larger --limit"
        )
    rng = np.random.default_rng(seed)
    n = len(target_ids)
    pairing = np.empty(n, dtype=int)
    for i in range(n):
        j = int(rng.integers(n))
        while patient_of[target_ids[j]] == patient_of[target_ids[i]]:
            j = int(rng.integers(n))
        pairing[i] = j
    return pairing


def term_recall(true_terms, reconstructed_text):
    if not true_terms:
        return None
    haystack = reconstructed_text.lower()
    recovered = sum(1 for term in true_terms if term.split(" (")[0].lower() in haystack)
    return recovered / len(true_terms)


def score_against_targets(reconstructions, target_ids, ground_truth, scorer):
    rouge_scores = []
    term_recalls = []
    for eid, reconstructed in zip(target_ids, reconstructions):
        true_text = (NOTES_DIR / f"{eid}.txt").read_text(encoding="utf-8")
        rouge_scores.append(scorer.score(true_text, reconstructed)["rougeL"].fmeasure)
        recall = term_recall(note_terms(ground_truth[eid]), reconstructed)
        if recall is not None:
            term_recalls.append(recall)
    mean_rouge = sum(rouge_scores) / len(rouge_scores)
    mean_term_recall = sum(term_recalls) / len(term_recalls) if term_recalls else None
    return mean_rouge, mean_term_recall


def print_scores(label, rouge, term_recall_value):
    print(f"{label} -- mean ROUGE-L F1: {rouge:.3f}", end="")
    if term_recall_value is not None:
        print(f", mean PHI term recall: {term_recall_value:.3f}")
    else:
        print()


def difference(a, b):
    if a is None or b is None:
        return None
    return a - b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N test encounters for a smoke test.",
    )
    args = parser.parse_args()

    ground_truth = load_ground_truth()
    target_ids = load_split("test")
    if args.limit is not None:
        if args.limit < 2:
            raise ValueError("--limit must be at least 2 so controls can be evaluated")
        target_ids = target_ids[:args.limit]
    train_ids = load_split("train")

    test_records = load_split_records("test")
    patient_of = {eid: test_records[eid]["patient_id"] for eid in target_ids}

    print(f"Loading GEIA attacker from {MODEL_DIR}")
    tokenizer, attacker, projection = load_decoder()

    embeddings = fetch_embeddings(target_ids)
    train_embeddings = fetch_embeddings(train_ids)
    baseline_embedding = train_embeddings.mean(axis=0)

    max_embedding_difference, min_embedding_cosine = embedding_consistency_check(
        target_ids, embeddings, test_records
    )

    n = len(target_ids)
    rolled = np.roll(np.arange(n), 1)
    same_patient_in_old_roll = int(sum(
        patient_of[target_ids[i]] == patient_of[target_ids[rolled[i]]] for i in range(n)
    ))
    print(f"Old roll-by-one control would have paired a target with the same patient's "
          f"embedding in {same_patient_in_old_roll} of {n} targets")

    pairing = cross_patient_pairing(target_ids, patient_of, SEED)
    shuffled_embeddings = embeddings[pairing]

    conditioning_sanity_check(
        embeddings[0], embeddings[pairing[0]], tokenizer, attacker, projection
    )

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    print(f"Reconstructing {n} held-out test notes from embeddings only "
          f"(none of these patients were in the attacker's train split)...")
    reconstructions = [
        reconstruct(embeddings[i], tokenizer, attacker, projection)
        for i in range(n)
    ]
    mean_rouge, mean_term_recall = score_against_targets(
        reconstructions, target_ids, ground_truth, scorer
    )

    print("Reconstructing the shuffled-embedding control: every target is paired "
          "with a random embedding from a different test patient...")
    shuffled_reconstructions = [
        reconstruct(shuffled_embeddings[i], tokenizer, attacker, projection)
        for i in range(n)
    ]
    shuffled_rouge, shuffled_term_recall = score_against_targets(
        shuffled_reconstructions, target_ids, ground_truth, scorer
    )

    print("Reconstructing the control condition: one decoding from the mean "
          "of the attacker's own train-split embeddings, carrying no "
          "information about which target patient is being attacked...")
    baseline_reconstruction = reconstruct(baseline_embedding, tokenizer, attacker, projection)
    baseline_rouge, baseline_term_recall = score_against_targets(
        [baseline_reconstruction] * n, target_ids, ground_truth, scorer
    )

    print()
    print_scores("Real attack", mean_rouge, mean_term_recall)
    print_scores("Shuffled control (different patient)", shuffled_rouge, shuffled_term_recall)
    print_scores("Baseline (no per-target info)", baseline_rouge, baseline_term_recall)
    print(f"ROUGE-L above shuffled: {mean_rouge - shuffled_rouge:+.3f}")
    print(f"ROUGE-L above baseline: {mean_rouge - baseline_rouge:+.3f}")
    recall_above_shuffled = difference(mean_term_recall, shuffled_term_recall)
    recall_above_baseline = difference(mean_term_recall, baseline_term_recall)
    if recall_above_shuffled is not None:
        print(f"PHI term recall above shuffled: {recall_above_shuffled:+.3f}")
    if recall_above_baseline is not None:
        print(f"PHI term recall above baseline: {recall_above_baseline:+.3f}")

    examples = [
        {
            "encounter_id": eid,
            "true_note": (NOTES_DIR / f"{eid}.txt").read_text(encoding="utf-8"),
            "reconstructed": reconstructions[i],
            "shuffled_reconstruction": shuffled_reconstructions[i],
        }
        for i, eid in enumerate(target_ids[:NUM_EXAMPLES_TO_SAVE])
    ]

    print("\nExample reconstruction:")
    example = examples[0]
    print(f"--- True note ({example['encounter_id']}) ---\n{example['true_note']}\n")
    print(f"--- Reconstructed from embedding only ---\n{example['reconstructed']}\n")
    print(f"--- Shuffled control (a different patient's embedding) ---\n"
          f"{example['shuffled_reconstruction']}\n")
    print(f"--- Baseline reconstruction (mean train embedding, same for every target) ---\n"
          f"{baseline_reconstruction}\n")

    save_result(
        run_id="attack_generative_inversion",
        config={
            "attack": "generative_inversion_geia_style",
            "threat_model": "white_box_stolen_vector_db",
            "embedding_model": VICTIM_MODEL,
            "decoder_backbone": "gpt2",
            "prefix_length": PREFIX_LENGTH,
            "target_encounters": n,
            "baseline": "mean_of_train_split_embeddings",
            "shuffled_control": "each_target_paired_with_random_embedding_from_different_test_patient",
            "shuffled_control_seed": SEED,
            "old_roll_same_patient_pairs": same_patient_in_old_roll,
            "embedding_consistency_max_abs_difference": max_embedding_difference,
            "embedding_consistency_min_cosine": min_embedding_cosine,
        },
        attack={
            "mean_rouge_l_f1": mean_rouge,
            "mean_phi_term_recall": mean_term_recall,
            "shuffled_mean_rouge_l_f1": shuffled_rouge,
            "shuffled_mean_phi_term_recall": shuffled_term_recall,
            "rouge_l_above_shuffled": mean_rouge - shuffled_rouge,
            "phi_term_recall_above_shuffled": recall_above_shuffled,
            "baseline_mean_rouge_l_f1": baseline_rouge,
            "baseline_mean_phi_term_recall": baseline_term_recall,
            "rouge_l_above_baseline": mean_rouge - baseline_rouge,
            "phi_term_recall_above_baseline": recall_above_baseline,
            "baseline_reconstruction": baseline_reconstruction,
            "examples": examples,
        },
    )


if __name__ == "__main__":
    main()