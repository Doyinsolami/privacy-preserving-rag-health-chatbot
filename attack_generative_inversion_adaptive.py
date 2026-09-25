import argparse
import json
from pathlib import Path

import numpy as np
import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

from attack_common import build_vocab, fetch_embeddings, load_ground_truth, load_split
from attack_generative_inversion import (
    NUM_EXAMPLES_TO_SAVE,
    SEED,
    collection_dp_config,
    conditioning_sanity_check,
    cross_patient_pairing,
    deltas,
    load_split_records,
    print_scores,
    reconstruct,
    score_against_targets,
)
from results import save_result
from train_attacker_adaptive import PREFIX_LENGTH, VICTIM_MODEL, Projection


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
NOTES_DIR = DATA_DIR / "notes"
ADAPTIVE_MODELS_DIR = PROJECT_ROOT / "models" / "adaptive_attackers"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_training_config(model_dir):
    config_path = model_dir / "training_run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing {config_path}. The adaptive checkpoint must include its training configuration."
        )
    return read_json(config_path)


def config_value(config, *names):
    for name in names:
        if name in config:
            return config[name]
    return None


def validate_adaptive_checkpoint(config, collection_name, allow_limited_model):
    training_mode = config_value(config, "training_mode", "attacker_mode")
    if training_mode not in (
    None,
    "adaptive",
    "adaptive_dp_trained",
    "adaptive_dp_embedding_attacker",
):
        raise ValueError(
            f"Checkpoint training mode is {training_mode!r}; an adaptive checkpoint is required."
        )

    trained_collection = config_value(config, "dp_collection", "collection")
    if trained_collection != collection_name:
        raise ValueError(
            f"Checkpoint was trained on {trained_collection!r}, but evaluation selected "
            f"{collection_name!r}. Use the model trained for the same DP collection."
        )

    train_limit = config_value(config, "train_limit", "limit")
    validation_limit = config_value(config, "validation_limit")
    limited = train_limit is not None or validation_limit is not None
    if limited and not allow_limited_model:
        raise ValueError(
            "This checkpoint was trained with a train or validation limit. Use the full "
            "checkpoint, or pass --allow-limited-model only for a smoke test."
        )

    configured_prefix = config_value(config, "prefix_length")
    if configured_prefix is not None and int(configured_prefix) != PREFIX_LENGTH:
        raise ValueError(
            f"Checkpoint prefix length is {configured_prefix}, but this evaluator expects "
            f"{PREFIX_LENGTH}."
        )


def load_adaptive_decoder(model_dir):
    gpt2_dir = model_dir / "gpt2"
    projection_path = model_dir / "projection.pt"
    if not gpt2_dir.exists():
        raise FileNotFoundError(f"Missing adaptive decoder directory: {gpt2_dir}")
    if not projection_path.exists():
        raise FileNotFoundError(f"Missing adaptive projection weights: {projection_path}")

    tokenizer = AutoTokenizer.from_pretrained(gpt2_dir)
    tokenizer.pad_token = tokenizer.eos_token
    attacker = AutoModelForCausalLM.from_pretrained(gpt2_dir).to(DEVICE).eval()

    projection = Projection().to(DEVICE)
    projection.load_state_dict(torch.load(projection_path, map_location=DEVICE))
    projection.eval()
    return tokenizer, attacker, projection


def validate_collection(target_collection, collection_name, dp_config):
    count = target_collection.count()
    if count <= 0:
        raise ValueError(f"Collection {collection_name!r} is empty.")
    if not dp_config["dp_applied"]:
        raise ValueError(
            f"Collection {collection_name!r} does not contain DP metadata. "
            "The adaptive evaluator requires a DP collection."
        )
    if dp_config["epsilon"] is None:
        raise ValueError(f"Collection {collection_name!r} has no stored epsilon value.")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a GEIA attacker trained adaptively on a DP embedding collection."
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="DP Chroma collection used for both adaptive training and evaluation.",
    )
    parser.add_argument(
        "--model-tag",
        default=None,
        help=(
            "Folder name under models/adaptive_attackers. "
            "Defaults to the collection name."
        ),
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Optional explicit adaptive checkpoint directory. Overrides --model-tag.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Result filename tag. Defaults to the collection name.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate the first N held-out test encounters for a smoke test.",
    )
    parser.add_argument(
        "--allow-limited-model",
        action="store_true",
        help="Allow a checkpoint trained with train or validation limits for smoke testing.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 2:
        raise ValueError("--limit must be at least 2 so controls can be evaluated")

    from ingest import get_collection

    target_collection = get_collection(args.collection)
    dp_config = collection_dp_config(target_collection)
    selected_count = validate_collection(target_collection, args.collection, dp_config)

    if args.model_dir:
        model_dir = Path(args.model_dir).expanduser().resolve()
    else:
        model_tag = args.model_tag or args.collection
        model_dir = ADAPTIVE_MODELS_DIR / model_tag

    if not model_dir.exists():
        raise FileNotFoundError(f"Adaptive checkpoint directory does not exist: {model_dir}")

    training_config = load_training_config(model_dir)
    validate_adaptive_checkpoint(
        training_config,
        args.collection,
        args.allow_limited_model,
    )

    print(f"Device: {DEVICE}")
    print(f"Selected DP collection: {args.collection} ({selected_count} records)")
    print(
        f"DP settings: epsilon={dp_config['epsilon']}, delta={dp_config['delta']}, "
        f"sigma={dp_config['noise_sigma']}"
    )
    print(f"Loading adaptive GEIA attacker from {model_dir}")

    ground_truth = load_ground_truth()
    target_ids = load_split("test")
    if args.limit is not None:
        target_ids = target_ids[: args.limit]
    train_ids = load_split("train")

    test_records = load_split_records("test")
    patient_of = {eid: test_records[eid]["patient_id"] for eid in target_ids}

    train_records = load_split_records("train")
    train_patients = {record["patient_id"] for record in train_records.values()}
    test_patients = {patient_of[eid] for eid in target_ids}
    overlap = train_patients & test_patients
    if overlap:
        raise RuntimeError(
            f"Patient leakage detected between attacker train and test splits: {len(overlap)} patients"
        )
    print(f"Held-out test encounters: {len(target_ids)} from {len(test_patients)} patients")
    print("Patient overlap between attacker training and test: 0")

    vocab = build_vocab(
        ground_truth,
        list(ground_truth.keys()),
        min_df=1,
        max_df_ratio=1.0,
        top_k=None,
    )
    print(
        f"Clinical detection vocabulary for precision: {len(vocab)} terms "
        "(every clinical term in the corpus)"
    )

    tokenizer, attacker, projection = load_adaptive_decoder(model_dir)

    embeddings = fetch_embeddings(target_ids, use_collection=target_collection)
    train_embeddings = fetch_embeddings(train_ids, use_collection=target_collection)
    if embeddings.ndim != 2 or train_embeddings.ndim != 2:
        raise ValueError("Expected two-dimensional embedding arrays from Chroma.")
    if embeddings.shape[1] != train_embeddings.shape[1]:
        raise ValueError("Train and test embeddings have different dimensions.")

    baseline_embedding = train_embeddings.mean(axis=0)
    n = len(target_ids)

    rolled = np.roll(np.arange(n), 1)
    same_patient_in_old_roll = int(
        sum(
            patient_of[target_ids[i]] == patient_of[target_ids[rolled[i]]]
            for i in range(n)
        )
    )
    print(
        "Old roll-by-one control would have paired a target with the same patient's "
        f"embedding in {same_patient_in_old_roll} of {n} targets"
    )

    pairing = cross_patient_pairing(target_ids, patient_of, SEED)
    shuffled_embeddings = embeddings[pairing]

    conditioning_sanity_check(
        embeddings[0],
        embeddings[pairing[0]],
        tokenizer,
        attacker,
        projection,
    )

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    print(
        f"Reconstructing {n} held-out test notes with the adaptive DP-trained attacker..."
    )
    reconstructions = [
        reconstruct(embeddings[i], tokenizer, attacker, projection)
        for i in range(n)
    ]
    real_scores = score_against_targets(
        reconstructions,
        target_ids,
        ground_truth,
        scorer,
        vocab,
    )

    print(
        "Reconstructing the shuffled-embedding control using a different test patient's "
        "DP embedding for every target..."
    )
    shuffled_reconstructions = [
        reconstruct(shuffled_embeddings[i], tokenizer, attacker, projection)
        for i in range(n)
    ]
    shuffled_scores = score_against_targets(
        shuffled_reconstructions,
        target_ids,
        ground_truth,
        scorer,
        vocab,
    )

    print(
        "Reconstructing the no-target-information baseline from the mean adaptive-training "
        "DP embedding..."
    )
    baseline_reconstruction = reconstruct(
        baseline_embedding,
        tokenizer,
        attacker,
        projection,
    )
    baseline_scores = score_against_targets(
        [baseline_reconstruction] * n,
        target_ids,
        ground_truth,
        scorer,
        vocab,
    )

    real_above_shuffled = deltas(real_scores, shuffled_scores)
    real_above_baseline = deltas(real_scores, baseline_scores)

    print()
    print_scores("Real adaptive attack", real_scores)
    print_scores("Shuffled control (different patient)", shuffled_scores)
    print_scores("Baseline (no per-target information)", baseline_scores)
    print()
    print_scores("Real above shuffled", real_above_shuffled)
    print_scores("Real above baseline", real_above_baseline)

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
    print(f"--- Adaptive reconstruction ---\n{example['reconstructed']}\n")
    print(
        "--- Shuffled control (different patient's DP embedding) ---\n"
        f"{example['shuffled_reconstruction']}\n"
    )
    print(
        "--- Baseline reconstruction (mean adaptive-training DP embedding) ---\n"
        f"{baseline_reconstruction}\n"
    )

    tag = args.tag or args.collection
    run_id = f"geia_adaptive_{tag}"
    save_result(
        run_id=run_id,
        config={
            "attack": "generative_inversion_geia_style",
            "threat_model": "white_box_stolen_vector_db_adaptive_attacker",
            "collection": args.collection,
            **dp_config,
            "attacker_mode": "adaptive_dp_trained",
            "adaptive_model_dir": str(model_dir),
            "adaptive_training_collection": config_value(
                training_config, "dp_collection", "collection"
            ),
            "embedding_model": VICTIM_MODEL,
            "decoder_backbone": "gpt2",
            "prefix_length": PREFIX_LENGTH,
            "target_encounters": n,
            "test_patients": len(test_patients),
            "train_test_patient_overlap": 0,
            "clinical_vocab_size": len(vocab),
            "clinical_prf_averaging": "micro",
            "baseline": "mean_of_dp_train_split_embeddings",
            "shuffled_control": (
                "each_target_paired_with_random_dp_embedding_from_different_test_patient"
            ),
            "shuffled_control_seed": SEED,
            "old_roll_same_patient_pairs": same_patient_in_old_roll,
        },
        attack={
            "real": real_scores,
            "shuffled_control": shuffled_scores,
            "baseline": baseline_scores,
            "real_above_shuffled": real_above_shuffled,
            "real_above_baseline": real_above_baseline,
            "baseline_reconstruction": baseline_reconstruction,
            "examples": examples,
        },
    )


if __name__ == "__main__":
    main()
