import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from attack_common import fetch_embeddings


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
GEIA_DIR = DATA_DIR / "geia"
MODEL_ROOT_DIR = PROJECT_ROOT / "models" / "adaptive_attackers"
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "chroma_db"

VICTIM_MODEL = "all-MiniLM-L6-v2"
ATTACKER_MODEL = "gpt2"
MINILM_DIM = 384
GPT2_DIM = 768
PREFIX_LENGTH = 10
MAX_LENGTH = 256
SEED = 42

SAFE_TAG_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AdaptiveNoteDataset(Dataset):
    def __init__(self, jsonl_path, embedding_collection, limit=None):
        self.ids = []
        self.patient_ids = []
        self.texts = []

        with Path(jsonl_path).open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                try:
                    encounter_id = record["encounter_id"]
                    patient_id = record["patient_id"]
                    text = record["text"]
                except KeyError as error:
                    raise ValueError(
                        f"Missing {error.args[0]!r} in {jsonl_path} "
                        f"at line {line_number}"
                    ) from error

                self.ids.append(encounter_id)
                self.patient_ids.append(patient_id)
                self.texts.append(text)

        if limit is not None:
            if limit <= 0:
                raise ValueError("Dataset limit must be greater than zero")
            self.ids = self.ids[:limit]
            self.patient_ids = self.patient_ids[:limit]
            self.texts = self.texts[:limit]

        if not self.ids:
            raise ValueError(f"No records loaded from {jsonl_path}")

        self.embeddings = fetch_embeddings(
            self.ids,
            use_collection=embedding_collection,
        )

        if self.embeddings.shape != (len(self.ids), MINILM_DIM):
            raise ValueError(
                f"Expected embedding matrix {(len(self.ids), MINILM_DIM)}, "
                f"found {self.embeddings.shape}"
            )

        if not np.all(np.isfinite(self.embeddings)):
            raise ValueError(f"Non-finite embeddings loaded for {jsonl_path}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], self.embeddings[index]

    @staticmethod
    def collate(batch):
        texts, embeddings = zip(*batch)
        return list(texts), np.stack(embeddings).astype(np.float32, copy=False)


class Projection(nn.Module):
    def __init__(
        self,
        in_dim=MINILM_DIM,
        out_dim=GPT2_DIM,
        prefix_length=PREFIX_LENGTH,
    ):
        super().__init__()
        self.prefix_length = prefix_length
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim * prefix_length),
        )

    def forward(self, embeddings):
        projected = self.net(embeddings)
        return projected.view(
            embeddings.size(0),
            self.prefix_length,
            self.out_dim,
        )


def build_prefix(embeddings, projection, attacker, tokenizer, device):
    prefix = projection(embeddings)
    bos_ids = torch.full(
        (embeddings.size(0), 1),
        tokenizer.bos_token_id,
        dtype=torch.long,
        device=device,
    )
    bos_embeddings = attacker.transformer.wte(bos_ids)
    return torch.cat((bos_embeddings, prefix), dim=1)


def train_on_batch(
    batch_texts,
    batch_embeddings,
    projection,
    attacker,
    tokenizer,
    device,
    shuffle_embeddings=False,
):
    embeddings = torch.as_tensor(
        batch_embeddings,
        dtype=torch.float32,
        device=device,
    )

    if shuffle_embeddings:
        embeddings = torch.roll(embeddings, shifts=1, dims=0)

    prefix_embeddings = build_prefix(
        embeddings,
        projection,
        attacker,
        tokenizer,
        device,
    )
    prefix_size = prefix_embeddings.size(1)

    encoded = tokenizer(
        batch_texts,
        add_special_tokens=False,
        padding=False,
        truncation=True,
        max_length=MAX_LENGTH - 1,
    )

    input_ids = torch.full(
        (len(batch_texts), MAX_LENGTH),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(batch_texts), MAX_LENGTH),
        dtype=torch.long,
        device=device,
    )

    for row, token_ids in enumerate(encoded["input_ids"]):
        sequence = token_ids + [tokenizer.eos_token_id]
        sequence_length = len(sequence)
        input_ids[row, :sequence_length] = torch.tensor(
            sequence,
            dtype=torch.long,
            device=device,
        )
        attention_mask[row, :sequence_length] = 1

    token_embeddings = attacker.transformer.wte(input_ids)
    inputs_embeddings = torch.cat(
        (prefix_embeddings, token_embeddings),
        dim=1,
    )
    model_attention_mask = torch.cat(
        (
            torch.ones(
                (len(batch_texts), prefix_size),
                dtype=torch.long,
                device=device,
            ),
            attention_mask,
        ),
        dim=1,
    )

    outputs = attacker(
        inputs_embeds=inputs_embeddings,
        attention_mask=model_attention_mask,
    )
    logits = outputs.logits

    shift_logits = logits[:, prefix_size - 1:-1, :].contiguous()
    shift_labels = input_ids.masked_fill(
        attention_mask == 0,
        -100,
    ).contiguous()

    return nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )



def evaluate(
    dataloader,
    projection,
    attacker,
    tokenizer,
    device,
):
    attacker.eval()
    projection.eval()
    total_loss = 0.0
    total_shuffled_loss = 0.0

    with torch.no_grad():
        for batch_texts, batch_embeddings in dataloader:
            total_loss += train_on_batch(
                batch_texts,
                batch_embeddings,
                projection,
                attacker,
                tokenizer,
                device,
            ).item()
            total_shuffled_loss += train_on_batch(
                batch_texts,
                batch_embeddings,
                projection,
                attacker,
                tokenizer,
                device,
                shuffle_embeddings=True,
            ).item()

    attacker.train()
    projection.train()

    return (
        total_loss / len(dataloader),
        total_shuffled_loss / len(dataloader),
    )


def save_checkpoint(attacker, tokenizer, projection, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    gpt2_dir = output_dir / "gpt2"
    attacker.save_pretrained(gpt2_dir)
    tokenizer.save_pretrained(gpt2_dir)
    torch.save(
        projection.state_dict(),
        output_dir / "projection.pt",
    )


def validate_collection(collection, collection_name):
    if collection.count() == 0:
        raise ValueError(f"Collection {collection_name!r} is empty")

    sample = collection.get(
        limit=1,
        include=["embeddings", "metadatas"],
    )
    metadata = (sample.get("metadatas") or [{}])[0] or {}
    embeddings = sample.get("embeddings")

    if not metadata.get("dp_applied", False):
        raise ValueError(
            f"Collection {collection_name!r} is not marked as DP-protected"
        )

    if embeddings is None or len(embeddings) != 1:
        raise ValueError(
            f"Could not read a sample embedding from {collection_name!r}"
        )

    embedding_dimension = len(embeddings[0])
    if embedding_dimension != MINILM_DIM:
        raise ValueError(
            f"Expected {MINILM_DIM}-dimensional embeddings, "
            f"found {embedding_dimension}"
        )

    return {
        "dp_applied": True,
        "epsilon": metadata.get("dp_epsilon"),
        "delta": metadata.get("dp_delta"),
        "clip_norm": metadata.get("dp_clip_norm"),
        "sensitivity": metadata.get("dp_sensitivity"),
        "noise_sigma": metadata.get("dp_noise_sigma"),
        "dp_seed": metadata.get("dp_seed"),
        "mechanism": metadata.get("dp_mechanism"),
        "privacy_unit": metadata.get("dp_unit"),
        "adjacency": metadata.get("dp_adjacency"),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--projection_lr", type=float, default=5e-4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument(
        "--dp-collection",
        required=True,
        help="DP Chroma collection used for adaptive attacker training.",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help=(
            "Output folder name under models/adaptive_attackers. "
            "Defaults to the DP collection name. Use a separate tag for smoke tests."
        ),
    )
    parser.add_argument(
        "--chroma-path",
        default=str(DEFAULT_CHROMA_DIR),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing adaptive attacker output folder.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.epochs <= 0:
        raise SystemExit("--epochs must be greater than zero")
    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be greater than zero")
    if args.lr <= 0:
        raise SystemExit("--lr must be greater than zero")
    if args.projection_lr <= 0:
        raise SystemExit("--projection_lr must be greater than zero")
    if args.patience <= 0:
        raise SystemExit("--patience must be greater than zero")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than zero")
    if args.validation_limit is not None and args.validation_limit <= 0:
        raise SystemExit("--validation-limit must be greater than zero")


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    run_tag = args.run_tag or args.dp_collection
    if run_tag in {".", ".."} or not SAFE_TAG_RE.fullmatch(run_tag):
        raise SystemExit(
            "--run-tag may contain only letters, numbers, underscores, dots, and hyphens"
        )

    model_output_dir = MODEL_ROOT_DIR / run_tag
    if model_output_dir.exists() and any(model_output_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(
                f"Output directory {model_output_dir} already exists and is not empty. "
                "Use a different --run-tag or pass --overwrite."
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")

    import chromadb

    client = chromadb.PersistentClient(
        path=str(Path(args.chroma_path).resolve())
    )
    embedding_collection = client.get_collection(args.dp_collection)
    dp_config = validate_collection(
        embedding_collection,
        args.dp_collection,
    )
    print(
        f"Adaptive training collection: {args.dp_collection} "
        f"({embedding_collection.count()} records)"
    )
    print(
        f"DP settings: epsilon={dp_config['epsilon']}, "
        f"delta={dp_config['delta']}, sigma={dp_config['noise_sigma']}"
    )

    train_dataset = AdaptiveNoteDataset(
        GEIA_DIR / "train.jsonl",
        embedding_collection,
        limit=args.limit,
    )
    validation_dataset = AdaptiveNoteDataset(
        GEIA_DIR / "validation.jsonl",
        embedding_collection,
        limit=args.validation_limit,
    )

    train_patient_ids = set(train_dataset.patient_ids)
    validation_patient_ids = set(validation_dataset.patient_ids)
    patient_overlap = train_patient_ids & validation_patient_ids
    if patient_overlap:
        raise RuntimeError(
            f"Patient leakage detected between train and validation: "
            f"{len(patient_overlap)} overlapping patients"
        )

    print(
        f"Training on {len(train_dataset)} notes from "
        f"{len(train_patient_ids)} patients"
    )
    print(
        f"Validating on {len(validation_dataset)} notes from "
        f"{len(validation_patient_ids)} patients"
    )
    print("Patient overlap between train and validation: 0")

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    validation_generator = torch.Generator()
    validation_generator.manual_seed(args.seed)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate,
        generator=train_generator,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=validation_dataset.collate,
        generator=validation_generator,
    )

    attacker = AutoModelForCausalLM.from_pretrained(ATTACKER_MODEL).to(device)
    tokenizer = AutoTokenizer.from_pretrained(ATTACKER_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    projection = Projection().to(device)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": projection.parameters(),
                "lr": args.projection_lr,
            },
            {
                "params": attacker.parameters(),
                "lr": args.lr,
            },
        ]
    )

    best_validation_loss = float("inf")
    best_shuffled_validation_loss = None
    best_epoch = None
    epochs_without_improvement = 0
    epochs_completed = 0
    history = []

    attacker.train()
    projection.train()

    for epoch in range(args.epochs):
        running_loss = 0.0

        for step, (batch_texts, batch_embeddings) in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = train_on_batch(
                batch_texts,
                batch_embeddings,
                projection,
                attacker,
                tokenizer,
                device,
            )
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if step % 20 == 0:
                print(
                    f"epoch {epoch} step {step} "
                    f"loss {loss.item():.4f}"
                )

        average_train_loss = running_loss / len(train_dataloader)
        epochs_completed = epoch + 1

        validation_generator.manual_seed(args.seed)
        validation_loss, shuffled_validation_loss = evaluate(
            validation_dataloader,
            projection,
            attacker,
            tokenizer,
            device,
        )
        gap = shuffled_validation_loss - validation_loss

        history.append(
            {
                "epoch": epoch,
                "train_loss": average_train_loss,
                "validation_loss": validation_loss,
                "shuffled_validation_loss": shuffled_validation_loss,
                "gap": gap,
            }
        )

        print(
            f"== epoch {epoch} done, "
            f"average train loss {average_train_loss:.4f}, "
            f"validation loss {validation_loss:.4f}, "
            f"shuffled validation loss {shuffled_validation_loss:.4f}, "
            f"gap {gap:.4f} =="
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_shuffled_validation_loss = shuffled_validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                attacker,
                tokenizer,
                projection,
                model_output_dir,
            )
            print(
                f"Validation loss improved, saved checkpoint "
                f"from epoch {epoch}"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for "
                f"{epochs_without_improvement} epoch(s)"
            )
            if epochs_without_improvement >= args.patience:
                print(
                    f"Stopping early after epoch {epoch}: "
                    f"no validation improvement in "
                    f"{args.patience} epochs"
                )
                break

    if best_epoch is None or best_shuffled_validation_loss is None:
        raise RuntimeError("Training completed without saving a checkpoint")

    run_config = {
        "training_mode": "adaptive_dp_embedding_attacker",
        "seed": args.seed,
        "dp_collection": args.dp_collection,
        "run_tag": run_tag,
        **dp_config,
        "epochs_requested": args.epochs,
        "epochs_completed": epochs_completed,
        "early_stopped": epochs_completed < args.epochs,
        "patience": args.patience,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "best_shuffled_validation_loss": best_shuffled_validation_loss,
        "best_gap": best_shuffled_validation_loss - best_validation_loss,
        "history": history,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "projection_lr": args.projection_lr,
        "prefix_length": PREFIX_LENGTH,
        "train_limit": args.limit,
        "validation_limit": args.validation_limit,
        "victim_model": VICTIM_MODEL,
        "attacker_model": ATTACKER_MODEL,
        "max_length": MAX_LENGTH,
        "train_notes": len(train_dataset),
        "validation_notes": len(validation_dataset),
        "train_patients": len(train_patient_ids),
        "validation_patients": len(validation_patient_ids),
        "patient_overlap_count": 0,
        "device": str(device),
        "model_output_dir": str(model_output_dir.resolve()),
    }

    model_output_dir.mkdir(parents=True, exist_ok=True)
    (model_output_dir / "training_run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )

    print(
        f"Best adaptive model from epoch {best_epoch}, "
        f"validation loss {best_validation_loss:.4f}, "
        f"saved to {model_output_dir}"
    )


if __name__ == "__main__":
    main()
