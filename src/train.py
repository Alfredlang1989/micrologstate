from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .model import MicroLogClassifier
from .ontology import DOMAINS


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {"text", "domain", "health", "abstain", "group"} - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_no}: missing {sorted(missing)}")
            if row["domain"] not in DOMAINS:
                raise ValueError(f"{path}:{line_no}: invalid domain {row['domain']!r}")
            row["abstain"] = int(row["abstain"])
            if row["abstain"] not in (0, 1):
                raise ValueError(f"{path}:{line_no}: abstain must be 0 or 1")
            health = row.get("health")
            if row["abstain"] == 0 and health not in (0, 1):
                raise ValueError(f"{path}:{line_no}: decidable row needs health 0/1")
            row["health"] = int(health) if health in (0, 1) else None
            row["sample_weight"] = float(row.get("sample_weight", 1.0))
            row["reason"] = str(row.get("reason") or "")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty dataset: {path}")
    return rows


def device_name(value: str) -> str:
    return value if value != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")


def encode(rows: list[dict[str, Any]], data_path: Path, encoder_path: Path, cache_dir: Path, batch_size: int, device: str) -> np.ndarray:
    if not encoder_path.exists():
        raise FileNotFoundError(f"local encoder missing: {encoder_path}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{sha256(data_path)}|{encoder_path.resolve()}".encode()).hexdigest()[:24]
    cache_file = cache_dir / f"embeddings-v3-{key}.npy"
    if cache_file.exists():
        cached = np.load(cache_file, mmap_mode="r")
        if len(cached) == len(rows):
            print(f"Loading cached embeddings: {cache_file}")
            return cached
    encoder = SentenceTransformer(str(encoder_path), device=device, local_files_only=True, trust_remote_code=False)
    values = encoder.encode(
        [str(row["text"]) for row in rows],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    np.save(cache_file, values)
    return values


def split(rows: list[dict[str, Any]], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(rows))
    groups = np.asarray([str(row["group"]) for row in rows])
    if len(set(groups.tolist())) < 10:
        raise ValueError("at least ten template groups are required")
    first = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train, rest = next(first.split(indices, groups=groups))
    second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
    val_local, test_local = next(second.split(rest, groups=groups[rest]))
    return train, rest[val_local], rest[test_local]


def targets(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    lookup = {label: index for index, label in enumerate(DOMAINS)}
    return {
        "domain": np.asarray([lookup[row["domain"]] for row in rows], dtype=np.int64),
        "health": np.asarray([row["health"] if row["health"] in (0, 1) else -100 for row in rows], dtype=np.int64),
        "abstain": np.asarray([row["abstain"] for row in rows], dtype=np.int64),
        "health_mask": np.asarray([1.0 if row["abstain"] == 0 else 0.0 for row in rows], dtype=np.float32),
        "weight": np.asarray([row["sample_weight"] for row in rows], dtype=np.float32),
    }


def loader(embeddings: np.ndarray, target: dict[str, np.ndarray], indices: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = [torch.from_numpy(np.asarray(embeddings[indices], dtype=np.float32))]
    tensors.extend(torch.from_numpy(np.asarray(target[key][indices])) for key in ("domain", "health", "abstain", "health_mask", "weight"))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, pin_memory=torch.cuda.is_available())


def balanced_weights(values: np.ndarray, indices: np.ndarray, count: int, device: torch.device, ignore: int | None = None) -> torch.Tensor:
    selected = values[indices]
    if ignore is not None:
        selected = selected[selected != ignore]
    counts = np.bincount(selected, minlength=count).astype(np.float64)
    result = np.zeros(count, dtype=np.float64)
    present = counts > 0
    result[present] = counts[present].sum() / (present.sum() * counts[present])
    return torch.tensor(result, dtype=torch.float32, device=device)


def weighted_mean(loss: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    effective = weight if mask is None else weight * mask
    return (loss * effective).sum() / effective.sum().clamp_min(1e-8)


def loss_for(logits: dict[str, torch.Tensor], batch: tuple[torch.Tensor, ...], functions: dict[str, nn.Module], abstain_weight: float) -> torch.Tensor:
    _, domain, health, abstain, health_mask, sample_weight = batch
    domain_loss = weighted_mean(functions["domain"](logits["domain"], domain), sample_weight)
    abstain_loss = weighted_mean(functions["abstain"](logits["abstain"], abstain), sample_weight)
    health_loss = weighted_mean(functions["health"](logits["health"], health.clamp_min(0)), sample_weight, health_mask)
    return domain_loss + health_loss + abstain_weight * abstain_loss


@torch.inference_mode()
def predictions(model: nn.Module, batches: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    output: dict[str, list[Any]] = defaultdict(list)
    for raw in batches:
        batch = tuple(item.to(device, non_blocking=True) for item in raw)
        x, domain, health, abstain, health_mask, _ = batch
        logits = model(x)
        domain_prob = torch.softmax(logits["domain"], -1)
        health_prob = torch.softmax(logits["health"], -1)
        abstain_prob = torch.softmax(logits["abstain"], -1)
        dc, dp = domain_prob.max(-1)
        hc, hp = health_prob.max(-1)
        output["domain_true"].extend(domain.cpu().tolist())
        output["health_true"].extend(health.cpu().tolist())
        output["abstain_true"].extend(abstain.cpu().tolist())
        output["health_mask"].extend(health_mask.cpu().tolist())
        output["domain_pred"].extend(dp.cpu().tolist())
        output["health_pred"].extend(hp.cpu().tolist())
        output["abstain_pred"].extend(abstain_prob.argmax(-1).cpu().tolist())
        output["abstain_prob"].extend(abstain_prob[:, 1].cpu().tolist())
        output["accept_score"].extend((dc * hc * abstain_prob[:, 0]).cpu().tolist())
    return {key: np.asarray(value) for key, value in output.items()}


def metrics(data: dict[str, np.ndarray]) -> dict[str, float]:
    mask = data["health_mask"] > 0.5
    return {
        "domain_accuracy": float(accuracy_score(data["domain_true"], data["domain_pred"])),
        "domain_macro_f1": float(f1_score(data["domain_true"], data["domain_pred"], average="macro", zero_division=0)),
        "health_macro_f1": float(f1_score(data["health_true"][mask], data["health_pred"][mask], average="macro", zero_division=0)),
        "abstain_macro_f1": float(f1_score(data["abstain_true"], data["abstain_pred"], average="macro", zero_division=0)),
    }


def calibrate(data: dict[str, np.ndarray], target_precision: float, target_unknown_recall: float) -> dict[str, float]:
    correct = (
        (data["domain_pred"] == data["domain_true"])
        & ((data["health_mask"] < 0.5) | (data["health_pred"] == data["health_true"]))
        & (data["abstain_true"] == 0)
    )
    best = (0.5, 0.5, -1.0)
    for abstain_threshold in np.linspace(0.20, 0.80, 25):
        for accept_threshold in np.linspace(0.20, 0.90, 29):
            accepted = (data["abstain_prob"] < abstain_threshold) & (data["accept_score"] >= accept_threshold)
            precision = float(correct[accepted].mean()) if accepted.any() else 1.0
            unknown = data["abstain_true"] == 1
            unknown_recall = float((~accepted[unknown]).mean()) if unknown.any() else 1.0
            coverage = float(accepted.mean())
            feasible = precision >= target_precision and unknown_recall >= target_unknown_recall
            score = coverage if feasible else precision + unknown_recall - 2.0
            if score > best[2]:
                best = (float(abstain_threshold), float(accept_threshold), score)
    return {"abstain_probability": best[0], "accept_score": best[1]}


def reason_centroids(rows: list[dict[str, Any]], embeddings: np.ndarray, train_indices: np.ndarray, minimum: int = 3) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[np.ndarray]] = defaultdict(list)
    for index in train_indices:
        row = rows[int(index)]
        if row["abstain"] or row["health"] not in (0, 1):
            continue
        reason = row["reason"].strip()
        if not reason or reason in {"mention", "irrelevant", "sequence_only"}:
            continue
        grouped[(row["domain"], int(row["health"]), reason)].append(np.asarray(embeddings[index], dtype=np.float32))
    output: list[dict[str, Any]] = []
    for (domain, health, reason), vectors in sorted(grouped.items()):
        if len(vectors) < minimum:
            continue
        centroid = np.mean(np.stack(vectors), axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-8)
        output.append({"domain": domain, "health": health, "reason": reason, "count": len(vectors), "vector": torch.tensor(centroid)})
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Train fixed domain/health/abstain heads")
    parser.add_argument("--data", type=Path, default=Path("data/processed/log_states_v3.jsonl"))
    parser.add_argument("--encoder", "--encoder-path", dest="encoder", type=Path, default=Path("models/bge-small-en-v1.5"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/micro-log-state-v3.pt"))
    parser.add_argument("--report", type=Path, default=Path("reports/training_v3_metrics.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--target-unknown-recall", type=float, default=0.90)
    parser.add_argument("--abstain-loss-weight", type=float, default=1.25)
    args = parser.parse_args()

    seed_everything(args.seed)
    selected_device = device_name(args.device)
    device = torch.device(selected_device)
    rows = load_rows(args.data)
    embeddings = encode(rows, args.data, args.encoder, args.cache_dir, args.encode_batch_size, selected_device)
    target = targets(rows)
    train_idx, val_idx, test_idx = split(rows, args.seed)
    print(f"Rows: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print("Train domains:", Counter(rows[int(i)]["domain"] for i in train_idx))

    model = MicroLogClassifier(int(embeddings.shape[1]), args.hidden_dim, {"domain": len(DOMAINS), "health": 2, "abstain": 2}, args.dropout).to(device)
    trainable = sum(parameter.numel() for parameter in model.parameters())
    print(f"Trainable parameters: {trainable:,}")
    functions: dict[str, nn.Module] = {
        "domain": nn.CrossEntropyLoss(weight=balanced_weights(target["domain"], train_idx, len(DOMAINS), device), reduction="none"),
        "health": nn.CrossEntropyLoss(weight=balanced_weights(target["health"], train_idx, 2, device, -100), reduction="none"),
        "abstain": nn.CrossEntropyLoss(weight=balanced_weights(target["abstain"], train_idx, 2, device), reduction="none"),
    }
    train_loader = loader(embeddings, target, train_idx, args.train_batch_size, True)
    val_loader = loader(embeddings, target, val_idx, args.train_batch_size, False)
    test_loader = loader(embeddings, target, test_idx, args.train_batch_size, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1.0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for raw in tqdm(train_loader, desc=f"epoch {epoch:02d}", leave=False):
            batch = tuple(item.to(device, non_blocking=True) for item in raw)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(model(batch[0]), batch, functions, args.abstain_loss_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach().cpu())
        val = metrics(predictions(model, val_loader, device))
        score = 0.35 * val["domain_macro_f1"] + 0.35 * val["health_macro_f1"] + 0.30 * val["abstain_macro_f1"]
        history.append({"epoch": epoch, "loss": running / max(1, len(train_loader)), "score": score, "metrics": val})
        print(f"epoch={epoch:02d} loss={history[-1]['loss']:.4f} score={score:.3f} {val}")
        if score > best_score + 1e-4:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping")
                break

    if best_state is None:
        raise RuntimeError("no checkpoint produced")
    model.load_state_dict(best_state)
    model.to(device)
    val_data = predictions(model, val_loader, device)
    thresholds = calibrate(val_data, args.target_precision, args.target_unknown_recall)
    test = metrics(predictions(model, test_loader, device))
    prototypes = reason_centroids(rows, embeddings, train_idx)

    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": 3,
        "model_state": best_state,
        "embedding_dim": int(embeddings.shape[1]),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "domains": DOMAINS,
        "thresholds": thresholds,
        "encoder_path": str(args.encoder),
        "reason_centroids": prototypes,
        "training_data": str(args.data),
        "training_data_sha256": sha256(args.data),
        "trainable_parameters": trainable,
        "seed": args.seed,
    }, args.artifact)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "version": 3,
        "rows": len(rows),
        "splits": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "thresholds": thresholds,
        "test": test,
        "best_validation_score": best_score,
        "reason_prototypes": [{k: v for k, v in item.items() if k != "vector"} for item in prototypes],
        "history": history,
    }, indent=2), encoding="utf-8")
    print(f"Saved model: {args.artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
