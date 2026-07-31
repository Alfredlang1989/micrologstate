from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .model import MicroLogClassifier
from .ontology import DOMAINS, Prediction, render_nagios


def resolve_device(value: str) -> str:
    if value != "auto":
        return value
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_encoder_path(checkpoint_path: Path, configured: str) -> Path:
    path = Path(configured)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return checkpoint_path.resolve().parent.parent / path


def load_runtime(artifact: Path, device_name: str) -> tuple[dict[str, Any], MicroLogClassifier, SentenceTransformer, torch.device]:
    checkpoint = torch.load(artifact, map_location="cpu", weights_only=False)
    if int(checkpoint.get("version", 0)) != 3:
        raise ValueError(f"Expected v3 checkpoint, got version={checkpoint.get('version')}")
    device = torch.device(device_name)
    model = MicroLogClassifier(
        embedding_dim=int(checkpoint["embedding_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        label_sizes={"domain": len(checkpoint.get("domains", DOMAINS)), "health": 2, "abstain": 2},
        dropout=float(checkpoint.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    encoder_path = resolve_encoder_path(artifact, str(checkpoint["encoder_path"]))
    if not encoder_path.exists():
        raise FileNotFoundError(f"Local encoder not found: {encoder_path}")
    encoder = SentenceTransformer(
        str(encoder_path), device=device_name, local_files_only=True, trust_remote_code=False
    )
    return checkpoint, model, encoder, device


def retrieve_reason(embedding: np.ndarray, domain: str, health: int, centroids: list[dict[str, Any]]) -> tuple[str | None, float | None]:
    candidates = [item for item in centroids if item["domain"] == domain and int(item["health"]) == health]
    if not candidates:
        return None, None
    vector = torch.tensor(embedding, dtype=torch.float32)
    vector = vector / vector.norm().clamp_min(1e-8)
    best_reason = None
    best_score = -1.0
    for item in candidates:
        centroid = item["vector"].float()
        score = float(torch.dot(vector, centroid).item())
        if score > best_score:
            best_score = score
            best_reason = str(item["reason"])
    return best_reason, best_score


@torch.no_grad()
def predict_one(text: str, checkpoint: dict[str, Any], model: MicroLogClassifier, encoder: SentenceTransformer, device: torch.device) -> dict[str, Any]:
    embedding = encoder.encode([text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0].astype(np.float32)
    x = torch.from_numpy(embedding).unsqueeze(0).to(device)
    logits = model(x)
    domain_probs = torch.softmax(logits["domain"], dim=-1)[0]
    health_probs = torch.softmax(logits["health"], dim=-1)[0]
    abstain_probs = torch.softmax(logits["abstain"], dim=-1)[0]

    domain_conf, domain_index = domain_probs.max(dim=-1)
    health_conf, health_index = health_probs.max(dim=-1)
    domain = checkpoint.get("domains", DOMAINS)[int(domain_index)]
    health = int(health_index)
    abstain_probability = float(abstain_probs[1])
    accept_score = float(domain_conf * health_conf * abstain_probs[0])
    thresholds = checkpoint["thresholds"]
    rejected = (
        domain == "UNKNOWN"
        or abstain_probability >= float(thresholds["abstain_probability"])
        or accept_score < float(thresholds["accept_score"])
    )

    reason = None
    reason_confidence = None
    if not rejected:
        reason, reason_confidence = retrieve_reason(embedding, domain, health, checkpoint.get("reason_centroids", []))

    prediction = Prediction(
        domain=domain,
        health=None if rejected else health,
        abstain=rejected,
        confidence=accept_score,
        reason=reason,
        reason_confidence=reason_confidence,
    )
    nagios_code, nagios = render_nagios(prediction)
    return {
        "text": text,
        "domain": domain,
        "health": None if rejected else ("OK" if health == 0 else "BAD"),
        "health_value": None if rejected else health,
        "abstain": rejected,
        "reason": reason,
        "reason_confidence": reason_confidence,
        "domain_confidence": float(domain_conf),
        "health_confidence": float(health_conf),
        "abstain_probability": abstain_probability,
        "accept_score": accept_score,
        "thresholds": thresholds,
        "nagios_code": nagios_code,
        "nagios": nagios,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline prediction with the v3 micro log classifier")
    parser.add_argument("text", nargs="*")
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/micro-log-state-v3.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--stdin", action="store_true", help="Read one log line per stdin line and keep the encoder loaded")
    args = parser.parse_args()

    device_name = resolve_device(args.device)
    checkpoint, model, encoder, device = load_runtime(args.artifact, device_name)
    texts: list[str] = []
    if args.stdin:
        texts.extend(line.rstrip("\n") for line in sys.stdin if line.strip())
    if args.text:
        texts.append(" ".join(args.text))
    if not texts:
        parser.error("provide text or use --stdin")

    for text in texts:
        result = predict_one(text, checkpoint, model, encoder, device)
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result["nagios"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
