from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch import nn

from .ontology import Classification, DOMAINS, render_nagios


class MicroLogClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        label_sizes: Mapping[str, int],
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.adapter = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict(
            {name: nn.Linear(self.hidden_dim, int(size)) for name, size in label_sizes.items()}
        )

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.adapter(embeddings)
        return {name: head(hidden) for name, head in self.heads.items()}


def resolve_device(value: str) -> str:
    if value != "auto":
        return value
    return "cuda" if torch.cuda.is_available() else "cpu"


class ModelEngine:
    def __init__(self, encoder_path: Path, artifact_path: Path, device_name: str = "auto") -> None:
        self.encoder_path = encoder_path.resolve()
        self.artifact_path = artifact_path.resolve()
        self.device_name = resolve_device(device_name)
        self.device = torch.device(self.device_name)
        if not self.encoder_path.exists():
            raise FileNotFoundError(f"embedding model not found: {self.encoder_path}")
        if not self.artifact_path.exists():
            raise FileNotFoundError(f"classifier checkpoint not found: {self.artifact_path}")

        self.checkpoint: dict[str, Any] = torch.load(
            self.artifact_path, map_location="cpu", weights_only=False
        )
        if int(self.checkpoint.get("version", 0)) != 3:
            raise ValueError(
                f"alfilogd runtime expects v3 checkpoint, got version={self.checkpoint.get('version')}"
            )
        self.domains: list[str] = list(self.checkpoint.get("domains", DOMAINS))
        self.model = MicroLogClassifier(
            embedding_dim=int(self.checkpoint["embedding_dim"]),
            hidden_dim=int(self.checkpoint["hidden_dim"]),
            label_sizes={"domain": len(self.domains), "health": 2, "abstain": 2},
            dropout=float(self.checkpoint.get("dropout", 0.0)),
        )
        self.model.load_state_dict(self.checkpoint["model_state"])
        self.model.to(self.device).eval()
        self.encoder = SentenceTransformer(
            str(self.encoder_path),
            device=self.device_name,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.thresholds = dict(self.checkpoint["thresholds"])
        self.reason_centroids = list(self.checkpoint.get("reason_centroids", []))
        for item in self.reason_centroids:
            vector = item["vector"]
            if not isinstance(vector, torch.Tensor):
                vector = torch.tensor(vector, dtype=torch.float32)
            item["vector"] = vector.float().cpu()

    def _retrieve_reason(self, embedding: np.ndarray, domain: str, health: int) -> tuple[str | None, float | None]:
        candidates = [
            item
            for item in self.reason_centroids
            if item["domain"] == domain and int(item["health"]) == health
        ]
        if not candidates:
            return None, None
        vector = torch.tensor(embedding, dtype=torch.float32)
        vector = vector / vector.norm().clamp_min(1e-8)
        best_reason: str | None = None
        best_score = -1.0
        for item in candidates:
            score = float(torch.dot(vector, item["vector"]).item())
            if score > best_score:
                best_score = score
                best_reason = str(item["reason"])
        return best_reason, best_score

    @torch.inference_mode()
    def classify_batch(self, texts: Sequence[str]) -> list[Classification]:
        if not texts:
            return []
        embeddings = self.encoder.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=max(1, min(256, len(texts))),
        ).astype(np.float32)
        tensor = torch.from_numpy(embeddings).to(self.device)
        logits = self.model(tensor)
        domain_probs = torch.softmax(logits["domain"], dim=-1).cpu()
        health_probs = torch.softmax(logits["health"], dim=-1).cpu()
        abstain_probs = torch.softmax(logits["abstain"], dim=-1).cpu()

        results: list[Classification] = []
        for index, text in enumerate(texts):
            domain_conf, domain_index = domain_probs[index].max(dim=-1)
            health_conf, health_index = health_probs[index].max(dim=-1)
            domain = self.domains[int(domain_index)]
            health = int(health_index)
            abstain_probability = float(abstain_probs[index, 1])
            accept_score = float(domain_conf * health_conf * abstain_probs[index, 0])
            rejected = (
                domain == "UNKNOWN"
                or abstain_probability >= float(self.thresholds["abstain_probability"])
                or accept_score < float(self.thresholds["accept_score"])
            )
            reason = None
            reason_confidence = None
            if not rejected:
                reason, reason_confidence = self._retrieve_reason(
                    embeddings[index], domain, health
                )
            effective_health = None if rejected else health
            nagios_code, nagios = render_nagios(
                domain, effective_health, rejected, accept_score, reason
            )
            results.append(
                Classification(
                    text=text,
                    domain=domain,
                    health=effective_health,
                    abstain=rejected,
                    reason=reason,
                    reason_confidence=reason_confidence,
                    domain_confidence=float(domain_conf),
                    health_confidence=float(health_conf),
                    abstain_probability=abstain_probability,
                    accept_score=accept_score,
                    nagios_code=nagios_code,
                    nagios=nagios,
                )
            )
        return results
