"""NumPy FM matching the organizer's feature encoding and optimizer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rex.data.views import FeatureView, TargetView


FIELDS = ("user_id", "video_id", "author_id", "tab", "dur_bucket")


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


@dataclass
class CategoricalEncoder:
    edges: np.ndarray
    vocabs: list[dict[str, int]]
    unknown: np.ndarray
    offsets: np.ndarray
    dimension: int

    @classmethod
    def fit(cls, view: FeatureView) -> "CategoricalEncoder":
        edges = np.quantile(view.arrays["duration_ms"], np.linspace(0, 1, 11)[1:-1])
        raw = cls._raw_fields(view, edges)
        vocabs: list[dict[str, int]] = []
        for column in raw:
            vocab: dict[str, int] = {}
            for value in column:
                key = str(value)
                if key not in vocab:
                    vocab[key] = len(vocab)
            vocabs.append(vocab)
        unknown = np.asarray([len(vocab) for vocab in vocabs], dtype=np.int32)
        field_dims = np.asarray([len(vocab) + 1 for vocab in vocabs], dtype=np.int32)
        offsets = np.cumsum(np.concatenate(([0], field_dims[:-1]))).astype(np.int32)
        return cls(edges, vocabs, unknown, offsets, int(field_dims.sum()))

    @staticmethod
    def _raw_fields(view: FeatureView, edges: np.ndarray) -> list[np.ndarray]:
        duration_bucket = np.searchsorted(edges, view.arrays["duration_ms"]).astype(str)
        return [
            view.arrays["user_id"],
            view.arrays["video_id"],
            view.arrays["author_id"],
            view.arrays["tab"],
            duration_bucket,
        ]

    def transform(self, view: FeatureView) -> np.ndarray:
        raw = self._raw_fields(view, self.edges)
        encoded = np.empty((view.rows, len(FIELDS)), dtype=np.int32)
        for field_index, column in enumerate(raw):
            vocab = self.vocabs[field_index]
            encoded[:, field_index] = np.fromiter(
                (vocab.get(str(value), int(self.unknown[field_index])) for value in column),
                dtype=np.int32,
                count=view.rows,
            ) + self.offsets[field_index]
        return encoded

    def to_json(self) -> dict[str, Any]:
        return {
            "edges": self.edges.tolist(),
            "vocabs": self.vocabs,
            "unknown": self.unknown.tolist(),
            "offsets": self.offsets.tolist(),
            "dimension": self.dimension,
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "CategoricalEncoder":
        return cls(
            edges=np.asarray(value["edges"], dtype=np.float64),
            vocabs=[{str(k): int(v) for k, v in item.items()} for item in value["vocabs"]],
            unknown=np.asarray(value["unknown"], dtype=np.int32),
            offsets=np.asarray(value["offsets"], dtype=np.int32),
            dimension=int(value["dimension"]),
        )


class FM:
    def __init__(self, dimension: int, k: int, lr: float, l2: float, seed: int):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dimension, k)).astype(np.float32)
        self.W = np.zeros(dimension, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr = lr
        self.l2 = l2
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        embeddings = self.V[features]
        summed = embeddings.sum(axis=1)
        interaction = 0.5 * (
            (summed**2).sum(axis=1) - (embeddings**2).sum(axis=(1, 2))
        )
        return self.b + self.W[features].sum(axis=1) + interaction, embeddings, summed

    def step(self, features: np.ndarray, targets: np.ndarray) -> float:
        batch = len(targets)
        logits, embeddings, summed = self.logits(features)
        gradient = ((sigmoid(logits) - targets) / batch).astype(np.float32)
        self.apply_score_gradients(features, gradient, embeddings=embeddings, summed=summed)
        probabilities = sigmoid(logits)
        return float(
            -np.mean(
                targets * np.log(probabilities + 1e-9)
                + (1 - targets) * np.log(1 - probabilities + 1e-9)
            )
        )

    def apply_score_gradients(
        self,
        features: np.ndarray,
        score_gradients: np.ndarray,
        *,
        embeddings: np.ndarray | None = None,
        summed: np.ndarray | None = None,
    ) -> None:
        if embeddings is None or summed is None:
            _, embeddings, summed = self.logits(features)
        gradient = np.asarray(score_gradients, dtype=np.float32)
        gradient_v = np.zeros_like(self.V)
        gradient_w = np.zeros_like(self.W)
        np.add.at(gradient_w, features, gradient[:, None])
        np.add.at(
            gradient_v,
            features,
            gradient[:, None, None] * (summed[:, None, :] - embeddings),
        )
        gradient_v += self.l2 * self.V
        gradient_w += self.l2 * self.W
        self.t += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for parameter, grad, momentum, variance in (
            (self.V, gradient_v, self.mV, self.vV),
            (self.W, gradient_w, self.mW, self.vW),
        ):
            momentum *= beta1
            momentum += (1 - beta1) * grad
            variance *= beta2
            variance += (1 - beta2) * (grad * grad)
            parameter -= self.lr * (momentum / (1 - beta1**self.t)) / (
                np.sqrt(variance / (1 - beta2**self.t)) + epsilon
            )
        self.b -= self.lr * gradient.sum()

    def predict(self, features: np.ndarray, batch_size: int = 200_000) -> np.ndarray:
        return np.concatenate(
            [self.logits(features[i : i + batch_size])[0] for i in range(0, len(features), batch_size)]
        )


class OfficialFMPlugin:
    """Worker-safe FM using train-loss early stopping; baseline verifier can use fixed epochs."""

    def fit(
        self,
        train_features: FeatureView,
        train_targets: TargetView,
        config: dict[str, Any],
        seed: int,
        output_dir: Path,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        encoder = CategoricalEncoder.fit(train_features)
        encoded = encoder.transform(train_features)
        model = FM(
            encoder.dimension,
            k=int(config.get("k", 16)),
            lr=float(config.get("lr", 0.001)),
            l2=float(config.get("l2", 1e-6)),
            seed=seed,
        )
        epochs = int(config.get("epochs", 8))
        batch_size = int(config.get("batch_size", 8192))
        rng = np.random.default_rng(seed)
        history: list[float] = []
        for _ in range(epochs):
            order = rng.permutation(train_features.rows)
            losses = []
            for offset in range(0, len(order), batch_size):
                batch = order[offset : offset + batch_size]
                losses.append(model.step(encoded[batch], train_targets.labels[batch]))
            mean_loss = float(np.mean(losses))
            if not np.isfinite(mean_loss):
                raise FloatingPointError("FM training produced non-finite loss")
            history.append(mean_loss)
        path = output_dir / "model.npz"
        np.savez_compressed(path, V=model.V, W=model.W, b=np.asarray([model.b]), t=np.asarray([model.t]))
        (output_dir / "encoder.json").write_text(
            json.dumps(encoder.to_json(), sort_keys=True), encoding="utf-8"
        )
        (output_dir / "training.json").write_text(
            json.dumps({"loss": history, "seed": seed, "config": config}, indent=2),
            encoding="utf-8",
        )
        return path

    def predict(
        self,
        model_artifact: Path,
        features: FeatureView,
        config: dict[str, Any],
        output_dir: Path,
    ) -> np.ndarray:
        encoder = CategoricalEncoder.from_json(
            json.loads((model_artifact.parent / "encoder.json").read_text(encoding="utf-8"))
        )
        with np.load(model_artifact, allow_pickle=False) as saved:
            model = FM(encoder.dimension, saved["V"].shape[1], 0.001, 1e-6, 0)
            model.V = saved["V"]
            model.W = saved["W"]
            model.b = np.float32(saved["b"][0])
            model.t = int(saved["t"][0])
        return model.predict(encoder.transform(features))
