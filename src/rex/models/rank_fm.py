"""FM trained with fixed-K same-user pairwise logistic loss."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rex.data.groups import sample_same_user_pairs, user_groups
from rex.data.views import FeatureView, TargetView
from rex.losses.ranking import delta_ndcg_weights, pair_logistic_gradient, pair_logistic_loss
from rex.models.official_fm import CategoricalEncoder, FM, OfficialFMPlugin


class RankFMPlugin(OfficialFMPlugin):
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
        pair_batch = int(config.get("pair_batch_size", 4096))
        negatives = int(config.get("negatives_per_positive", 3))
        max_pairs = config.get("max_pairs")
        bce_weight = float(config.get("bce_weight", 0.05))
        rng = np.random.default_rng(seed)
        history: list[dict[str, float]] = []
        for epoch in range(epochs):
            sample = sample_same_user_pairs(
                train_features.arrays["user_id"],
                train_targets.labels,
                negatives_per_positive=negatives,
                seed=seed + epoch,
                max_pairs=int(max_pairs) if max_pairs is not None else None,
            )
            if not len(sample.positive_indices):
                raise ValueError("training data has no discriminative user groups")
            pair_weights: np.ndarray | None = None
            if config.get("pair_weighting") == "delta_ndcg5":
                scores = model.predict(encoded)
                ranks = np.empty(len(scores), dtype=np.int64)
                positive_counts = np.zeros(len(scores), dtype=np.int64)
                for indices in user_groups(train_features.arrays["user_id"]).values():
                    order_for_user = np.argsort(-scores[indices], kind="stable")
                    ranks[indices[order_for_user]] = np.arange(len(indices))
                    positive_counts[indices] = int(train_targets.labels[indices].sum())
                pair_weights = delta_ndcg_weights(
                    ranks[sample.positive_indices],
                    ranks[sample.negative_indices],
                    positive_counts[sample.positive_indices],
                    cutoff=5,
                )
                # Pairs outside the cutoff contain no nDCG signal; retaining a tiny
                # floor avoids an all-zero batch while preserving metric alignment.
                pair_weights = np.maximum(pair_weights, 1e-4)
            order = rng.permutation(len(sample.positive_indices))
            losses: list[float] = []
            for offset in range(0, len(order), pair_batch):
                selection = order[offset : offset + pair_batch]
                pos_index = sample.positive_indices[selection]
                neg_index = sample.negative_indices[selection]
                pos_score = model.logits(encoded[pos_index])[0]
                neg_score = model.logits(encoded[neg_index])[0]
                difference = pos_score - neg_score
                weights = pair_weights[selection] if pair_weights is not None else None
                pair_gradient = pair_logistic_gradient(difference, weights) / len(difference)
                features = np.concatenate((encoded[pos_index], encoded[neg_index]), axis=0)
                score_gradient = np.concatenate((pair_gradient, -pair_gradient), axis=0)
                model.apply_score_gradients(features, score_gradient)
                losses.append(pair_logistic_loss(difference, weights))
            if bce_weight > 0:
                bce_size = min(len(encoded), int(config.get("bce_batch_size", 8192)))
                bce_indices = rng.choice(len(encoded), size=bce_size, replace=False)
                original_lr = model.lr
                model.lr *= bce_weight
                bce_loss = model.step(encoded[bce_indices], train_targets.labels[bce_indices])
                model.lr = original_lr
            else:
                bce_loss = 0.0
            mean_pair = float(np.mean(losses))
            if not np.isfinite(mean_pair):
                raise FloatingPointError("RankFM produced non-finite pair loss")
            history.append(
                {
                    "pair_loss": mean_pair,
                    "bce_loss": bce_loss,
                    "pairs": len(sample.positive_indices),
                    "users_with_pairs": sample.users_with_pairs,
                }
            )
        path = output_dir / "model.npz"
        np.savez_compressed(path, V=model.V, W=model.W, b=np.asarray([model.b]), t=np.asarray([model.t]))
        (output_dir / "encoder.json").write_text(
            json.dumps(encoder.to_json(), sort_keys=True), encoding="utf-8"
        )
        (output_dir / "training.json").write_text(
            json.dumps({"history": history, "seed": seed, "config": config}, indent=2),
            encoding="utf-8",
        )
        return path
