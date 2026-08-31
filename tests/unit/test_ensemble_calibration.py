from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from rex.models.ensemble import (
    BlendSelection,
    PredictionVector,
    ShadowBlendFold,
    SupportShrinkageSpec,
    SupportShrinkageState,
    apply_train_support_shrinkage,
    blend_scores,
    fit_train_support_shrinkage,
    load_blend_selection,
    load_support_shrinkage,
    require_exact_prediction_alignment,
    save_blend_selection,
    save_support_shrinkage,
    select_shadow_blend,
    support_candidate_weights,
)


def _shadow_fold(name: str = "A") -> ShadowBlendFold:
    rows = np.arange(12, dtype=np.int64)
    users = np.repeat(np.asarray(["u1", "u2", "u3"]), 4)
    labels = np.tile(np.asarray([1, 0, 1, 0], dtype=np.float64), 3)
    left = np.asarray(
        [
            -0.8019314252534474,
            -1.324358995628145,
            -0.24836162209524854,
            0.4204452380655215,
            1.1360465324896427,
            0.10970639932180819,
            -0.5526473205362324,
            -0.7847803553442784,
            0.7487457707345911,
            1.6347830429585775,
            0.27276877584472176,
            -1.2333286640307717,
        ]
    )
    right = np.asarray(
        [
            -0.9582652054360887,
            1.6000190889991115,
            0.2028824405086084,
            -1.7321348424395848,
            -0.08369619281702581,
            -1.1632259734447485,
            -0.6292880940615545,
            -0.48800582327685743,
            -0.7133133716322436,
            0.5533784703532895,
            -0.06308597192528916,
            -0.5894312580326048,
        ]
    )
    return ShadowBlendFold(
        name=name,
        row_ids=rows,
        user_ids=users,
        labels=labels,
        left=PredictionVector("fm", rows, users, left),
        right=PredictionVector("tree", rows, users, right),
    )


def test_shadow_blend_selection_is_deterministic_and_regularized() -> None:
    folds = [_shadow_fold("A"), _shadow_fold("B")]
    unregularized = select_shadow_blend(folds, grid_size=11, regularization_strength=0.0)
    repeated = select_shadow_blend(folds, grid_size=11, regularization_strength=0.0)
    conservative = select_shadow_blend(folds, grid_size=11, regularization_strength=0.5)

    assert unregularized.to_json() == repeated.to_json()
    assert unregularized.selection_split == "shadow_only"
    assert unregularized.weights == pytest.approx((0.6, 0.4))
    assert unregularized.selected_mean_primary > unregularized.stronger_single_primary
    assert conservative.stronger_branch == "fm"
    assert conservative.weights == (1.0, 0.0)
    assert len(unregularized.fold_sha256) == 2


def test_shadow_blend_state_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    selection = select_shadow_blend([_shadow_fold()], grid_size=11)
    path = save_blend_selection(tmp_path / "selection.json", selection)

    assert load_blend_selection(path).to_json() == selection.to_json()

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["weights"] = [1.0, 0.0]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="state hash"):
        load_blend_selection(path)


def test_shadow_blend_rejects_row_or_user_misalignment() -> None:
    fold = _shadow_fold()
    moved_rows = fold.right.row_ids.copy()
    moved_rows[[0, 1]] = moved_rows[[1, 0]]
    wrong_rows = PredictionVector("tree", moved_rows, fold.user_ids, fold.right.scores)
    with pytest.raises(ValueError, match="row IDs"):
        require_exact_prediction_alignment(fold.left, wrong_rows)

    moved_users = fold.right.user_ids.copy()
    moved_users[0] = "other"
    wrong_users = PredictionVector("tree", fold.row_ids, moved_users, fold.right.scores)
    with pytest.raises(ValueError, match="user IDs"):
        require_exact_prediction_alignment(fold.left, wrong_users)


def test_shadow_blend_fold_rejects_official_validation() -> None:
    fold = _shadow_fold()
    with pytest.raises(ValueError, match="shadow-only"):
        ShadowBlendFold(
            name="official",
            row_ids=fold.row_ids,
            user_ids=fold.user_ids,
            labels=fold.labels,
            left=fold.left,
            right=fold.right,
            split="official_valid",
        )


def test_prediction_contract_and_blend_fail_closed() -> None:
    with pytest.raises(ValueError, match="unique"):
        PredictionVector(
            "bad",
            np.asarray([1, 1]),
            np.asarray(["u", "u"]),
            np.asarray([0.1, 0.2]),
        )
    with pytest.raises(ValueError, match="NaN or Inf"):
        PredictionVector(
            "bad",
            np.asarray([1]),
            np.asarray(["u"]),
            np.asarray([np.nan]),
        )
    with pytest.raises(ValueError, match="normalization"):
        blend_scores(
            np.asarray(["u"]),
            [np.asarray([0.1])],
            np.asarray([1.0]),
            normalization="unknown",
        )


def test_train_support_shrinkage_uses_counts_and_predeclared_prior() -> None:
    spec = SupportShrinkageSpec(
        key_name="video_id", prior_strength=1.0, minimum_candidate_weight=0.0
    )
    state = fit_train_support_shrinkage(np.asarray(["a", "a", "a", "b"]), spec)
    weights = support_candidate_weights(state, np.asarray(["a", "b", "unseen"])).tolist()
    assert weights == pytest.approx([0.75, 0.5, 0.0])
    assert state.fitted_on == "train_support_only"
    assert state.uses_labels is False

    rows = np.arange(3)
    users = np.asarray(["u", "u", "u"])
    candidate = PredictionVector("candidate", rows, users, np.asarray([1.0, 1.0, 1.0]))
    anchor = PredictionVector("anchor", rows, users, np.asarray([0.0, 0.0, 0.0]))
    result = apply_train_support_shrinkage(
        state, np.asarray(["a", "b", "unseen"]), candidate, anchor
    )
    assert result.tolist() == pytest.approx([0.75, 0.5, 0.0])


def test_support_apply_api_cannot_accept_validation_labels() -> None:
    parameters = inspect.signature(apply_train_support_shrinkage).parameters
    assert "labels" not in parameters
    assert "targets" not in parameters


def test_support_state_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    state = fit_train_support_shrinkage(
        np.asarray([1, 1, 2], dtype=np.int64),
        SupportShrinkageSpec("author_id", prior_strength=4.0, minimum_candidate_weight=0.1),
    )
    path = save_support_shrinkage(tmp_path / "support.json", state)
    restored = load_support_shrinkage(path)
    assert isinstance(restored, SupportShrinkageState)
    assert restored.to_json() == state.to_json()

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["counts"]["int:1"] = 100
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="state hash"):
        load_support_shrinkage(path)


def test_support_shrinkage_requires_aligned_anchor() -> None:
    state = fit_train_support_shrinkage(
        np.asarray(["a"]), SupportShrinkageSpec("video_id", prior_strength=1.0)
    )
    candidate = PredictionVector(
        "candidate", np.asarray([1]), np.asarray(["u"]), np.asarray([0.5])
    )
    anchor = PredictionVector(
        "anchor", np.asarray([2]), np.asarray(["u"]), np.asarray([0.4])
    )
    with pytest.raises(ValueError, match="row IDs"):
        apply_train_support_shrinkage(state, np.asarray(["a"]), candidate, anchor)


def test_shadow_selection_json_rejects_non_shadow_state() -> None:
    state = select_shadow_blend([_shadow_fold()])
    payload = state.to_json()
    payload.pop("state_sha256")
    payload["selection_split"] = "official_valid"
    payload["state_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="state hash"):
        BlendSelection.from_json(payload)
