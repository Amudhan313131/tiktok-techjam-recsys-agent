from __future__ import annotations

import pytest
from pydantic import ValidationError

from rex.contracts import Metrics, RunRequest
from rex.control.budget import metric_units, update_metric_trackers
from rex.control.state_machine import EXPERIMENT_TRANSITIONS, IllegalTransition, require_experiment_transition
from rex.contracts import ExperimentState


HASH = "0" * 64


def test_primary_must_equal_component_mean() -> None:
    with pytest.raises(ValidationError):
        Metrics(
            GAUC=0.6,
            **{"nDCG@5": 0.4},
            primary=0.6,
            users=1,
            rows=2,
            evaluator_sha256=HASH,
            split="valid",
        )


def test_predict_request_cannot_receive_test_targets() -> None:
    with pytest.raises(ValidationError):
        RunRequest(
            run_id="r",
            experiment_id="e",
            attempt_id="a",
            commit_sha="c",
            plugin="fixture:Plugin",
            config_path="config.json",
            config_sha256=HASH,
            seed=0,
            rung="predict",
            split="test",
            feature_view_path="features.npz",
            target_view_path="targets.npz",
            output_dir="out",
            deadline_epoch_ms=1,
            timeout_seconds=1,
            data_view_sha256=HASH,
            environment_sha256=HASH,
        )


def test_every_declared_experiment_transition_is_legal() -> None:
    for current, targets in EXPERIMENT_TRANSITIONS.items():
        for target in targets:
            require_experiment_transition(current, target)


def test_terminal_transition_is_rejected() -> None:
    with pytest.raises(IllegalTransition):
        require_experiment_transition(ExperimentState.PROMOTED, ExperimentState.PROPOSED)


def test_best_ever_and_epsilon_plateau_are_independent() -> None:
    update = update_metric_trackers(
        previous_best_units=metric_units(0.6000),
        candidate_primary=0.6010,
        previous_streak=0,
        epsilon_units=metric_units(0.002),
        patience=3,
    )
    assert update.is_new_best is True
    assert update.non_improvement_streak == 1
    assert update.best_primary_units == metric_units(0.6010)


def test_epsilon_patience_converges_on_third_non_improvement() -> None:
    best = metric_units(0.6000)
    streak = 0
    for score in (0.6010, 0.6015, 0.6016):
        update = update_metric_trackers(
            previous_best_units=best,
            candidate_primary=score,
            previous_streak=streak,
            epsilon_units=metric_units(0.002),
            patience=3,
        )
        best, streak = update.best_primary_units, update.non_improvement_streak
    assert update.converged is True
