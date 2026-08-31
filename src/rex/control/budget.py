"""Monotonic budget and official convergence tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import yaml


SCALE = 1_000_000_000


def metric_units(value: float) -> int:
    return int((Decimal(str(value)) * SCALE).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class BudgetConfig:
    max_hypotheses: int
    max_official_evaluations: int
    wall_clock_seconds: int
    finalization_reserve_seconds: int
    convergence_epsilon_units: int
    convergence_patience: int
    max_repairs_per_experiment: int
    default_attempt_timeout_seconds: int
    min_valid_families_before_plateau: int = 3
    cheap_min_probability_positive: float = 0.70
    full_min_pooled_probability_positive: float = 0.90

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BudgetConfig":
        with Path(path).open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        config = cls(
            max_hypotheses=int(raw["max_hypotheses"]),
            max_official_evaluations=int(raw["max_official_evaluations"]),
            wall_clock_seconds=int(raw["wall_clock_seconds"]),
            finalization_reserve_seconds=int(raw["finalization_reserve_seconds"]),
            convergence_epsilon_units=metric_units(float(raw["convergence_epsilon"])),
            convergence_patience=int(raw["convergence_patience"]),
            max_repairs_per_experiment=int(raw["max_repairs_per_experiment"]),
            default_attempt_timeout_seconds=int(raw["default_attempt_timeout_seconds"]),
            min_valid_families_before_plateau=int(raw.get("min_valid_families_before_plateau", 3)),
            cheap_min_probability_positive=float(raw.get("cheap_min_probability_positive", 0.70)),
            full_min_pooled_probability_positive=float(
                raw.get("full_min_pooled_probability_positive", 0.90)
            ),
        )
        if config.min_valid_families_before_plateau < 1:
            raise ValueError("min_valid_families_before_plateau must be positive")
        for name, value in (
            ("cheap_min_probability_positive", config.cheap_min_probability_positive),
            (
                "full_min_pooled_probability_positive",
                config.full_min_pooled_probability_positive,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        return config


@dataclass(frozen=True)
class TrackerUpdate:
    best_primary_units: int
    is_new_best: bool
    non_improvement_streak: int
    converged: bool
    delta_units: int


def update_metric_trackers(
    *,
    previous_best_units: int | None,
    candidate_primary: float,
    previous_streak: int,
    epsilon_units: int,
    patience: int,
) -> TrackerUpdate:
    candidate_units = metric_units(candidate_primary)
    if previous_best_units is None:
        return TrackerUpdate(candidate_units, True, 0, False, candidate_units)
    delta = candidate_units - previous_best_units
    is_new_best = delta > 0
    streak = 0 if delta > epsilon_units else previous_streak + 1
    return TrackerUpdate(
        best_primary_units=max(previous_best_units, candidate_units),
        is_new_best=is_new_best,
        non_improvement_streak=streak,
        converged=streak >= patience,
        delta_units=delta,
    )


def deadline_epoch_ms(wall_clock_seconds: int, now: float | None = None) -> int:
    return int(((time.time() if now is None else now) + wall_clock_seconds) * 1000)


def seconds_remaining(deadline_ms: int, now: float | None = None) -> float:
    return max(0.0, deadline_ms / 1000.0 - (time.time() if now is None else now))


def should_finalize(deadline_ms: int, reserve_seconds: int, now: float | None = None) -> bool:
    return seconds_remaining(deadline_ms, now) <= reserve_seconds


def scientific_plateau_reached(
    *,
    stop_reason: str | None,
    valid_family_count: int,
    minimum_valid_families: int,
) -> bool:
    """A plateau is meaningful only after diverse valid scientific evidence."""

    return stop_reason == "epsilon_plateau" and valid_family_count >= minimum_valid_families
