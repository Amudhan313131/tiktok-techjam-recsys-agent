from __future__ import annotations

from pathlib import Path

import pytest

from rex.agents.recovery import RepairAction, plan_repair
from rex.agents.search_policy import SearchPolicy
from rex.contracts import AttemptStatus
from rex.control.production_supervisor import ProductionRunConfig
from rex.data.manifest import repo_root


def test_shipped_production_config_is_runnable_and_excludes_unsupported_cards() -> None:
    config = ProductionRunConfig.load(repo_root() / "configs/run/production.yaml")
    assert config.scientific_execution_enabled
    assert config.method_cards["E15"].feature_recipe == "control"
    assert "E09" not in config.method_cards
    assert not ({"E11", "E12", "E13", "E14"} & set(config.method_cards))
    assert config.method_cards["E10"].feature_recipe == "shadow_blend"
    assert config.method_cards["E03"].feature_recipe == "history_length"
    assert config.method_cards["E07"].feature_recipe == "author_duration_affinity"
    assert config.method_cards["E08"].feature_recipe == "recency_history"
    assert config.scientific_execution == {
        "max_parallel_workers": 4,
        "max_parallel_folds": 3,
        "parallel_candidate_control": True,
        "max_memory_mb": 1536,
    }
    assert config.baseline_cache_dir is None
    assert config.control_cache_dir is None


def test_production_loader_honors_docker_capability_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    data = tmp_path / "data"
    baseline_cache = runs / "cache" / "baseline"
    control_cache = runs / "cache" / "controls"
    linux_lock = repo_root() / "requirements-lock-linux-arm64.txt"
    monkeypatch.setenv("REX_SOURCE_ROOT", str(repo_root()))
    monkeypatch.setenv("REX_RUNS_ROOT", str(runs))
    monkeypatch.setenv("REX_DATA_ROOT", str(data))
    monkeypatch.setenv("REX_BASELINE_CACHE_DIR", str(baseline_cache))
    monkeypatch.setenv("REX_CONTROL_CACHE_DIR", str(control_cache))
    monkeypatch.setenv("REX_ENVIRONMENT_LOCK", str(linux_lock))

    config = ProductionRunConfig.load(repo_root() / "configs/run/production.yaml")

    assert config.project_root == repo_root()
    assert config.runs_dir == runs
    assert config.raw_data_dir == data
    assert config.data_manifest == runs / "data" / "data_manifest.json"
    assert config.baseline_cache_dir == baseline_cache
    assert config.control_cache_dir == control_cache
    assert config.environment_lock == linux_lock


def test_proposal_context_contains_versioned_method_card_and_only_evidence_ids() -> None:
    card = next(item for item in SearchPolicy().cards if item.card_id == "E01")
    context = SearchPolicy.proposal_context(
        card,
        evidence_artifact_ids=["evidence-b", "evidence-a"],
        incumbent_experiment_id="incumbent",
        incumbent_primary_units=600_000_000,
        hypotheses_remaining=49,
        seconds_remaining=100,
    )
    assert context["method_card"]["citation_id"] == "method-card:1.0:E01"
    assert context["method_card"]["primary_change"]
    assert context["method_card"]["falsifier"]
    assert {item["source_id"] for item in context["method_sources"]} == {
        "ranknet",
        "factorization_machines",
    }
    assert context["evidence_artifact_ids"] == ["evidence-a", "evidence-b"]
    assert set(context["evidence_artifact_ids"]) == {"evidence-a", "evidence-b"}
    assert "evidence_payloads" not in context


def test_blend_requires_promoted_pairwise_and_distinct_tree_history_branch() -> None:
    assert "two_branches_supported" not in SearchPolicy.evidence_flags({"E01"})
    assert "two_branches_supported" not in SearchPolicy.evidence_flags({"E02"})
    assert "two_branches_supported" in SearchPolicy.evidence_flags({"E01", "E02"})


def test_typed_repairs_are_global_and_status_specific() -> None:
    timeout = plan_repair(AttemptStatus.TIMEOUT, 0, phase="cheap")
    oom = plan_repair(AttemptStatus.OOM, 1, phase="full")
    exhausted = plan_repair(AttemptStatus.NAN, 2, phase="official_valid")
    interrupted = plan_repair(AttemptStatus.INTERRUPTED, 2, phase="full")
    assert timeout.action == RepairAction.REDUCE_WORKLOAD
    assert timeout.overrides["workers"] == 1
    assert oom.repair_number == 2
    assert not exhausted.repair
    assert interrupted.action == RepairAction.RESUME
    assert not interrupted.consumes_budget


def test_production_loader_rejects_deferred_neural_card(tmp_path: Path) -> None:
    path = tmp_path / "production.yaml"
    path.write_text(
        "execution_mode: production\n"
        "budget_config: configs/budget.yaml\n"
        "method_cards:\n"
        "  E11:\n"
        "    config: configs/experiments/e11.yaml\n"
        "    feature_recipe: neural_history\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="deferred"):
        ProductionRunConfig.load(path)


def test_production_loader_rejects_unsupported_e09_card(tmp_path: Path) -> None:
    path = tmp_path / "production.yaml"
    path.write_text(
        "execution_mode: production\n"
        "budget_config: configs/budget.yaml\n"
        "method_cards:\n"
        "  E09:\n"
        "    config: configs/experiments/e09.yaml\n"
        "    feature_recipe: candidate_history\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported"):
        ProductionRunConfig.load(path)
