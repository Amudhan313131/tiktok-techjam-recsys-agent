"""Crash-resumable orchestration of the constrained patch-validation transaction."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rex.agents.patch_guard import PatchPolicy
from rex.agents.services import AgentDecision, CodingService, ProposalService
from rex.agents.static_audit import audit_changed_files, audit_fixture_bias_only
from rex.agents.workspace import GitWorkspace
from rex.contracts import ExperimentProposal, ExperimentState
from rex.execution.artifacts import artifact_ref, atomic_write_json
from rex.store.repository import ExperimentRepository


@dataclass(frozen=True)
class PreparedExperiment:
    proposal: ExperimentProposal
    workspace: GitWorkspace
    commit_sha: str
    log_artifact_ids: tuple[str, ...]


class PatchTransactionCoordinator:
    """Prepare one hypothesis through static and fixture gates.

    Training/evaluation rungs remain subprocess requests handled by the protected
    runner. Keeping patch preparation separate makes kill-and-resume idempotent.
    """

    def __init__(
        self,
        *,
        repository: ExperimentRepository,
        proposal_service: ProposalService,
        coding_service: CodingService,
        project_root: str | Path,
        worktree_root: str | Path,
        patch_policy: PatchPolicy,
        static_command: tuple[str, ...] = ("python3", "-m", "compileall", "-q", "src"),
        fixture_command: tuple[str, ...] = ("python3", "-m", "pytest", "-q", "tests/fixture"),
        command_timeout_seconds: int = 120,
    ):
        self.repository = repository
        self.proposal_service = proposal_service
        self.coding_service = coding_service
        self.project_root = Path(project_root).resolve()
        self.worktree_root = Path(worktree_root).resolve()
        self.patch_policy = patch_policy
        self.static_command = static_command
        self.fixture_command = fixture_command
        self.command_timeout_seconds = command_timeout_seconds

    def prepare(
        self,
        *,
        run_id: str,
        parent_commit: str,
        proposal_context: dict[str, Any],
        coding_context: dict[str, Any],
        max_hypotheses: int = 50,
    ) -> PreparedExperiment:
        proposal_decision = self.proposal_service.propose(proposal_context)
        proposal = ExperimentProposal.model_validate(proposal_decision.parsed)
        self.repository.create_experiment(
            run_id,
            proposal,
            parent_commit,
            max_hypotheses=max_hypotheses,
            experiment_kind="fixture" if proposal_context.get("fixture_only") else None,
        )
        artifact_dir = self.worktree_root / "_artifacts" / proposal.experiment_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        self._record_llm_decision(
            run_id=run_id,
            experiment_id=proposal.experiment_id,
            role="proposal",
            context=proposal_context,
            decision=proposal_decision,
            artifact_dir=artifact_dir,
        )
        workspace = GitWorkspace.create(
            self.project_root, self.worktree_root, proposal.experiment_id, parent_commit
        )
        self.repository.record_experiment_workspace(
            proposal.experiment_id,
            workspace_path=str(workspace.root),
            branch_name=workspace.branch,
        )
        self._transition(
            proposal.experiment_id,
            ExperimentState.PROPOSED,
            ExperimentState.WORKTREE_READY,
            "worktree-ready",
            {"branch": workspace.branch, "path": str(workspace.root)},
        )
        try:
            patch_decision = self.coding_service.create_patch(proposal, coding_context)
            self._record_llm_decision(
                run_id=run_id,
                experiment_id=proposal.experiment_id,
                role="patch",
                context={"proposal": proposal.model_dump(mode="json"), **coding_context},
                decision=patch_decision,
                artifact_dir=artifact_dir,
            )
            patch = patch_decision.parsed.patch  # type: ignore[attr-defined]
            paths = workspace.apply(patch, self.patch_policy, proposal.files_to_change)
            audit_changed_files(workspace.root, paths)
            if coding_context.get("fixture_only"):
                audit_fixture_bias_only(self.project_root, workspace.root, paths)
            patch_path = artifact_dir / "patch.diff"
            patch_path.write_text(patch, encoding="utf-8")
            patch_ref = artifact_ref(patch_path, "patch")
            self.repository.register_artifact(patch_ref, experiment_id=proposal.experiment_id)
            self._transition(
                proposal.experiment_id,
                ExperimentState.WORKTREE_READY,
                ExperimentState.PATCHED,
                "patched",
                {"paths": paths, "patch_artifact_id": patch_ref.artifact_id},
            )
            static_ref = self._run_gate(
                proposal.experiment_id, workspace.root, artifact_dir, "static", self.static_command
            )
            self._transition(
                proposal.experiment_id,
                ExperimentState.PATCHED,
                ExperimentState.STATIC_VALID,
                "static-valid",
                {"log_artifact_id": static_ref.artifact_id},
            )
            fixture_ref = self._run_gate(
                proposal.experiment_id, workspace.root, artifact_dir, "fixture", self.fixture_command
            )
            self._transition(
                proposal.experiment_id,
                ExperimentState.STATIC_VALID,
                ExperimentState.FIXTURE_VALID,
                "fixture-valid",
                {"log_artifact_id": fixture_ref.artifact_id},
            )
            commit_sha = workspace.commit(f"rex: {proposal.experiment_id} {proposal.primary_change}")
            self.repository.record_experiment_workspace(
                proposal.experiment_id,
                workspace_path=str(workspace.root),
                branch_name=workspace.branch,
                commit_sha=commit_sha,
            )
            return PreparedExperiment(
                proposal=proposal,
                workspace=workspace,
                commit_sha=commit_sha,
                log_artifact_ids=(patch_ref.artifact_id, static_ref.artifact_id, fixture_ref.artifact_id),
            )
        except BaseException as error:
            current = ExperimentState(self.repository.get_experiment(proposal.experiment_id)["state"])
            if ExperimentState.FAILED_REPAIRABLE in _allowed_failure_targets(current):
                self._transition(
                    proposal.experiment_id,
                    current,
                    ExperimentState.FAILED_REPAIRABLE,
                    "prepare-failed",
                    {"reason": f"{type(error).__name__}: {str(error)[-1000:]}"},
                )
            raise

    def _record_llm_decision(
        self,
        *,
        run_id: str,
        experiment_id: str,
        role: str,
        context: dict[str, Any],
        decision: AgentDecision,
        artifact_dir: Path,
    ) -> None:
        request_path = artifact_dir / f"{role}-request.json"
        response_path = artifact_dir / f"{role}-response.json"
        atomic_write_json(request_path, {"role": role, "context": context})
        response = decision.response
        atomic_write_json(
            response_path,
            {
                "provider": response.provider,
                "model": response.model,
                "request_id": response.request_id,
                "value": response.value,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "wall_seconds": response.wall_seconds,
                "attempts": response.attempts,
                "schema_valid": response.schema_valid,
                "fallback_chain": list(response.fallback_chain),
                "fallback_errors": list(response.fallback_errors),
                "raw_response": response.raw_response,
                "stdout": response.stdout,
                "stderr": response.stderr,
            },
        )
        request_ref = artifact_ref(request_path, "llm_request")
        response_ref = artifact_ref(response_path, "llm_response")
        self.repository.register_artifact(request_ref, experiment_id=experiment_id)
        self.repository.register_artifact(response_ref, experiment_id=experiment_id)
        self.repository.record_llm_call(
            call_id=f"{experiment_id}:{role}",
            run_id=run_id,
            experiment_id=experiment_id,
            role=role,
            provider=response.provider,
            model=response.model,
            request_artifact_id=request_ref.artifact_id,
            response_artifact_id=response_ref.artifact_id,
            schema_valid=response.schema_valid,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            wall_seconds=response.wall_seconds,
            request_id=response.request_id,
        )

    def _run_gate(
        self,
        experiment_id: str,
        cwd: Path,
        artifact_dir: Path,
        name: str,
        command: tuple[str, ...],
    ):
        log_path = artifact_dir / f"{name}.log"
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                text=True,
                capture_output=True,
                timeout=self.command_timeout_seconds,
            )
            output = f"$ {' '.join(command)}\n{result.stdout}\n{result.stderr}"
        except subprocess.TimeoutExpired as error:
            output = f"$ {' '.join(command)}\nTIMEOUT\n{error.stdout or ''}\n{error.stderr or ''}"
            log_path.write_text(output, encoding="utf-8")
            raise RuntimeError(f"{name} gate timed out") from error
        log_path.write_text(output, encoding="utf-8")
        ref = artifact_ref(log_path, f"{name}_log")
        self.repository.register_artifact(ref, experiment_id=experiment_id)
        if result.returncode:
            raise RuntimeError(f"{name} gate failed with exit {result.returncode}: {result.stderr[-1000:]}")
        return ref

    def _transition(
        self,
        experiment_id: str,
        current: ExperimentState,
        target: ExperimentState,
        suffix: str,
        payload: dict[str, Any],
    ) -> None:
        self.repository.transition_experiment(
            experiment_id,
            current,
            target,
            payload=payload,
            idempotency_key=f"{experiment_id}:{suffix}",
        )


def _allowed_failure_targets(state: ExperimentState) -> set[ExperimentState]:
    from rex.control.state_machine import EXPERIMENT_TRANSITIONS

    return EXPERIMENT_TRANSITIONS[state]
