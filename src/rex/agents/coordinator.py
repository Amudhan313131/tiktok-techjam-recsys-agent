"""Crash-resumable orchestration of the constrained patch-validation transaction."""

from __future__ import annotations

import json
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from rex.agents.patch_guard import PatchPolicy, validate_patch
from rex.agents.provider import ProviderResponse
from rex.agents.services import AgentDecision, CodingService, PatchResponse, ProposalService
from rex.agents.static_audit import audit_changed_files, audit_fixture_bias_only
from rex.agents.workspace import GitWorkspace
from rex.contracts import ExperimentProposal, ExperimentState
from rex.execution.artifacts import artifact_ref, atomic_write_json
from rex.execution.gate import execute_gate
from rex.execution.sandbox import SandboxMode
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
        static_command: tuple[str, ...] = (sys.executable, "-m", "compileall", "-q", "src"),
        fixture_command: tuple[str, ...] = (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/fixture",
        ),
        command_timeout_seconds: int = 120,
        sandbox_mode: SandboxMode | str = SandboxMode.FIXTURE,
        trusted_output_root: str | Path | None = None,
        checkpoint: Callable[[str, str], None] | None = None,
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
        self.sandbox_mode = SandboxMode(sandbox_mode)
        self.trusted_output_root = (
            None if trusted_output_root is None else Path(trusted_output_root).resolve()
        )
        self.checkpoint = checkpoint
        if self.sandbox_mode == SandboxMode.PRODUCTION and self.trusted_output_root is None:
            raise ValueError("production patch gates require a trusted output root")

    def prepare(
        self,
        *,
        run_id: str,
        parent_commit: str,
        proposal_context: dict[str, Any],
        coding_context: dict[str, Any],
        max_hypotheses: int = 50,
        external_parent: bool = False,
    ) -> PreparedExperiment:
        proposal, proposal_decision, artifact_dir, proposal_created = self._durable_proposal(
            run_id,
            proposal_context,
            parent_commit,
            max_hypotheses,
            external_parent,
        )
        self._record_llm_decision(
            run_id=run_id,
            experiment_id=proposal.experiment_id,
            role="proposal",
            context=proposal_context,
            decision=proposal_decision,
            artifact_dir=artifact_dir,
        )
        if proposal_created:
            self._checkpoint("proposal_persisted", proposal.experiment_id)
        workspace = self._ensure_workspace(proposal.experiment_id, parent_commit)
        try:
            while True:
                experiment = self.repository.get_experiment(proposal.experiment_id)
                state = ExperimentState(experiment["state"])
                if state == ExperimentState.PROPOSED:
                    self.repository.record_experiment_workspace(
                        proposal.experiment_id,
                        workspace_path=str(workspace.root),
                        branch_name=workspace.branch,
                    )
                    self._transition(
                        proposal.experiment_id,
                        state,
                        ExperimentState.WORKTREE_READY,
                        "worktree-ready",
                        {"branch": workspace.branch, "path": str(workspace.root)},
                    )
                    self._checkpoint("worktree_ready", proposal.experiment_id)
                    continue
                if state == ExperimentState.WORKTREE_READY:
                    patch_context = {
                        "proposal": proposal.model_dump(mode="json"),
                        **coding_context,
                    }
                    patch_decision, generated = self._durable_patch(
                        proposal, patch_context, artifact_dir
                    )
                    if generated:
                        self._checkpoint("patch_decision", proposal.experiment_id)
                    self._record_llm_decision(
                        run_id=run_id,
                        experiment_id=proposal.experiment_id,
                        role="patch",
                        context=patch_context,
                        decision=patch_decision,
                        artifact_dir=artifact_dir,
                    )
                    patch = PatchResponse.model_validate(patch_decision.parsed).patch
                    paths = self._apply_or_verify_patch(workspace, patch, proposal)
                    audit_changed_files(workspace.root, paths)
                    if coding_context.get("fixture_only"):
                        audit_fixture_bias_only(self.project_root, workspace.root, paths)
                    patch_path = artifact_dir / "patch.diff"
                    patch_path.write_text(patch, encoding="utf-8")
                    patch_ref = artifact_ref(patch_path, "patch")
                    self.repository.register_artifact(
                        patch_ref, experiment_id=proposal.experiment_id
                    )
                    self._checkpoint("patch_applied", proposal.experiment_id)
                    self._transition(
                        proposal.experiment_id,
                        state,
                        ExperimentState.PATCHED,
                        "patched",
                        {"paths": paths, "patch_artifact_id": patch_ref.artifact_id},
                    )
                    self._checkpoint("patched", proposal.experiment_id)
                    continue
                if state == ExperimentState.PATCHED:
                    static_ref = self._run_gate(
                        proposal.experiment_id,
                        workspace.root,
                        artifact_dir,
                        "static",
                        self.static_command,
                    )
                    self._transition(
                        proposal.experiment_id,
                        state,
                        ExperimentState.STATIC_VALID,
                        "static-valid",
                        {"log_artifact_id": static_ref.artifact_id},
                    )
                    self._checkpoint("static_valid", proposal.experiment_id)
                    continue
                if state == ExperimentState.STATIC_VALID:
                    fixture_ref = self._run_gate(
                        proposal.experiment_id,
                        workspace.root,
                        artifact_dir,
                        "fixture",
                        self.fixture_command,
                    )
                    self._transition(
                        proposal.experiment_id,
                        state,
                        ExperimentState.FIXTURE_VALID,
                        "fixture-valid",
                        {"log_artifact_id": fixture_ref.artifact_id},
                    )
                    self._checkpoint("fixture_valid", proposal.experiment_id)
                    continue
                if state != ExperimentState.FIXTURE_VALID:
                    raise RuntimeError(
                        f"cannot resume patch transaction from experiment state {state}"
                    )
                commit_sha = str(experiment.get("commit_sha") or "")
                if not commit_sha:
                    commit_sha = self._commit_or_recover(workspace, proposal, parent_commit)
                    self.repository.record_experiment_workspace(
                        proposal.experiment_id,
                        workspace_path=str(workspace.root),
                        branch_name=workspace.branch,
                        commit_sha=commit_sha,
                    )
                    self._checkpoint("committed", proposal.experiment_id)
                log_artifact_ids = self._preparation_artifact_ids(proposal.experiment_id)
                return PreparedExperiment(
                    proposal=proposal,
                    workspace=workspace,
                    commit_sha=commit_sha,
                    log_artifact_ids=log_artifact_ids,
                )
        except Exception as error:
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

    def _apply_or_verify_patch(
        self,
        workspace: GitWorkspace,
        patch: str,
        proposal: ExperimentProposal,
    ) -> tuple[str, ...]:
        expected = validate_patch(
            patch,
            self.patch_policy,
            declared_files=proposal.files_to_change,
        )
        status = self._git(
            workspace.root, "status", "--porcelain", "--untracked-files=normal"
        )
        if not status:
            return workspace.apply(patch, self.patch_policy, proposal.files_to_change)
        if any(line.startswith("??") for line in status.splitlines()):
            raise RuntimeError("dirty patch worktree contains unrelated untracked files")
        observed = {
            line.strip()
            for line in self._git(workspace.root, "diff", "--name-only").splitlines()
            if line.strip()
        }
        if observed != set(expected):
            raise RuntimeError("dirty patch worktree differs from the durable patch paths")
        reverse = subprocess.run(
            ["git", "apply", "--reverse", "--check", "-"],
            cwd=workspace.root,
            input=patch,
            text=True,
            capture_output=True,
            check=False,
        )
        if reverse.returncode != 0:
            raise RuntimeError("dirty patch worktree does not match the durable applied patch")
        return expected

    def _durable_proposal(
        self,
        run_id: str,
        context: dict[str, Any],
        parent_commit: str,
        max_hypotheses: int,
        external_parent: bool,
    ) -> tuple[ExperimentProposal, AgentDecision, Path, bool]:
        expected_id = str(context.get("experiment_id") or "")
        with self.repository.database.connect() as connection:
            rows = connection.execute(
                "SELECT experiment_id,proposal_json FROM experiments WHERE run_id=? AND state NOT IN "
                "('PROMOTED','REJECTED','ABANDONED','FAILED_FINAL') "
                "ORDER BY iteration_number",
                (run_id,),
            ).fetchall()
        if expected_id:
            mismatched = [row for row in rows if row["experiment_id"] != expected_id]
            if mismatched:
                raise RuntimeError(
                    "another active patch transaction exists for this run: "
                    + ", ".join(str(row["experiment_id"]) for row in mismatched)
                )
            rows = [row for row in rows if row["experiment_id"] == expected_id]
        if len(rows) > 1:
            raise RuntimeError("patch transaction database contains multiple proposals")
        if rows:
            experiment_id = str(rows[0]["experiment_id"])
            if expected_id and expected_id != experiment_id:
                raise RuntimeError("durable patch transaction proposal identity drifted")
            artifact_dir = self.worktree_root / "_artifacts" / experiment_id
            decision = self._load_llm_decision("proposal", artifact_dir)
            proposal = ExperimentProposal.model_validate(decision.parsed)
            return proposal, decision, artifact_dir, False

        artifact_dir = self.worktree_root / "_artifacts" / expected_id if expected_id else None
        response_path = None if artifact_dir is None else artifact_dir / "proposal-response.json"
        if response_path is not None and response_path.is_file():
            decision = self._load_llm_decision("proposal", artifact_dir)
        else:
            decision = self.proposal_service.propose(context)
            proposal_value = ExperimentProposal.model_validate(decision.parsed)
            artifact_dir = self.worktree_root / "_artifacts" / proposal_value.experiment_id
            self._persist_llm_decision_files("proposal", context, decision, artifact_dir)
            self._checkpoint("proposal_boundary", proposal_value.experiment_id)
        proposal = ExperimentProposal.model_validate(decision.parsed)
        self._validate_allowed_files(proposal, context)
        durable = proposal.model_copy(update={"parent_id": None}) if external_parent else proposal
        self.repository.create_experiment(
            run_id,
            durable,
            parent_commit,
            max_hypotheses=max_hypotheses,
            experiment_kind="fixture" if context.get("fixture_only") else None,
        )
        return proposal, decision, artifact_dir, True

    def _durable_patch(
        self,
        proposal: ExperimentProposal,
        context: dict[str, Any],
        artifact_dir: Path,
    ) -> tuple[AgentDecision, bool]:
        if (artifact_dir / "patch-response.json").is_file():
            return self._load_llm_decision("patch", artifact_dir), False
        decision = self.coding_service.create_patch(proposal, context)
        self._persist_llm_decision_files("patch", context, decision, artifact_dir)
        return decision, True

    @staticmethod
    def _validate_allowed_files(
        proposal: ExperimentProposal,
        context: dict[str, Any],
    ) -> None:
        allowed_files = context.get("allowed_files")
        if not isinstance(allowed_files, list):
            return
        unexpected = sorted(set(proposal.files_to_change).difference(allowed_files))
        if unexpected:
            raise RuntimeError(
                "proposal requested files outside its method-card allowlist: "
                + ", ".join(unexpected)
            )

    def _ensure_workspace(self, experiment_id: str, parent_commit: str) -> GitWorkspace:
        experiment = self.repository.get_experiment(experiment_id)
        expected_root = (self.worktree_root / experiment_id).resolve()
        root = Path(str(experiment.get("workspace_path") or expected_root)).resolve()
        try:
            root.relative_to(self.worktree_root)
        except ValueError as error:
            raise RuntimeError("durable patch worktree escaped its trusted root") from error
        branch = str(experiment.get("branch_name") or f"codex/rex-{experiment_id}")
        if not root.exists():
            root.parent.mkdir(parents=True, exist_ok=True)
            branch_exists = bool(self._git(self.project_root, "branch", "--list", branch))
            command = (
                ["git", "worktree", "add", str(root), branch]
                if branch_exists
                else ["git", "worktree", "add", "-b", branch, str(root), parent_commit]
            )
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(f"cannot restore patch worktree: {result.stderr[-1000:]}")
        workspace = GitWorkspace(root, branch)
        state = ExperimentState(experiment["state"])
        head = self._git(root, "rev-parse", "HEAD")
        if state == ExperimentState.PROPOSED and head != parent_commit:
            raise RuntimeError("new patch worktree is not at its durable parent commit")
        return workspace

    def _commit_or_recover(
        self,
        workspace: GitWorkspace,
        proposal: ExperimentProposal,
        parent_commit: str,
    ) -> str:
        dirty = bool(self._git(workspace.root, "status", "--porcelain", "--untracked-files=normal"))
        if dirty:
            return workspace.commit(f"rex: {proposal.experiment_id} {proposal.primary_change}")
        head = self._git(workspace.root, "rev-parse", "HEAD")
        if head == parent_commit:
            raise RuntimeError("fixture-valid patch transaction has no committed or pending change")
        return head

    def _preparation_artifact_ids(self, experiment_id: str) -> tuple[str, ...]:
        with self.repository.database.connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT artifact.artifact_id FROM artifact_links link JOIN artifacts artifact "
                "ON artifact.artifact_id=link.artifact_id WHERE link.experiment_id=? AND "
                "artifact.kind IN ('patch','static_log','fixture_log') ORDER BY artifact.kind",
                (experiment_id,),
            ).fetchall()
        return tuple(str(row["artifact_id"]) for row in rows)

    @staticmethod
    def _git(cwd: Path, *arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments], cwd=cwd, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(arguments)} failed: {result.stderr[-1000:]}")
        return result.stdout.strip()

    def _checkpoint(self, name: str, experiment_id: str) -> None:
        if self.checkpoint is not None:
            self.checkpoint(name, experiment_id)

    def _persist_llm_decision_files(
        self,
        role: str,
        context: dict[str, Any],
        decision: AgentDecision,
        artifact_dir: Path,
    ) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        request_path = artifact_dir / f"{role}-request.json"
        response_path = artifact_dir / f"{role}-response.json"
        if not request_path.exists():
            atomic_write_json(request_path, {"role": role, "context": context})
        if response_path.exists():
            return
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

    @staticmethod
    def _load_llm_decision(role: str, artifact_dir: Path) -> AgentDecision:
        path = artifact_dir / f"{role}-response.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        response = ProviderResponse(
            value=dict(payload["value"]),
            provider=str(payload["provider"]),
            model=str(payload["model"]),
            input_tokens=int(payload.get("input_tokens", 0)),
            output_tokens=int(payload.get("output_tokens", 0)),
            request_id=payload.get("request_id"),
            wall_seconds=float(payload.get("wall_seconds", 0)),
            attempts=int(payload.get("attempts", 1)),
            schema_valid=bool(payload.get("schema_valid", True)),
            raw_response=payload.get("raw_response"),
            stdout=payload.get("stdout"),
            stderr=payload.get("stderr"),
            fallback_chain=tuple(payload.get("fallback_chain", [])),
            fallback_errors=tuple(payload.get("fallback_errors", [])),
        )
        parsed = (
            ExperimentProposal.model_validate(response.value)
            if role == "proposal"
            else PatchResponse.model_validate(response.value)
        )
        return AgentDecision(parsed, response)

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
        self._persist_llm_decision_files(role, context, decision, artifact_dir)
        response = decision.response
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
        result = execute_gate(
            name=name,
            command=command,
            workspace=cwd,
            artifact_dir=artifact_dir,
            timeout_seconds=self.command_timeout_seconds,
            sandbox_mode=self.sandbox_mode,
            trusted_worktree_root=(
                self.worktree_root if self.sandbox_mode == SandboxMode.PRODUCTION else None
            ),
            trusted_output_root=self.trusted_output_root,
        )
        output = f"$ {' '.join(command)}\n{result.stdout}\n{result.stderr}"
        log_path.write_text(output, encoding="utf-8")
        ref = artifact_ref(log_path, f"{name}_log")
        self.repository.register_artifact(ref, experiment_id=experiment_id)
        evidence_ref = artifact_ref(result.evidence_path, f"{name}_sandbox_evidence")
        self.repository.register_artifact(evidence_ref, experiment_id=experiment_id)
        if result.profile_path is not None:
            profile_ref = artifact_ref(result.profile_path, f"{name}_sandbox_profile")
            self.repository.register_artifact(profile_ref, experiment_id=experiment_id)
        if result.timed_out:
            raise RuntimeError(f"{name} gate timed out")
        if result.return_code:
            raise RuntimeError(
                f"{name} gate failed with exit {result.return_code}: {result.stderr[-1000:]}"
            )
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
