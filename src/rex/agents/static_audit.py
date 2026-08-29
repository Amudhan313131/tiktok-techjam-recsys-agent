"""AST-based capability audit for code authored inside the patch allowlist."""

from __future__ import annotations

import ast
import math
from pathlib import Path


class StaticAuditRejected(RuntimeError):
    pass


DENIED_IMPORT_PREFIXES = (
    "rex.data.bootstrap",
    "rex.evaluation",
    "rex.store",
    "rex.control",
    "subprocess",
)
DENIED_STRING_FRAGMENTS = (
    "label_vault",
    "valid_targets",
    "test_targets",
    "long_view",
    "submit.py --score",
)


def audit_python_file(path: str | Path) -> None:
    candidate = Path(path)
    source = candidate.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(candidate))
    except SyntaxError as error:
        raise StaticAuditRejected(f"syntax error in {candidate}: {error}") from error
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            names = []
        for name in names:
            if any(
                name == denied or name.startswith(denied + ".")
                for denied in DENIED_IMPORT_PREFIXES
            ):
                violations.append(f"line {node.lineno}: denied import {name}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.lower()
            for fragment in DENIED_STRING_FRAGMENTS:
                if fragment.lower() in lowered:
                    violations.append(
                        f"line {node.lineno}: denied capability string {fragment!r}"
                    )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"eval", "exec", "compile"}
        ):
            violations.append(f"line {node.lineno}: dynamic code execution is forbidden")
    if violations:
        raise StaticAuditRejected(
            f"static capability audit failed for {candidate}: " + "; ".join(violations)
        )


def audit_changed_files(worktree: str | Path, relative_paths: tuple[str, ...]) -> None:
    root = Path(worktree).resolve()
    for relative in relative_paths:
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root):
            raise StaticAuditRejected(f"changed path escapes worktree: {relative}")
        if candidate.suffix == ".py" and candidate.exists():
            audit_python_file(candidate)


def audit_fixture_bias_only(
    parent_root: str | Path,
    worktree: str | Path,
    relative_paths: tuple[str, ...],
) -> float:
    """Allow a live fixture patch to change one bounded numeric literal only.

    This semantic equality check is the fixture phase's untrusted-code boundary:
    the candidate file must be byte-identical to the trusted parent after the
    single ``DEFAULT_BIAS`` line is normalized.
    """

    expected = "src/rex/models/experimental/fixture.py"
    if relative_paths != (expected,):
        raise StaticAuditRejected("fixture patches may change only the fixture model")
    parent = (Path(parent_root).resolve() / expected).read_text(encoding="utf-8")
    candidate = (Path(worktree).resolve() / expected).read_text(encoding="utf-8")
    parent_lines = parent.splitlines(keepends=True)
    candidate_lines = candidate.splitlines(keepends=True)
    if len(parent_lines) != len(candidate_lines):
        raise StaticAuditRejected("fixture patch changed the trusted file structure")
    changed = [
        index
        for index, (before, after) in enumerate(zip(parent_lines, candidate_lines, strict=True))
        if before != after
    ]
    expected_lines = [
        index for index, line in enumerate(parent_lines) if line.strip() == "DEFAULT_BIAS = 0.0"
    ]
    if len(expected_lines) != 1 or changed != expected_lines:
        raise StaticAuditRejected("fixture patch must change only DEFAULT_BIAS = 0.0")
    changed_line = candidate_lines[changed[0]].strip()
    prefix = "DEFAULT_BIAS = "
    if not changed_line.startswith(prefix):
        raise StaticAuditRejected("fixture patch removed DEFAULT_BIAS")
    try:
        value = ast.literal_eval(changed_line[len(prefix) :])
    except (SyntaxError, ValueError) as error:
        raise StaticAuditRejected("fixture bias must be a numeric literal") from error
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StaticAuditRejected("fixture bias must be a numeric literal")
    numeric = float(value)
    if not math.isfinite(numeric) or abs(numeric) > 0.25:
        raise StaticAuditRejected("fixture bias must be finite and within [-0.25, 0.25]")
    return numeric
