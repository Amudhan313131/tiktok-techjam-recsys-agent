"""Safe, dependency-free configuration for structured LLM providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


_MODES = {"codex_cli", "claude_cli", "openai_api", "auto", "fixed"}
_AUTO_PROVIDERS = {"codex_cli", "claude_cli", "openai_api", "fixed"}


@dataclass(frozen=True)
class CodexCLIConfig:
    executable: str = "codex"
    model: str | None = None
    working_directory: Path | None = None


@dataclass(frozen=True)
class ClaudeCLIConfig:
    executable: str = "claude"
    model: str | None = None
    working_directory: Path | None = None


@dataclass(frozen=True)
class OpenAIAPIConfig:
    api_key_env: str = "OPENAI_API_KEY"
    model_env: str = "OPENAI_MODEL"
    model: str | None = None
    max_calls_per_run: int = 60
    max_total_tokens: int = 300_000


@dataclass(frozen=True)
class AutoProviderConfig:
    provider_order: tuple[str, ...] = (
        "codex_cli",
        "claude_cli",
        "openai_api",
        "fixed",
    )
    allow_paid_api_fallback: bool = False


@dataclass(frozen=True)
class ProviderConfig:
    mode: str = "codex_cli"
    retries: int = 2
    timeout_seconds: float = 180.0
    max_output_tokens_per_call: int = 3000
    retry_backoff_seconds: float = 0.0
    codex_cli: CodexCLIConfig = field(default_factory=CodexCLIConfig)
    claude_cli: ClaudeCLIConfig = field(default_factory=ClaudeCLIConfig)
    openai_api: OpenAIAPIConfig = field(default_factory=OpenAIAPIConfig)
    auto: AutoProviderConfig = field(default_factory=AutoProviderConfig)


def load_provider_config(value: Mapping[str, Any] | None) -> ProviderConfig:
    """Load the ``llm`` mapping while rejecting inline credentials and typos."""

    raw = dict(value or {})
    _reject_unknown(
        raw,
        {
            "mode",
            "retries",
            "timeout_seconds",
            "max_output_tokens_per_call",
            "retry_backoff_seconds",
            "codex_cli",
            "claude_cli",
            "openai_api",
            "auto",
        },
        "llm",
    )
    mode = str(raw.get("mode", "codex_cli"))
    if mode not in _MODES:
        raise ValueError(f"llm.mode must be one of {sorted(_MODES)}")

    codex_raw = _mapping(raw.get("codex_cli"), "llm.codex_cli")
    _reject_unknown(codex_raw, {"executable", "model", "working_directory"}, "llm.codex_cli")
    working_directory = codex_raw.get("working_directory")
    codex = CodexCLIConfig(
        executable=str(codex_raw.get("executable", "codex")),
        model=_optional_string(codex_raw.get("model")),
        working_directory=Path(working_directory).expanduser() if working_directory else None,
    )

    claude_raw = _mapping(raw.get("claude_cli"), "llm.claude_cli")
    _reject_unknown(
        claude_raw,
        {"executable", "model", "working_directory"},
        "llm.claude_cli",
    )
    claude_working_directory = claude_raw.get("working_directory")
    claude = ClaudeCLIConfig(
        executable=str(claude_raw.get("executable", "claude")),
        model=_optional_string(claude_raw.get("model")),
        working_directory=(
            Path(claude_working_directory).expanduser()
            if claude_working_directory
            else None
        ),
    )

    openai_raw = _mapping(raw.get("openai_api"), "llm.openai_api")
    if "api_key" in openai_raw:
        raise ValueError("llm.openai_api.api_key is forbidden; use OPENAI_API_KEY")
    _reject_unknown(
        openai_raw,
        {
            "api_key_env",
            "model_env",
            "model",
            "store",
            "max_calls_per_run",
            "max_total_tokens",
        },
        "llm.openai_api",
    )
    if openai_raw.get("store", False) is not False:
        raise ValueError("llm.openai_api.store must remain false")
    openai = OpenAIAPIConfig(
        api_key_env=str(openai_raw.get("api_key_env", "OPENAI_API_KEY")),
        model_env=str(openai_raw.get("model_env", "OPENAI_MODEL")),
        model=_optional_string(openai_raw.get("model")),
        max_calls_per_run=_positive_int(
            openai_raw.get("max_calls_per_run", 60), "llm.openai_api.max_calls_per_run"
        ),
        max_total_tokens=_positive_int(
            openai_raw.get("max_total_tokens", 300_000), "llm.openai_api.max_total_tokens"
        ),
    )

    auto_raw = _mapping(raw.get("auto"), "llm.auto")
    _reject_unknown(auto_raw, {"provider_order", "allow_paid_api_fallback"}, "llm.auto")
    provider_order_value = auto_raw.get(
        "provider_order", ("codex_cli", "claude_cli", "openai_api", "fixed")
    )
    if not isinstance(provider_order_value, (list, tuple)):
        raise ValueError("llm.auto.provider_order must be a list")
    provider_order = tuple(str(item) for item in provider_order_value)
    if not provider_order or len(provider_order) != len(set(provider_order)):
        raise ValueError("llm.auto.provider_order must contain unique providers")
    unknown_providers = set(provider_order) - _AUTO_PROVIDERS
    if unknown_providers:
        raise ValueError(
            f"unknown providers in llm.auto.provider_order: {sorted(unknown_providers)}"
        )
    allow_paid = auto_raw.get("allow_paid_api_fallback", False)
    if not isinstance(allow_paid, bool):
        raise ValueError("llm.auto.allow_paid_api_fallback must be a boolean")
    auto = AutoProviderConfig(provider_order=provider_order, allow_paid_api_fallback=allow_paid)

    retries = _non_negative_int(raw.get("retries", 2), "llm.retries")
    timeout_seconds = _positive_float(raw.get("timeout_seconds", 180.0), "llm.timeout_seconds")
    max_output_tokens = _positive_int(
        raw.get("max_output_tokens_per_call", 3000), "llm.max_output_tokens_per_call"
    )
    retry_backoff = _non_negative_float(
        raw.get("retry_backoff_seconds", 0.0), "llm.retry_backoff_seconds"
    )
    return ProviderConfig(
        mode=mode,
        retries=retries,
        timeout_seconds=timeout_seconds,
        max_output_tokens_per_call=max_output_tokens,
        retry_backoff_seconds=retry_backoff,
        codex_cli=codex,
        claude_cli=claude,
        openai_api=openai,
        auto=auto,
    )


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return dict(value)


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {path} keys: {sorted(unknown)}")


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _non_negative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _positive_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{path} must be positive")
    return float(value)


def _non_negative_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{path} must be non-negative")
    return float(value)
