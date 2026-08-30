"""Structured LLM providers with bounded retries and explicit fallback routing.

Providers return data only. They never apply patches or mutate a worktree. The
router intentionally treats paid OpenAI API fallback as opt-in.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from jsonschema import ValidationError, validate

from rex.agents.provider_config import ProviderConfig


@dataclass(frozen=True)
class ProviderResponse:
    value: dict[str, Any]
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    request_id: str | None = None
    wall_seconds: float = 0.0
    attempts: int = 1
    schema_valid: bool = True
    raw_response: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    fallback_chain: tuple[str, ...] = ()
    fallback_errors: tuple[str, ...] = ()


class StructuredProvider(Protocol):
    def generate(
        self,
        *,
        role: str,
        system: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> ProviderResponse: ...


class ProviderError(RuntimeError):
    """A provider failure whose public message is safe to persist."""

    def __init__(self, message: str, *, code: str, retryable: bool):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class ProviderConfigurationError(ProviderError):
    def __init__(self, message: str):
        super().__init__(message, code="configuration", retryable=False)


class ProviderBudgetExceeded(ProviderError):
    def __init__(self, message: str):
        super().__init__(message, code="budget_exceeded", retryable=False)


class ProviderSchemaError(ProviderError):
    def __init__(self, message: str):
        super().__init__(message, code="invalid_schema_output", retryable=True)


class ProviderTimeoutError(ProviderError):
    def __init__(self, message: str = "provider request timed out"):
        super().__init__(message, code="timeout", retryable=True)


class ProviderExecutionError(ProviderError):
    def __init__(self, message: str, *, code: str = "execution", retryable: bool = False):
        super().__init__(message, code=code, retryable=retryable)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    wall_seconds: float


CommandRunner = Callable[[list[str], str, float], CommandResult]
DirectoryCommandRunner = Callable[[list[str], str, float, Path], CommandResult]
ClientFactory = Callable[..., Any]


class FakeProvider:
    """Replay schema-compatible responses without credentials or network access."""

    def __init__(self, responses: list[dict[str, Any]]):
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        *,
        role: str,
        system: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> ProviderResponse:
        if not self._responses:
            raise RuntimeError("fake provider response queue exhausted")
        self.calls.append({"role": role, "system": system, "prompt": prompt, "schema": schema})
        return ProviderResponse(
            value=self._responses.pop(0),
            provider="fake",
            model="deterministic-replay",
        )


class FixedQueueProvider:
    """Serve pre-approved, deterministic responses without using an LLM."""

    def __init__(
        self,
        responses: Iterable[dict[str, Any]] | Mapping[str, Iterable[dict[str, Any]]],
    ):
        self._global: deque[dict[str, Any]] | None = None
        self._by_role: dict[str, deque[dict[str, Any]]] = {}
        if isinstance(responses, Mapping):
            self._by_role = {role: deque(items) for role, items in responses.items()}
        else:
            self._global = deque(responses)
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        *,
        role: str,
        system: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> ProviderResponse:
        queue = self._by_role.get(role) if self._global is None else self._global
        if queue is None or not queue:
            raise ProviderConfigurationError(f"fixed response queue exhausted for role {role!r}")
        value = queue.popleft()
        _validate_value(value, schema)
        self.calls.append({"role": role, "system": system, "prompt": prompt, "schema": schema})
        return ProviderResponse(
            value=value,
            provider="fixed",
            model="preapproved-queue",
            raw_response=_canonical_json(value),
        )


class AnthropicProvider:
    """Lazy Anthropic adapter; importing the core platform never requires its SDK."""

    def __init__(self, model: str, *, max_tokens: int = 3000):
        self.model = model
        self.max_tokens = max_tokens

    def generate(
        self,
        *,
        role: str,
        system: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> ProviderResponse:
        try:
            import anthropic
        except ImportError as error:  # pragma: no cover - optional live dependency
            raise RuntimeError(
                "install anthropic and set ANTHROPIC_API_KEY for live autonomy"
            ) from error
        tool_name = f"rex_{role}"
        started = time.monotonic()
        response = anthropic.Anthropic().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=[
                {
                    "name": tool_name,
                    "description": f"Return the structured {role} decision.",
                    "input_schema": schema,
                }
            ],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
                value = dict(block.input)
                _validate_value(value, schema)
                return ProviderResponse(
                    value=value,
                    provider="anthropic",
                    model=self.model,
                    input_tokens=int(response.usage.input_tokens),
                    output_tokens=int(response.usage.output_tokens),
                    request_id=getattr(response, "id", None),
                    wall_seconds=time.monotonic() - started,
                    raw_response=_canonical_json(value),
                )
        raise RuntimeError(f"provider did not call required tool {tool_name}")


class CodexCLIProvider:
    """Structured provider backed by a locally authenticated ``codex exec``."""

    def __init__(
        self,
        *,
        executable: str = "codex",
        model: str | None = None,
        working_directory: str | Path | None = None,
        timeout_seconds: float = 180.0,
        max_output_tokens: int = 3000,
        command_runner: CommandRunner | None = None,
    ):
        self.executable = executable
        self.model = model
        self.working_directory = (
            None if working_directory is None else Path(working_directory).resolve()
        )
        self.timeout_seconds = timeout_seconds
        # Kept for config/evidence parity. The CLI has no portable max-token flag.
        self.max_output_tokens = max_output_tokens
        self._command_runner = command_runner or _run_command

    def generate(
        self,
        *,
        role: str,
        system: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> ProviderResponse:
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="rex-codex-") as temporary_directory:
            temp = Path(temporary_directory)
            schema_path = temp / "schema.json"
            response_path = temp / "response.json"
            working_directory = self.working_directory or temp
            cli_schema, uses_json_envelope = _codex_cli_schema(schema)
            schema_path.write_text(_canonical_json(cli_schema), encoding="utf-8")
            args = [
                self.executable,
                "--ask-for-approval",
                "never",
                "exec",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--json",
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(response_path),
                "--cd",
                str(working_directory),
            ]
            if self.working_directory is None:
                args.append("--skip-git-repo-check")
            if self.model:
                args.extend(["--model", self.model])
            args.append("-")
            if uses_json_envelope:
                input_text = (
                    f"{system.strip()}\n\n"
                    "The CLI response envelope has one field named payload_json. "
                    f"Set payload_json to a compact serialized JSON object for the {role} "
                    "decision. The decoded object must satisfy this original JSON Schema:\n"
                    f"{_canonical_json(schema)}\n\n"
                    "Do not use Markdown fences or explanatory text inside payload_json.\n\n"
                    f"{prompt}"
                )
            else:
                input_text = (
                    f"{system.strip()}\n\n"
                    f"Return only the required structured {role} object.\n\n{prompt}"
                )
            try:
                result = self._command_runner(args, input_text, self.timeout_seconds)
            except (subprocess.TimeoutExpired, TimeoutError) as error:
                raise ProviderTimeoutError() from error
            except OSError as error:
                raise ProviderExecutionError(
                    f"could not execute Codex CLI: {type(error).__name__}",
                    code="cli_unavailable",
                    retryable=False,
                ) from error

            safe_stderr = redact_secrets(result.stderr)
            if result.returncode != 0:
                failure = _codex_failure_message(result.stdout, result.stderr)
                message = _bounded_error(redact_secrets(failure))
                raise ProviderExecutionError(
                    message,
                    code="cli_exit",
                    retryable=_codex_failure_is_retryable(failure),
                )
            if not response_path.is_file():
                raise ProviderExecutionError(
                    "Codex CLI did not create its structured response file",
                    code="incomplete_response",
                    retryable=True,
                )
            cli_raw = response_path.read_text(encoding="utf-8")
            if uses_json_envelope:
                envelope = _parse_and_validate(cli_raw, cli_schema)
                raw = str(envelope["payload_json"])
            else:
                raw = cli_raw
            value = _parse_and_validate(raw, schema)
            input_tokens, output_tokens, request_id = _codex_metadata(result.stdout)
            return ProviderResponse(
                value=value,
                provider="codex_cli",
                model=self.model or "codex-configured-default",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                request_id=request_id,
                wall_seconds=result.wall_seconds or (time.monotonic() - started),
                raw_response=raw,
                stdout=redact_secrets(result.stdout),
                stderr=safe_stderr,
            )


class ClaudeCLIProvider:
    """Structured provider backed by a locally authenticated ``claude -p``."""

    def __init__(
        self,
        *,
        executable: str = "claude",
        model: str | None = None,
        working_directory: str | Path | None = None,
        timeout_seconds: float = 180.0,
        max_output_tokens: int = 3000,
        command_runner: DirectoryCommandRunner | None = None,
    ):
        self.executable = executable
        self.model = model
        self.working_directory = (
            None if working_directory is None else Path(working_directory).resolve()
        )
        self.timeout_seconds = timeout_seconds
        # Claude Code does not expose a portable output-token ceiling in print mode.
        self.max_output_tokens = max_output_tokens
        self._command_runner = command_runner or _run_command_in_directory

    def generate(
        self,
        *,
        role: str,
        system: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> ProviderResponse:
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="rex-claude-") as temporary_directory:
            temp = Path(temporary_directory)
            working_directory = self.working_directory or temp
            args = [
                self.executable,
                "--print",
                "--output-format",
                "json",
                "--json-schema",
                _canonical_json(schema),
                "--permission-mode",
                "dontAsk",
                "--tools",
                "",
                "--disable-slash-commands",
                "--no-session-persistence",
                "--no-chrome",
                "--strict-mcp-config",
                "--mcp-config",
                '{"mcpServers":{}}',
                "--setting-sources",
                "",
            ]
            if self.model:
                args.extend(["--model", self.model])
            input_text = (
                f"{system.strip()}\n\n"
                f"Return only the required structured {role} object.\n\n{prompt}"
            )
            try:
                result = self._command_runner(
                    args, input_text, self.timeout_seconds, working_directory
                )
            except (subprocess.TimeoutExpired, TimeoutError) as error:
                raise ProviderTimeoutError() from error
            except OSError as error:
                raise ProviderExecutionError(
                    f"could not execute Claude CLI: {type(error).__name__}",
                    code="cli_unavailable",
                    retryable=False,
                ) from error

            safe_stdout = redact_secrets(result.stdout)
            safe_stderr = redact_secrets(result.stderr)
            if result.returncode != 0:
                message = _bounded_error(safe_stderr or "Claude CLI exited unsuccessfully")
                raise ProviderExecutionError(
                    message,
                    code="cli_exit",
                    retryable=_claude_failure_is_retryable(result.stderr),
                )
            envelope = _claude_envelope(result.stdout)
            if envelope.get("is_error") is True:
                raise ProviderExecutionError(
                    _bounded_error(str(envelope.get("result") or "Claude CLI failed")),
                    code="cli_result_error",
                    retryable=_claude_failure_is_retryable(str(envelope.get("result", ""))),
                )
            value = _claude_structured_value(envelope, schema)
            input_tokens, output_tokens, request_id = _claude_metadata(envelope)
            return ProviderResponse(
                value=value,
                provider="claude_cli",
                model=self.model or str(envelope.get("model") or "claude-configured-default"),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                request_id=request_id,
                wall_seconds=result.wall_seconds or (time.monotonic() - started),
                raw_response=_canonical_json(value),
                stdout=safe_stdout,
                stderr=safe_stderr,
            )


class OpenAIResponsesProvider:
    """OpenAI Responses API adapter using an environment-supplied API key."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        model_env: str = "OPENAI_MODEL",
        timeout_seconds: float = 180.0,
        max_output_tokens: int = 3000,
        max_calls: int = 60,
        max_total_tokens: int = 300_000,
        initial_calls: int = 0,
        initial_tokens: int = 0,
        environ: Mapping[str, str] | None = None,
        client_factory: ClientFactory | None = None,
    ):
        if initial_calls < 0 or initial_tokens < 0:
            raise ProviderConfigurationError(
                "initial OpenAI usage counters must be non-negative"
            )
        environment = os.environ if environ is None else environ
        self.model = model or environment.get(model_env)
        self.api_key_env = api_key_env
        self.model_env = model_env
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.max_calls = max_calls
        self.max_total_tokens = max_total_tokens
        self._api_key = environment.get(api_key_env)
        self._client_factory = client_factory
        self._calls = initial_calls
        self._tokens = initial_tokens
        self._budget_lock = threading.Lock()

    @property
    def calls_used(self) -> int:
        with self._budget_lock:
            return self._calls

    @property
    def tokens_used(self) -> int:
        with self._budget_lock:
            return self._tokens

    def generate(
        self,
        *,
        role: str,
        system: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> ProviderResponse:
        if not self._api_key:
            raise ProviderConfigurationError(f"{self.api_key_env} is not set")
        if not self.model:
            raise ProviderConfigurationError(f"model is not configured; set {self.model_env}")
        started = time.monotonic()
        try:
            client = self._make_client()
            self._reserve_call()
            response = client.responses.create(
                model=self.model,
                instructions=system,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": _schema_name(role),
                        "schema": schema,
                        "strict": True,
                    }
                },
                max_output_tokens=self.max_output_tokens,
                store=False,
                tools=[],
            )
        except ProviderError:
            raise
        except Exception as error:
            raise _classify_openai_error(error, self._api_key) from error

        input_tokens, output_tokens, total_tokens = _openai_usage(response)
        with self._budget_lock:
            # Tokens were consumed even when the returned content is incomplete
            # or fails local schema validation.
            self._tokens += total_tokens
            if self._tokens > self.max_total_tokens:
                raise ProviderBudgetExceeded(
                    "OpenAI API token budget was exceeded by the completed response"
                )
        status = getattr(response, "status", "completed")
        if status != "completed":
            raise ProviderExecutionError(
                f"OpenAI response status was {status!r}",
                code="incomplete_response",
                retryable=True,
            )
        raw = getattr(response, "output_text", None)
        if not isinstance(raw, str) or not raw.strip():
            raise ProviderExecutionError(
                "OpenAI response did not contain output_text",
                code="incomplete_response",
                retryable=True,
            )
        value = _parse_and_validate(raw, schema)
        return ProviderResponse(
            value=value,
            provider="openai_api",
            model=str(getattr(response, "model", None) or self.model),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            request_id=getattr(response, "id", None),
            wall_seconds=time.monotonic() - started,
            raw_response=raw,
        )

    def _reserve_call(self) -> None:
        with self._budget_lock:
            if self._calls >= self.max_calls:
                raise ProviderBudgetExceeded("OpenAI API call budget exhausted")
            if self._tokens >= self.max_total_tokens:
                raise ProviderBudgetExceeded("OpenAI API token budget exhausted")
            self._calls += 1

    def _make_client(self) -> Any:
        factory = self._client_factory
        if factory is None:
            try:
                from openai import OpenAI
            except ImportError as error:  # pragma: no cover - optional live dependency
                raise ProviderConfigurationError(
                    "install the openai package to use openai_api mode"
                ) from error
            factory = OpenAI
        return factory(
            api_key=self._api_key,
            timeout=self.timeout_seconds,
            max_retries=0,
        )


class ProviderRouter:
    """Route structured calls with bounded retries and explicit degradation."""

    def __init__(
        self,
        providers: Mapping[str, StructuredProvider],
        *,
        mode: str = "codex_cli",
        retries: int = 2,
        provider_order: tuple[str, ...] = (
            "codex_cli",
            "claude_cli",
            "openai_api",
            "fixed",
        ),
        allow_paid_api_fallback: bool = False,
        backoff_seconds: float = 0.0,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        if mode not in {"codex_cli", "claude_cli", "openai_api", "auto", "fixed"}:
            raise ProviderConfigurationError(f"unsupported provider mode {mode!r}")
        if retries < 0:
            raise ProviderConfigurationError("provider retries must be non-negative")
        self.providers = dict(providers)
        self.mode = mode
        self.retries = retries
        self.provider_order = provider_order
        self.allow_paid_api_fallback = allow_paid_api_fallback
        self.backoff_seconds = backoff_seconds
        self.sleeper = sleeper
        self.events: list[dict[str, Any]] = []

    @classmethod
    def from_config(
        cls,
        config: ProviderConfig,
        *,
        fixed_provider: StructuredProvider | None = None,
        codex_command_runner: CommandRunner | None = None,
        claude_command_runner: DirectoryCommandRunner | None = None,
        openai_client_factory: ClientFactory | None = None,
        environ: Mapping[str, str] | None = None,
        initial_openai_calls: int = 0,
        initial_openai_tokens: int = 0,
    ) -> ProviderRouter:
        providers: dict[str, StructuredProvider] = {
            "codex_cli": CodexCLIProvider(
                executable=config.codex_cli.executable,
                model=config.codex_cli.model,
                working_directory=config.codex_cli.working_directory,
                timeout_seconds=config.timeout_seconds,
                max_output_tokens=config.max_output_tokens_per_call,
                command_runner=codex_command_runner,
            ),
            "claude_cli": ClaudeCLIProvider(
                executable=config.claude_cli.executable,
                model=config.claude_cli.model,
                working_directory=config.claude_cli.working_directory,
                timeout_seconds=config.timeout_seconds,
                max_output_tokens=config.max_output_tokens_per_call,
                command_runner=claude_command_runner,
            ),
            "openai_api": OpenAIResponsesProvider(
                model=config.openai_api.model,
                api_key_env=config.openai_api.api_key_env,
                model_env=config.openai_api.model_env,
                timeout_seconds=config.timeout_seconds,
                max_output_tokens=config.max_output_tokens_per_call,
                max_calls=config.openai_api.max_calls_per_run,
                max_total_tokens=config.openai_api.max_total_tokens,
                initial_calls=initial_openai_calls,
                initial_tokens=initial_openai_tokens,
                environ=environ,
                client_factory=openai_client_factory,
            ),
        }
        if fixed_provider is not None:
            providers["fixed"] = fixed_provider
        return cls(
            providers,
            mode=config.mode,
            retries=config.retries,
            provider_order=config.auto.provider_order,
            allow_paid_api_fallback=config.auto.allow_paid_api_fallback,
            backoff_seconds=config.retry_backoff_seconds,
        )

    def generate(
        self,
        *,
        role: str,
        system: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> ProviderResponse:
        chain = self._route()
        errors: list[str] = []
        attempted: list[str] = []
        total_attempts = 0
        for provider_name in chain:
            attempted.append(provider_name)
            provider = self.providers.get(provider_name)
            if provider is None:
                errors.append(f"{provider_name}:not_configured")
                continue
            last_error: ProviderError | None = None
            for retry_index in range(self.retries + 1):
                total_attempts += 1
                try:
                    response = provider.generate(
                        role=role,
                        system=system,
                        prompt=prompt,
                        schema=schema,
                    )
                    if errors:
                        self.events.append(
                            {
                                "type": "provider_degraded",
                                "role": role,
                                "from": attempted[:-1],
                                "to": provider_name,
                                "errors": tuple(errors),
                            }
                        )
                    return replace(
                        response,
                        attempts=total_attempts,
                        fallback_chain=tuple(attempted),
                        fallback_errors=tuple(errors),
                    )
                except ProviderError as error:
                    last_error = error
                except Exception as error:
                    last_error = ProviderExecutionError(
                        f"{provider_name} failed with {type(error).__name__}",
                        code="unexpected",
                        retryable=False,
                    )
                if last_error is None or not last_error.retryable:
                    break
                if retry_index < self.retries and self.backoff_seconds > 0:
                    self.sleeper(self.backoff_seconds * (2**retry_index))
            assert last_error is not None
            errors.append(f"{provider_name}:{last_error.code}")
        summary = ", ".join(errors) or "no configured providers"
        raise ProviderExecutionError(
            f"all allowed providers failed ({summary})",
            code="all_providers_failed",
            retryable=False,
        )

    def _route(self) -> tuple[str, ...]:
        if self.mode == "fixed":
            return ("fixed",)
        if self.mode in {"codex_cli", "claude_cli", "openai_api"}:
            return (self.mode,)
        route: list[str] = []
        for provider_name in self.provider_order:
            if provider_name == "openai_api" and not self.allow_paid_api_fallback:
                continue
            if provider_name not in route:
                route.append(provider_name)
        return tuple(route)


def response_json(response: ProviderResponse) -> str:
    return _canonical_json(response.value)


def redact_secrets(text: str, *secrets: str | None) -> str:
    """Redact supplied secrets and common OpenAI credential/header shapes."""

    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    redacted = re.sub(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+", r"\1[REDACTED]", redacted)
    redacted = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", redacted)
    return redacted


def _run_command(args: list[str], input_text: str, timeout_seconds: float) -> CommandResult:
    return _run_command_in_directory(args, input_text, timeout_seconds, Path.cwd())


def _run_command_in_directory(
    args: list[str], input_text: str, timeout_seconds: float, working_directory: Path
) -> CommandResult:
    started = time.monotonic()
    process = subprocess.Popen(  # noqa: S603 - exact executable is operator configuration
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        cwd=working_directory,
    )
    try:
        stdout, stderr = process.communicate(input_text, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        raise
    return CommandResult(
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
        wall_seconds=time.monotonic() - started,
    )


def _parse_and_validate(raw: str, schema: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as error:
        raise ProviderSchemaError("provider returned invalid JSON") from error
    if not isinstance(value, dict):
        raise ProviderSchemaError("provider response must be a JSON object")
    _validate_value(value, schema)
    return value


def _validate_value(value: dict[str, Any], schema: dict[str, Any]) -> None:
    try:
        validate(instance=value, schema=schema)
    except ValidationError as error:
        path = ".".join(str(item) for item in error.absolute_path) or "<root>"
        raise ProviderSchemaError(
            f"provider response failed schema validation at {path}"
        ) from error


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


_CODEX_JSON_ENVELOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"payload_json": {"type": "string"}},
    "required": ["payload_json"],
    "additionalProperties": False,
}


def _codex_cli_schema(schema: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Use native strict output when possible, otherwise carry validated JSON in a string.

    Codex strict output schemas require every object property to be required and
    `additionalProperties` to be false. Pydantic contracts intentionally use
    defaults and typed dictionaries, which cannot be represented by that strict
    subset without changing their meaning. The envelope remains structured at
    the CLI boundary; its decoded JSON is then validated against the unchanged
    application schema before it can reach the coordinator.
    """

    if _codex_schema_is_native_strict(schema):
        return schema, False
    return _CODEX_JSON_ENVELOPE_SCHEMA, True


def _codex_schema_is_native_strict(value: Any) -> bool:
    if isinstance(value, list):
        return all(_codex_schema_is_native_strict(item) for item in value)
    if not isinstance(value, dict):
        return True
    if "default" in value:
        return False
    is_object = value.get("type") == "object" or "properties" in value
    if is_object:
        properties = value.get("properties")
        if not isinstance(properties, dict):
            return False
        if value.get("additionalProperties") is not False:
            return False
        required = value.get("required")
        if not isinstance(required, list) or set(required) != set(properties):
            return False
    return all(_codex_schema_is_native_strict(item) for item in value.values())


def _schema_name(role: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]", "_", role).strip("_") or "decision"
    return f"rex_{normalized}"[:64]


def _codex_metadata(stdout: str) -> tuple[int, int, str | None]:
    input_tokens = 0
    output_tokens = 0
    request_id: str | None = None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_type = str(event.get("type", ""))
        if request_id is None:
            request_id = event.get("thread_id") or event.get("request_id")
            thread = event.get("thread")
            if request_id is None and isinstance(thread, dict):
                request_id = thread.get("id")
        if event_type == "thread.started":
            request_id = event.get("thread_id") or request_id
        usage = event.get("usage")
        if not isinstance(usage, dict):
            response = event.get("response")
            usage = response.get("usage") if isinstance(response, dict) else None
        if isinstance(usage, dict):
            input_tokens = max(input_tokens, int(usage.get("input_tokens", 0) or 0))
            output_tokens = max(output_tokens, int(usage.get("output_tokens", 0) or 0))
    return input_tokens, output_tokens, request_id


def _codex_failure_message(stdout: str, stderr: str) -> str:
    """Prefer the structured terminal Codex error over unrelated MCP warnings."""

    messages: list[str] = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        direct = event.get("message")
        if event.get("type") == "error" and isinstance(direct, str):
            messages.append(direct)
        error = event.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            messages.append(str(error["message"]))
    if messages:
        return messages[-1]
    return stderr or "Codex CLI exited unsuccessfully"


def _claude_envelope(stdout: str) -> dict[str, Any]:
    try:
        envelope = json.loads(stdout)
    except (json.JSONDecodeError, TypeError) as error:
        raise ProviderSchemaError("Claude CLI returned invalid JSON") from error
    if not isinstance(envelope, dict):
        raise ProviderSchemaError("Claude CLI response envelope must be a JSON object")
    return envelope


def _claude_structured_value(
    envelope: dict[str, Any], schema: dict[str, Any]
) -> dict[str, Any]:
    value = envelope.get("structured_output")
    if isinstance(value, dict):
        _validate_value(value, schema)
        return value
    result = envelope.get("result")
    if isinstance(result, str):
        return _parse_and_validate(result, schema)
    raise ProviderSchemaError("Claude CLI did not return structured_output")


def _claude_metadata(envelope: dict[str, Any]) -> tuple[int, int, str | None]:
    usage = envelope.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    request_id = envelope.get("session_id") or envelope.get("request_id")
    return input_tokens, output_tokens, None if request_id is None else str(request_id)


def _openai_usage(response: Any) -> tuple[int, int, int]:
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or 0)
    return input_tokens, output_tokens, total_tokens


def _classify_openai_error(error: Exception, api_key: str | None) -> ProviderError:
    type_name = type(error).__name__.lower()
    message_lower = str(error).lower()
    if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)) or "timeout" in type_name:
        return ProviderTimeoutError()
    status_code = getattr(error, "status_code", None)
    connection_messages = ("connection reset", "connection refused", "temporarily unavailable")
    connection_failure = "connection" in type_name or any(
        term in message_lower for term in connection_messages
    )
    if (
        status_code in {408, 409, 429}
        or (isinstance(status_code, int) and status_code >= 500)
        or connection_failure
    ):
        retryable = True
    elif status_code in {400, 401, 403, 404}:
        retryable = False
    else:
        retryable = False
    message = redact_secrets(str(error), api_key)
    code = f"http_{status_code}" if status_code is not None else "api_error"
    return ProviderExecutionError(_bounded_error(message), code=code, retryable=retryable)


def _codex_failure_is_retryable(stderr: str) -> bool:
    lowered = stderr.lower()
    transient_terms = ("timeout", "timed out", "rate limit", "429", "temporar", "connection")
    return any(term in lowered for term in transient_terms) or bool(
        re.search(r"\b5\d\d\b", lowered)
    )


def _claude_failure_is_retryable(message: str) -> bool:
    lowered = message.lower()
    transient_terms = (
        "timeout",
        "timed out",
        "rate limit",
        "overloaded",
        "429",
        "temporar",
        "connection",
    )
    return any(term in lowered for term in transient_terms) or bool(
        re.search(r"\b5\d\d\b", lowered)
    )


def _bounded_error(message: str, limit: int = 500) -> str:
    compact = " ".join(message.split())
    return compact[:limit] if compact else "provider failed without an error message"
