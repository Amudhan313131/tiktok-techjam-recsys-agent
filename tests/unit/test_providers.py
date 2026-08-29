from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rex.agents.provider import (
    ClaudeCLIProvider,
    CodexCLIProvider,
    CommandResult,
    FixedQueueProvider,
    OpenAIResponsesProvider,
    ProviderBudgetExceeded,
    ProviderConfigurationError,
    ProviderExecutionError,
    ProviderResponse,
    ProviderRouter,
    ProviderSchemaError,
    ProviderTimeoutError,
    redact_secrets,
)
from rex.agents.provider_config import load_provider_config


SCHEMA = {
    "type": "object",
    "properties": {"decision": {"type": "string"}},
    "required": ["decision"],
    "additionalProperties": False,
}


def generate(provider: Any) -> ProviderResponse:
    return provider.generate(role="proposal", system="system", prompt="prompt", schema=SCHEMA)


def openai_response(
    value: dict[str, Any],
    *,
    input_tokens: int = 5,
    output_tokens: int = 3,
) -> SimpleNamespace:
    return SimpleNamespace(
        id="resp_123",
        model="gpt-test",
        status="completed",
        output_text=json.dumps(value),
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


class ResponsesStub:
    def __init__(self, results: list[Any]):
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class ClientFactoryStub:
    def __init__(self, results: list[Any]):
        self.responses = ResponsesStub(results)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(responses=self.responses)


class HTTPError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


class APIConnectionError(Exception):
    pass


class AlwaysFailProvider:
    def __init__(self, error: Exception):
        self.error = error
        self.calls = 0

    def generate(self, **_: Any) -> ProviderResponse:
        self.calls += 1
        raise self.error


class CountingProvider:
    def __init__(self, name: str):
        self.name = name
        self.calls = 0

    def generate(self, **_: Any) -> ProviderResponse:
        self.calls += 1
        return ProviderResponse({"decision": "yes"}, self.name, "test")


def test_provider_config_defaults_to_codex_without_paid_fallback() -> None:
    config = load_provider_config(None)
    assert config.mode == "codex_cli"
    assert config.retries == 2
    assert not config.auto.allow_paid_api_fallback
    assert config.auto.provider_order == (
        "codex_cli",
        "claude_cli",
        "openai_api",
        "fixed",
    )
    assert config.claude_cli.executable == "claude"
    assert config.openai_api.api_key_env == "OPENAI_API_KEY"


def test_provider_config_rejects_inline_secret_and_unknown_keys() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        load_provider_config({"openai_api": {"api_key": "sk-do-not-store"}})
    with pytest.raises(ValueError, match="unknown llm keys"):
        load_provider_config({"secret": "sk-do-not-store"})


def test_provider_config_validates_auto_order_and_budgets() -> None:
    config = load_provider_config(
        {
            "mode": "auto",
            "openai_api": {"max_calls_per_run": 4, "max_total_tokens": 99},
            "auto": {
                "provider_order": ["codex_cli", "claude_cli", "openai_api", "fixed"],
                "allow_paid_api_fallback": True,
            },
        }
    )
    assert config.openai_api.max_calls_per_run == 4
    assert config.auto.allow_paid_api_fallback
    with pytest.raises(ValueError, match="unique"):
        load_provider_config({"auto": {"provider_order": ["fixed", "fixed"]}})


def test_codex_cli_uses_read_only_ephemeral_structured_mode(tmp_path: Path) -> None:
    observed: dict[str, Any] = {}

    def runner(args: list[str], input_text: str, timeout: float) -> CommandResult:
        observed.update(args=args, input_text=input_text, timeout=timeout)
        schema_path = Path(args[args.index("--output-schema") + 1])
        response_path = Path(args[args.index("--output-last-message") + 1])
        assert json.loads(schema_path.read_text()) == SCHEMA
        response_path.write_text('{"decision":"ship"}', encoding="utf-8")
        stdout = "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "thread_123"}),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {"input_tokens": 11, "output_tokens": 7},
                    }
                ),
            ]
        )
        return CommandResult(0, stdout, "", 0.25)

    response = generate(
        CodexCLIProvider(
            model="gpt-test",
            working_directory=tmp_path,
            timeout_seconds=17,
            command_runner=runner,
        )
    )

    args = observed["args"]
    assert args[:4] == ["codex", "--ask-for-approval", "never", "exec"]
    assert "--ephemeral" in args
    assert args[args.index("--sandbox") + 1] == "read-only"
    assert args[args.index("--ask-for-approval") + 1] == "never"
    assert args[args.index("--cd") + 1] == str(tmp_path.resolve())
    assert args[-1] == "-"
    assert "system" in observed["input_text"]
    assert observed["timeout"] == 17
    assert response.value == {"decision": "ship"}
    assert response.request_id == "thread_123"
    assert (response.input_tokens, response.output_tokens) == (11, 7)


def test_codex_cli_rejects_invalid_json_and_schema(tmp_path: Path) -> None:
    def runner_with(raw: str):
        def runner(args: list[str], _: str, __: float) -> CommandResult:
            Path(args[args.index("--output-last-message") + 1]).write_text(raw)
            return CommandResult(0, "", "", 0.01)

        return runner

    with pytest.raises(ProviderSchemaError):
        generate(CodexCLIProvider(working_directory=tmp_path, command_runner=runner_with("nope")))
    with pytest.raises(ProviderSchemaError):
        generate(
            CodexCLIProvider(
                working_directory=tmp_path,
                command_runner=runner_with('{"wrong":"value"}'),
            )
        )


def test_codex_cli_defaults_to_an_empty_ephemeral_context() -> None:
    observed_directory: Path | None = None

    def runner(args: list[str], _: str, __: float) -> CommandResult:
        nonlocal observed_directory
        observed_directory = Path(args[args.index("--cd") + 1])
        assert "--skip-git-repo-check" in args
        assert observed_directory != Path.cwd().resolve()
        assert list(observed_directory.iterdir()) == [
            observed_directory / "schema.json"
        ]
        Path(args[args.index("--output-last-message") + 1]).write_text(
            '{"decision":"safe"}', encoding="utf-8"
        )
        return CommandResult(0, "", "", 0.01)

    response = generate(CodexCLIProvider(command_runner=runner))
    assert response.value == {"decision": "safe"}
    assert observed_directory is not None


def test_codex_cli_normalizes_timeout() -> None:
    def runner(_: list[str], __: str, timeout: float) -> CommandResult:
        raise subprocess.TimeoutExpired("codex", timeout)

    with pytest.raises(ProviderTimeoutError) as raised:
        generate(CodexCLIProvider(command_runner=runner))
    assert raised.value.retryable


def test_claude_cli_uses_tool_free_ephemeral_structured_mode(tmp_path: Path) -> None:
    observed: dict[str, Any] = {}

    def runner(
        args: list[str], input_text: str, timeout: float, working_directory: Path
    ) -> CommandResult:
        observed.update(
            args=args,
            input_text=input_text,
            timeout=timeout,
            working_directory=working_directory,
        )
        return CommandResult(
            0,
            json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "structured_output": {"decision": "ship"},
                    "session_id": "session_123",
                    "usage": {"input_tokens": 13, "output_tokens": 5},
                }
            ),
            "",
            0.3,
        )

    response = generate(
        ClaudeCLIProvider(
            model="sonnet",
            working_directory=tmp_path,
            timeout_seconds=19,
            command_runner=runner,
        )
    )

    args = observed["args"]
    assert args[:4] == ["claude", "--print", "--output-format", "json"]
    assert json.loads(args[args.index("--json-schema") + 1]) == SCHEMA
    assert args[args.index("--permission-mode") + 1] == "dontAsk"
    assert args[args.index("--tools") + 1] == ""
    assert args[args.index("--mcp-config") + 1] == '{"mcpServers":{}}'
    assert args[args.index("--setting-sources") + 1] == ""
    assert "--no-session-persistence" in args
    assert "--disable-slash-commands" in args
    assert observed["working_directory"] == tmp_path.resolve()
    assert observed["timeout"] == 19
    assert "system" in observed["input_text"]
    assert response.value == {"decision": "ship"}
    assert response.request_id == "session_123"
    assert (response.input_tokens, response.output_tokens) == (13, 5)


def test_claude_cli_defaults_to_empty_context_and_rejects_bad_schema() -> None:
    observed_directory: Path | None = None

    def safe_runner(
        _: list[str], __: str, ___: float, working_directory: Path
    ) -> CommandResult:
        nonlocal observed_directory
        observed_directory = working_directory
        assert working_directory != Path.cwd().resolve()
        assert list(working_directory.iterdir()) == []
        return CommandResult(
            0,
            json.dumps({"structured_output": {"decision": "safe"}}),
            "",
            0.01,
        )

    assert generate(ClaudeCLIProvider(command_runner=safe_runner)).value == {
        "decision": "safe"
    }
    assert observed_directory is not None

    def bad_runner(
        _: list[str], __: str, ___: float, ____: Path
    ) -> CommandResult:
        return CommandResult(
            0,
            json.dumps({"structured_output": {"wrong": "value"}}),
            "",
            0.01,
        )

    with pytest.raises(ProviderSchemaError):
        generate(ClaudeCLIProvider(command_runner=bad_runner))


def test_claude_cli_normalizes_timeout() -> None:
    def runner(_: list[str], __: str, timeout: float, ___: Path) -> CommandResult:
        raise subprocess.TimeoutExpired("claude", timeout)

    with pytest.raises(ProviderTimeoutError) as raised:
        generate(ClaudeCLIProvider(command_runner=runner))
    assert raised.value.retryable


def test_openai_api_uses_env_strict_schema_and_no_storage() -> None:
    factory = ClientFactoryStub([openai_response({"decision": "ship"})])
    provider = OpenAIResponsesProvider(
        environ={"OPENAI_API_KEY": "sk-test-secret", "OPENAI_MODEL": "gpt-test"},
        client_factory=factory,
    )
    response = generate(provider)

    assert response.value == {"decision": "ship"}
    assert response.request_id == "resp_123"
    assert response.input_tokens == 5
    assert factory.calls == [
        {"api_key": "sk-test-secret", "timeout": 180.0, "max_retries": 0}
    ]
    request = factory.responses.calls[0]
    assert request["store"] is False
    assert request["tools"] == []
    assert request["text"]["format"] == {
        "type": "json_schema",
        "name": "rex_proposal",
        "schema": SCHEMA,
        "strict": True,
    }


def test_openai_api_requires_env_credentials_and_model() -> None:
    with pytest.raises(ProviderConfigurationError, match="OPENAI_API_KEY"):
        generate(OpenAIResponsesProvider(environ={}))
    with pytest.raises(ProviderConfigurationError, match="OPENAI_MODEL"):
        generate(OpenAIResponsesProvider(environ={"OPENAI_API_KEY": "secret"}))


def test_openai_error_is_classified_and_secret_is_redacted() -> None:
    secret = "sk-very-secret-key"
    factory = ClientFactoryStub([HTTPError(429, f"Authorization: Bearer {secret}")])
    provider = OpenAIResponsesProvider(
        environ={"OPENAI_API_KEY": secret, "OPENAI_MODEL": "gpt-test"},
        client_factory=factory,
    )
    with pytest.raises(ProviderExecutionError) as raised:
        generate(provider)
    assert raised.value.retryable
    assert raised.value.code == "http_429"
    assert secret not in str(raised.value)
    assert "[REDACTED]" in str(raised.value)


def test_openai_connection_failure_is_retryable() -> None:
    factory = ClientFactoryStub([APIConnectionError("connection reset by peer")])
    provider = OpenAIResponsesProvider(
        environ={"OPENAI_API_KEY": "secret", "OPENAI_MODEL": "gpt-test"},
        client_factory=factory,
    )
    with pytest.raises(ProviderExecutionError) as raised:
        generate(provider)
    assert raised.value.retryable
    assert raised.value.code == "api_error"


def test_openai_call_and_token_budgets_stop_new_requests() -> None:
    factory = ClientFactoryStub(
        [openai_response({"decision": "one"}), openai_response({"decision": "two"})]
    )
    provider = OpenAIResponsesProvider(
        environ={"OPENAI_API_KEY": "secret", "OPENAI_MODEL": "gpt-test"},
        client_factory=factory,
        max_calls=1,
    )
    generate(provider)
    with pytest.raises(ProviderBudgetExceeded, match="call budget"):
        generate(provider)
    assert len(factory.responses.calls) == 1

    token_factory = ClientFactoryStub([openai_response({"decision": "one"}, input_tokens=7)])
    token_provider = OpenAIResponsesProvider(
        environ={"OPENAI_API_KEY": "secret", "OPENAI_MODEL": "gpt-test"},
        client_factory=token_factory,
        max_total_tokens=10,
    )
    generate(token_provider)
    with pytest.raises(ProviderBudgetExceeded, match="token budget"):
        generate(token_provider)


def test_openai_budget_can_resume_and_fails_closed_on_oversized_response() -> None:
    resumed = OpenAIResponsesProvider(
        environ={"OPENAI_API_KEY": "secret", "OPENAI_MODEL": "gpt-test"},
        client_factory=ClientFactoryStub([]),
        max_calls=2,
        initial_calls=2,
        initial_tokens=7,
    )
    with pytest.raises(ProviderBudgetExceeded, match="call budget"):
        generate(resumed)
    assert resumed.tokens_used == 7

    oversized = OpenAIResponsesProvider(
        environ={"OPENAI_API_KEY": "secret", "OPENAI_MODEL": "gpt-test"},
        client_factory=ClientFactoryStub(
            [openai_response({"decision": "one"}, input_tokens=9)]
        ),
        max_total_tokens=5,
    )
    with pytest.raises(ProviderBudgetExceeded, match="exceeded by"):
        generate(oversized)
    assert oversized.calls_used == 1
    assert oversized.tokens_used > oversized.max_total_tokens


def test_auto_mode_prefers_local_claude_without_paid_api_permission() -> None:
    codex = AlwaysFailProvider(ProviderTimeoutError())
    claude = CountingProvider("claude_cli")
    api = CountingProvider("openai_api")
    fixed = FixedQueueProvider([{"decision": "fixed"}])
    router = ProviderRouter(
        {
            "codex_cli": codex,
            "claude_cli": claude,
            "openai_api": api,
            "fixed": fixed,
        },
        mode="auto",
        retries=0,
    )
    response = generate(router)
    assert response.provider == "claude_cli"
    assert response.fallback_chain == ("codex_cli", "claude_cli")
    assert api.calls == 0
    assert router.events[0]["type"] == "provider_degraded"


def test_auto_mode_uses_paid_api_when_explicitly_enabled() -> None:
    codex = AlwaysFailProvider(ProviderExecutionError("offline", code="offline"))
    claude = AlwaysFailProvider(ProviderExecutionError("offline", code="offline"))
    api = CountingProvider("openai_api")
    fixed = CountingProvider("fixed")
    router = ProviderRouter(
        {
            "codex_cli": codex,
            "claude_cli": claude,
            "openai_api": api,
            "fixed": fixed,
        },
        mode="auto",
        retries=0,
        allow_paid_api_fallback=True,
    )
    response = generate(router)
    assert response.provider == "openai_api"
    assert response.fallback_chain == ("codex_cli", "claude_cli", "openai_api")
    assert api.calls == 1
    assert fixed.calls == 0


def test_auto_router_retries_twice_then_degrades_without_new_hypothesis() -> None:
    codex = AlwaysFailProvider(ProviderTimeoutError())
    fixed = FixedQueueProvider([{"decision": "fixed"}])
    router = ProviderRouter(
        {"codex_cli": codex, "fixed": fixed},
        mode="auto",
        retries=2,
        provider_order=("codex_cli", "fixed"),
    )
    response = generate(router)
    assert codex.calls == 3
    assert response.attempts == 4
    assert response.fallback_errors == ("codex_cli:timeout",)


def test_explicit_live_mode_never_silently_falls_back_to_fixed() -> None:
    codex = AlwaysFailProvider(ProviderTimeoutError())
    fixed = FixedQueueProvider([{"decision": "fixed"}])
    router = ProviderRouter(
        {"codex_cli": codex, "fixed": fixed},
        mode="codex_cli",
        retries=0,
    )
    with pytest.raises(ProviderExecutionError, match="codex_cli:timeout"):
        generate(router)
    assert not fixed.calls


def test_fixed_queue_is_role_scoped_and_schema_checked() -> None:
    fixed = FixedQueueProvider(
        {
            "proposal": [{"decision": "proposal"}],
            "patch": [{"decision": "patch"}],
        }
    )
    assert generate(fixed).value["decision"] == "proposal"
    patch = fixed.generate(role="patch", system="", prompt="", schema=SCHEMA)
    assert patch.value["decision"] == "patch"
    with pytest.raises(ProviderConfigurationError, match="exhausted"):
        generate(fixed)


def test_router_reports_all_failures_without_leaking_details() -> None:
    router = ProviderRouter(
        {"codex_cli": AlwaysFailProvider(ProviderExecutionError("private", code="offline"))},
        mode="codex_cli",
        retries=0,
    )
    with pytest.raises(ProviderExecutionError) as raised:
        generate(router)
    assert raised.value.code == "all_providers_failed"
    assert "codex_cli:offline" in str(raised.value)
    assert "fixed:not_configured" not in str(raised.value)


def test_redact_secrets_covers_explicit_and_header_values() -> None:
    secret = "not-prefixed-secret"
    redacted = redact_secrets(
        f"key={secret}; Authorization: Bearer sk-abcdefghijk another sk-12345678", secret
    )
    assert secret not in redacted
    assert "sk-" not in redacted
