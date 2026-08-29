"""
Thin wrapper around the Anthropic API for the agent's two reasoning jobs:

1. reason_next_move() — Stage 3: free-form reasoning over EDA + run log +
   trick menu + budget to decide what to try next. Returns text; the
   orchestrator parses out the chosen move id.

2. structured_reflect() — Stage 6: forced JSON diagnosis via tool-use,
   validated against reflect_schema.json. This is the anti-lazy-LLM-speak
   mechanism — see agent/schemas/reflect_schema.json.

Model routing: use a cheaper model for high-volume/simple calls and the
stronger model for the two reasoning-heavy steps. Report actual usage
(input+output tokens) via the returned usage dict so state.py can accumulate
it into agent_state.json for the Feasibility numbers.
"""

import json
import os

import anthropic

STRONG_MODEL = "claude-sonnet-4-6"   # Stage 3 / Stage 6 — the reasoning-heavy calls
CHEAP_MODEL = "claude-haiku-4-5-20251001"  # reserved for any high-volume, low-stakes calls

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schemas", "reflect_schema.json")


def _client():
    return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env


def _load_reflect_tool():
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def reason_next_move(system_prompt: str, context: str, model: str = STRONG_MODEL):
    """
    Stage 3. Free-form reasoning call. `context` should already contain:
    EDA findings, the trick menu + diagnosis rules, the run log so far,
    and remaining budget. Returns (text_response, usage_dict).
    """
    client = _client()
    resp = client.messages.create(
        model=model,
        max_tokens=2000,
        system=system_prompt,
        messages=[{"role": "user", "content": context}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    return text, usage


def structured_reflect(system_prompt: str, context: str, model: str = STRONG_MODEL):
    """
    Stage 6. Forces the model to call the recsys_reflection tool, guaranteeing
    schema-conformant output. Returns (parsed_dict, usage_dict).
    """
    client = _client()
    tool = _load_reflect_tool()

    resp = client.messages.create(
        model=model,
        max_tokens=1500,
        system=system_prompt,
        tools=[tool],
        tool_choice={"type": "tool", "name": tool["name"]},
        messages=[{"role": "user", "content": context}],
    )

    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}

    for block in resp.content:
        if block.type == "tool_use" and block.name == tool["name"]:
            return block.input, usage

    raise RuntimeError("Model did not return the expected structured tool call — check schema/model.")
