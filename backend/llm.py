"""LLM backends, as Omnigent `Executor` subclasses.

* AnthropicExecutor — real Haiku/Sonnet via the Anthropic Messages API (forced tool use).
* OfflineExecutor   — deterministic RDKit-mutation stand-in used when no ANTHROPIC_API_KEY is
  available. It is a PLUMBING-TEST mock, not evidence about LLM behaviour.

Both emit a single ToolCallRequest per turn whose metadata carries token usage, so the
ledger can report token totals next to oracle calls (README disclosure #2).
"""
from __future__ import annotations

import json
import os
import zlib
from collections import defaultdict
from collections.abc import AsyncIterator

from omnigent.inner.executor import (Executor, ExecutorConfig, ExecutorError, ToolCallRequest,
                                     TurnComplete)

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-5-5"


class TokenLedger:
    def __init__(self):
        self.by_agent = defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "calls": 0})

    def add(self, agent: str, usage: dict | None):
        d = self.by_agent[agent]
        d["calls"] += 1
        for k in ("input_tokens", "output_tokens"):
            d[k] += int((usage or {}).get(k, 0))

    def totals(self) -> dict:
        t = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        for d in self.by_agent.values():
            for k in t:
                t[k] += d[k]
        return t

    def as_dict(self) -> dict:
        return {"total": self.totals(), "by_agent": {k: dict(v) for k, v in self.by_agent.items()}}


def llm_mode() -> str:
    m = os.getenv("LLM_MODE", "").lower()
    if m in ("offline", "anthropic"):
        return m
    return "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "offline"


class AnthropicExecutor(Executor):
    def __init__(self, client=None):
        if client is None:
            import anthropic
            client = anthropic.AsyncAnthropic()
        self.client = client
        self.no_forced_tool: set[str] = set()  # models that reject tool_choice type "tool" (claude-sonnet-5-5 does)

    async def _create(self, config: ExecutorConfig, system_prompt: str, user: str, spec: dict, force: bool):
        # No `temperature`: anthropic >= 1.11 removed sampling parameters from messages.create(), so passing it
        # raised TypeError on every call. `config.temperature` is therefore a no-op here (the adversary's 0.0 too).
        kw = dict(model=config.model, max_tokens=min(config.max_tokens, 4096), system=system_prompt,
                  messages=[{"role": "user", "content": user}],
                  tools=[{"name": spec["name"], "description": spec.get("description", ""),
                          "input_schema": spec["input_schema"]}])
        if force:
            kw["tool_choice"] = {"type": "tool", "name": spec["name"]}
        else:  # model cannot be forced: default tool_choice, so say it in the prompt instead
            kw["system"] = f"{system_prompt}\n\nRespond only by calling the `{spec['name']}` tool."
        return await self.client.messages.create(**kw)

    async def run_turn(self, messages, tools, system_prompt, config: ExecutorConfig | None = None) -> AsyncIterator:
        config = config or ExecutorConfig()
        user = "\n\n".join(m.content for m in messages if m.role == "user" and isinstance(m.content, str))
        spec = tools[0]  # our agents expose exactly one tool; force it where the model allows
        try:
            force = config.model not in self.no_forced_tool
            try:
                resp = await self._create(config, system_prompt, user, spec, force)
            except Exception as e:
                if not (force and "tool_choice" in str(e)):
                    raise
                self.no_forced_tool.add(config.model)  # remember, so only the first call to this model pays a retry
                resp = await self._create(config, system_prompt, user, spec, False)
        except Exception as e:  # surfaced to the runner, which treats it as an empty turn
            yield ExecutorError(message=f"{type(e).__name__}: {e}")
            return
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        for block in resp.content:
            if block.type == "tool_use":
                yield ToolCallRequest(name=block.name, args=dict(block.input), metadata={"usage": usage})
                return
        yield TurnComplete(response="", usage=usage)


class OfflineExecutor(Executor):
    """Deterministic mock. The user message is a JSON payload; `config.extra['agent']` selects behaviour."""

    def __init__(self, seed: int = 0, noise: float = 1.0):
        self.seed = seed
        self.exploit = False
        self.noise = noise  # scales the rate of deliberate rule-violating proposals
        self._turn = 0

    async def run_turn(self, messages, tools, system_prompt, config: ExecutorConfig | None = None):
        from agents import offline  # local import: avoids a cycle
        config = config or ExecutorConfig()
        agent = config.extra.get("agent", "")
        payload = json.loads(next(m.content for m in reversed(messages) if m.role == "user"))
        self._turn += 1
        args = offline.respond(agent, payload, seed=zlib.crc32(f"{self.seed}|{agent}|{self._turn}".encode()),
                               noise=self.noise, exploit=self.exploit)
        yield ToolCallRequest(name=tools[0]["name"], args=args, metadata={"usage": {}})


def make_executor(mode: str | None = None, seed: int = 0) -> Executor:
    return AnthropicExecutor() if (mode or llm_mode()) == "anthropic" else OfflineExecutor(seed=seed)
