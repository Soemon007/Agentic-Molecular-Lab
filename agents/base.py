"""Agent registration + the single-shot agent runner (Omnigent AgentDef / Executor / FunctionTool)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from omnigent.inner.datamodel import AgentDef, ExecutorSpec, Message
from omnigent.inner.executor import (ExecutorConfig, ExecutorError, TextChunk, ToolCallRequest, TurnComplete)
from omnigent.inner.tools import FunctionTool


def make_agent(name: str, prompt: str, model: str, tool_name: str, tool_desc: str, schema: dict) -> AgentDef:
    tool = FunctionTool(name=tool_name, description=tool_desc, input_schema=schema,
                        callable=lambda **kw: kw)  # the agent's tool only echoes: it can't touch the oracle
    return AgentDef(name=name, prompt=prompt, executor=ExecutorSpec(model=model), tools={tool_name: tool})


@dataclass
class AgentResult:
    agent: str
    tool: str | None = None
    args: dict = field(default_factory=dict)
    denied: str | None = None
    error: str | None = None


async def run_agent(agent: AgentDef, executor, payload: dict, gate, ledger, temperature: float = 0.7) -> AgentResult:
    """One turn: executor proposes a tool call -> orchestration-layer policies vet it -> result returned.
    The agent never executes anything itself."""
    tools = [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
             for t in agent.tools.values()]
    cfg = ExecutorConfig(model=agent.executor.model, temperature=temperature, max_tokens=4096,
                         extra={"agent": agent.name})
    msgs = [Message(role="user", content=json.dumps(payload))]
    res = AgentResult(agent=agent.name)
    async for ev in executor.run_turn(msgs, tools, agent.prompt or "", cfg):
        if isinstance(ev, ToolCallRequest):
            ledger.add(agent.name, ev.metadata.get("usage"))
            verdict = await gate.check("tool_call", {"tool": ev.name, "arguments": ev.args, "agent": agent.name})
            if not verdict.allowed:
                res.denied = verdict.reason
                return res
            res.tool, res.args = ev.name, ev.args
            return res
        if isinstance(ev, ExecutorError):
            res.error = ev.message
            return res
        if isinstance(ev, TurnComplete):
            ledger.add(agent.name, ev.usage)
            return res
    return res
