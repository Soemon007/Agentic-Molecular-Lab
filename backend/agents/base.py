"""Agent registration + the single-shot agent runner (Omnigent AgentDef / Executor / FunctionTool)."""
from __future__ import annotations

import json
import time
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
    call_id: str | None = None
    tool: str | None = None
    args: dict = field(default_factory=dict)
    denied: str | None = None
    error: str | None = None


async def run_agent(agent: AgentDef, executor, payload: dict, gate, ledger, temperature: float = 0.7,
                    chatlog=None, rnd: int | None = None) -> AgentResult:
    """One turn: executor proposes a tool call -> orchestration-layer policies vet it -> result returned.
    The agent never executes anything itself."""
    tools = [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
             for t in agent.tools.values()]
    cfg = ExecutorConfig(model=agent.executor.model, temperature=temperature, max_tokens=4096,
                         extra={"agent": agent.name})
    msgs = [Message(role="user", content=json.dumps(payload))]
    res = AgentResult(agent=agent.name, call_id=chatlog.new_call_id() if chatlog else None)
    t0, usage, attempted = time.time(), None, None
    async for ev in executor.run_turn(msgs, tools, agent.prompt or "", cfg):
        if isinstance(ev, ToolCallRequest):
            usage = ev.metadata.get("usage")
            ledger.add(agent.name, usage)
            attempted = {"tool": ev.name, "args": ev.args}
            verdict = await gate.check("tool_call", {"tool": ev.name, "arguments": ev.args, "agent": agent.name})
            if not verdict.allowed:
                res.denied = verdict.reason
            else:
                res.tool, res.args = ev.name, ev.args
            break
        if isinstance(ev, ExecutorError):
            res.error = ev.message
            break
        if isinstance(ev, TurnComplete):
            usage = ev.usage
            ledger.add(agent.name, usage)
            break
    if chatlog:
        chatlog.agent_call(call_id=res.call_id, round=rnd, agent=agent.name, model=agent.executor.model,
                           system_prompt=agent.prompt, input=payload, output=attempted or {"tool": None, "args": {}},
                           usage=usage, latency_ms=int((time.time() - t0) * 1000),
                           verdict={"allowed": res.denied is None and res.error is None,
                                    "reason": res.denied or res.error})
    return res
