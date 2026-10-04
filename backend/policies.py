"""Orchestration-layer policies, as Omnigent `FunctionPolicy` objects (ALLOW / ASK / DENY).

These run in code on every tool call, regardless of what any prompt says:
  1. write_permission     — only the oracle wrapper writes scores; agents get only their own tool
  2. budget_cap           — hard stop at the oracle-call ceiling
  3. electrophile_approval — electrophile alert pauses for human approval (AUTOPILOT=true: log + auto-reject)
Policy events are appended to backend/data/policy_events.jsonl.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from omnigent.inner.policies import FunctionPolicy, PolicyAction

import chem_core

AUTOPILOT = os.getenv("AUTOPILOT", "false").lower() == "true"
# autopilot: log policy event + auto-reject (unattended seed runs)
# interactive: input() prompt (recorded demo)

AGENT_TOOLS = {  # the ONLY tool each agent may call
    "scout": "submit_brief", "branch_a": "submit_proposals", "branch_b": "submit_proposals",
    "branch_c": "submit_proposals", "adversary": "submit_diagnosis", "coordinator": "submit_plan",
    "oracle": "oracle_evaluate",
}
SCORE_KEYS = {"score", "fitness", "oracle_score", "drd2", "predicted_score"}


def write_permission(event: dict):
    d = event["data"]
    if event["type"] != "tool_call":
        return None
    agent, tool = d.get("agent"), d.get("tool")
    if AGENT_TOOLS.get(agent) != tool:
        return {"result": "DENY", "reason": f"{agent} may not call {tool} (only {AGENT_TOOLS.get(agent)})"}
    for p in d.get("arguments", {}).get("proposals", []):
        if isinstance(p, dict) and SCORE_KEYS & {k.lower() for k in p}:
            return {"result": "DENY", "reason": f"{agent} tried to write a score; only the oracle wrapper may"}
    return {"result": "ALLOW"}


def make_budget_cap(oracle, ceiling: int):
    def budget_cap(event: dict):
        d = event["data"]
        if event["type"] != "tool_call" or d.get("tool") != "oracle_evaluate":
            return None
        if oracle.calls >= ceiling:
            return {"result": "DENY", "reason": f"oracle budget exhausted ({oracle.calls}/{ceiling})"}
        return {"result": "ALLOW"}
    return budget_cap


def electrophile_approval(event: dict):
    d = event["data"]
    if event["type"] != "tool_call" or d.get("tool") != "oracle_evaluate":
        return None
    flagged = chem_core.electrophile_alerts(d["arguments"].get("alerts", []))
    if flagged:  # BRENK non-electrophile alerts / PAINS are telemetry only and never reach here
        return {"result": "ASK", "reason": f"electrophile alert(s): {', '.join(flagged)}"}
    return {"result": "ALLOW"}


@dataclass
class Verdict:
    allowed: bool
    action: str
    reason: str | None = None


class PolicyGate:
    """Evaluates policies in declaration order: DENY short-circuits, ASK goes to the approver."""

    def __init__(self, policies: list[FunctionPolicy], log_path=None, autopilot=None, approver=None):
        self.policies = policies
        self.approver = approver  # optional `async (reason, content) -> bool` (the UI); replaces input()/autopilot
        self.on_event = None  # optional `(record) -> None`, called for every DENY / ASK record
        self.log_path = Path(log_path) if log_path else Path(__file__).resolve().parent / "data" / "policy_events.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.autopilot = AUTOPILOT if autopilot is None else autopilot
        self.counts = {"ALLOW": 0, "DENY": 0, "ASK": 0, "ASK_APPROVED": 0, "ASK_REJECTED": 0}

    def _log(self, **kw):
        rec = {"t": time.strftime("%FT%T"), **kw}
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        if self.on_event:
            self.on_event(rec)

    async def check(self, phase: str, content: dict) -> Verdict:
        ask_reason = None
        for p in self.policies:
            r = await p.evaluate(content, phase, {})
            if r.action == PolicyAction.DENY:
                self.counts["DENY"] += 1
                self._log(event="DENY", policy=p.name, tool=content.get("tool"), agent=content.get("agent"),
                          reason=r.reason)
                return Verdict(False, "DENY", r.reason)
            if r.action == PolicyAction.ASK:
                ask_reason = r.reason
        if ask_reason:
            self.counts["ASK"] += 1
            if self.approver:
                approved = bool(await self.approver(ask_reason, content))
            else:
                approved = self._approve(ask_reason, content)
            self.counts["ASK_APPROVED" if approved else "ASK_REJECTED"] += 1
            self._log(event="ASK", approved=approved, autopilot=self.autopilot and not self.approver,
                      reason=ask_reason, smiles=content.get("arguments", {}).get("smiles"),
                      branch=content.get("branch"))
            return Verdict(approved, "ASK", ask_reason)
        self.counts["ALLOW"] += 1
        return Verdict(True, "ALLOW")

    def _approve(self, reason: str, content: dict) -> bool:
        if self.autopilot:
            return False  # unattended: log + auto-reject
        smi = content.get("arguments", {}).get("smiles")
        return input(f"\n[APPROVAL] {reason}\n  molecule: {smi}\n  send to oracle? [y/N] ").strip().lower() == "y"


def build_gate(oracle, ceiling: int, log_path=None, autopilot=None, approver=None) -> PolicyGate:
    return PolicyGate([
        FunctionPolicy(name="write_permission", on=["tool_call"], callable=write_permission),
        FunctionPolicy(name="budget_cap", on=["tool_call"], callable=make_budget_cap(oracle, ceiling)),
        FunctionPolicy(name="electrophile_approval", on=["tool_call"], callable=electrophile_approval),
    ], log_path=log_path, autopilot=autopilot, approver=approver)
