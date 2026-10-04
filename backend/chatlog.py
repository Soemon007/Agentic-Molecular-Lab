"""Agent chat log: one JSONL record per agent call, plus one per proposal outcome.

backend/data/chats/<run_id>.jsonl
  {"kind":"agent_call", call_id, run_id, round, agent, model, system_prompt, input, output:{tool,args},
   usage, latency_ms, verdict:{allowed,reason}}
  {"kind":"proposal_outcome", call_id, round, agent, parent_id, smiles, gate_reason|null, oracle_score|null,
   ad_similarity, sa_score, alerts, policy}
Joining outcomes to calls on call_id shows which prompts/instructions lead to accepted, high-scoring
(or exploit-flavoured) molecules — the data needed to tune agent prompts.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


class ChatLog:
    def __init__(self, run_id: str, path: Path | None = None):
        self.run_id = run_id
        self.path = Path(path) if path else DATA / "chats" / f"{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ids = itertools.count(1)

    def _write(self, rec: dict):
        with self.path.open("a") as f:
            f.write(json.dumps({"ts": time.strftime("%FT%T"), "run_id": self.run_id, **rec}, default=str) + "\n")

    def new_call_id(self) -> str:
        return f"{self.run_id}-{next(self._ids):05d}"

    def agent_call(self, **kw):
        self._write({"kind": "agent_call", **kw})

    def proposal_outcome(self, **kw):
        self._write({"kind": "proposal_outcome", **kw})

    def event(self, name: str, **kw):  # e.g. adversary triggers, quota changes
        self._write({"kind": "event", "event": name, **kw})
