"""The live Anthropic path: contract tests against the installed SDK, and how a failing live run surfaces.

These exist because the first live run failed on call one: `messages.create()` no longer accepts `temperature`
(anthropic >= 1.11), every agent call raised TypeError, and the run ended as a quiet "stalled" with 5 seed molecules.
"""
import asyncio
import inspect
from types import SimpleNamespace

import anthropic
from anthropic.resources.messages import AsyncMessages, Messages
from omnigent.inner.executor import ExecutorConfig, ExecutorError, ToolCallRequest
from omnigent.inner.datamodel import Message

import orchestrator
import server
from baselines import single_call_llm
from llm import AnthropicExecutor
from orchestrator import run_lab, summarize

TOOL = {"name": "submit_proposals", "description": "d", "input_schema": {"type": "object"}}


def _resp(content):
    return SimpleNamespace(content=content, usage=SimpleNamespace(input_tokens=7, output_tokens=3))


def test_anthropic_executor_only_passes_kwargs_the_installed_sdk_accepts():
    seen = {}

    class Messages_:
        async def create(self, **kw):
            seen.update(kw)
            return _resp([SimpleNamespace(type="tool_use", name="submit_proposals", input={"proposals": []})])

    async def go():
        ex = AnthropicExecutor(client=SimpleNamespace(messages=Messages_()))
        cfg = ExecutorConfig(model="claude-haiku-4-5-20251001", temperature=0.7, max_tokens=4096)
        return [ev async for ev in ex.run_turn([Message(role="user", content="{}")], [TOOL], "sys", cfg)]

    events = asyncio.run(go())
    accepted = set(inspect.signature(AsyncMessages.create).parameters)
    assert set(seen) <= accepted, f"kwargs the SDK would reject: {set(seen) - accepted}"
    assert isinstance(events[0], ToolCallRequest) and events[0].metadata["usage"] == {"input_tokens": 7, "output_tokens": 3}


def test_model_that_rejects_forced_tool_choice_falls_back_to_auto_and_remembers_it():
    """claude-sonnet-5-5 answers a forced tool_choice with HTTP 400; the adversary runs on it."""
    sent = []
    err = ('BadRequestError: Error code: 400 - {"type": "error", "error": {"type": "invalid_request_error", '
           '"message": "tool_choice: type \\"tool\\" and \\"any\\" are not supported for this model."}}')

    class Messages_:
        async def create(self, **kw):
            sent.append(kw)
            if "tool_choice" in kw:
                raise RuntimeError(err)
            return _resp([SimpleNamespace(type="tool_use", name="submit_proposals", input={"proposals": []})])

    ex = AnthropicExecutor(client=SimpleNamespace(messages=Messages_()))

    async def turn():
        cfg = ExecutorConfig(model="claude-sonnet-5-5", max_tokens=4096)
        return [ev async for ev in ex.run_turn([Message(role="user", content="{}")], [TOOL], "sys", cfg)]

    first, second = asyncio.run(turn()), asyncio.run(turn())
    assert isinstance(first[0], ToolCallRequest) and isinstance(second[0], ToolCallRequest)
    assert [("tool_choice" in kw) for kw in sent] == [True, False, False]  # forced once, then remembered
    assert "submit_proposals" in sent[1]["system"]  # the instruction moves into the prompt
    accepted = set(inspect.signature(AsyncMessages.create).parameters)
    assert all(set(kw) <= accepted for kw in sent)


def test_unrelated_api_errors_are_not_retried():
    calls = []

    class Messages_:
        async def create(self, **kw):
            calls.append(kw)
            raise RuntimeError("RateLimitError: 429")

    ex = AnthropicExecutor(client=SimpleNamespace(messages=Messages_()))

    async def turn():
        return [ev async for ev in ex.run_turn([Message(role="user", content="{}")], [TOOL], "sys",
                                               ExecutorConfig(model="claude-haiku-4-5-20251001"))]

    events = asyncio.run(turn())
    assert len(calls) == 1 and isinstance(events[0], ExecutorError) and "429" in events[0].message


def test_single_call_baseline_only_passes_kwargs_the_installed_sdk_accepts(tmp_path):
    seen = {}

    class Fake:
        class messages:
            @staticmethod
            def create(**kw):
                seen.update(kw)
                return _resp([SimpleNamespace(type="text", text="CCO\nCCN\n")])

    single_call_llm.run(client=Fake(), out=str(tmp_path / "t.csv"))
    accepted = set(inspect.signature(Messages.create).parameters)
    assert set(seen) <= accepted, f"kwargs the SDK would reject: {set(seen) - accepted}"
    assert anthropic.__version__  # the contract is checked against whatever SDK is installed


class _FailingExecutor:
    async def run_turn(self, messages, tools, system_prompt, config=None):
        yield ExecutorError(message="TypeError: boom")


def test_failing_agent_calls_end_the_run_as_an_error_with_the_real_reason(tmp_path, monkeypatch):
    import policies
    policies.AUTOPILOT = True
    monkeypatch.setattr(orchestrator, "make_executor", lambda mode=None, seed=0: _FailingExecutor())
    sess = asyncio.run(run_lab(budget=60, branches="abc", llm="offline", seed=0, verbose=False,
                               traj_path=tmp_path / "t.csv", run_id="stall_err_test"))
    sess.chatlog.path.unlink()
    assert sess.stalled and sess.stall_is_error and sess.oracle.calls == 5  # only the seed molecules were scored
    assert "boom" in sess.stall_reason
    assert summarize(sess)["stall_reason"] == sess.stall_reason


def _worker_status(tmp_path, monkeypatch, *, stall_is_error):
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path / "runs")

    async def fake_run_lab(**kw):
        return SimpleNamespace(stalled=True, stall_is_error=stall_is_error, stall_reason="agent calls are failing (scout: boom)",
                               oracle=SimpleNamespace(calls=5))

    monkeypatch.setattr(server, "run_lab", fake_run_lab)
    monkeypatch.setattr(server, "summarize", lambda s: {"stalled": True, "stall_reason": s.stall_reason})
    h = server.Handle("t", {"budget": 50, "branches": "abc", "llm": "anthropic", "seed": 0, "adversary": True,
                            "seed_mode": "cold"})
    server._worker(h)
    return h


def test_worker_reports_agent_failures_as_status_error(tmp_path, monkeypatch):
    h = _worker_status(tmp_path, monkeypatch, stall_is_error=True)
    assert h.status == "error" and "boom" in h.error and "5 oracle calls" in h.error


def test_worker_keeps_a_rejection_only_stall_as_finished(tmp_path, monkeypatch):
    h = _worker_status(tmp_path, monkeypatch, stall_is_error=False)
    assert h.status == "finished" and h.error is None and h.summary["stalled"]
