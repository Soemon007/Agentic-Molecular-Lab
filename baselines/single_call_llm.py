"""Single-call LLM baseline: one Haiku call for 50 SMILES -> Gatekeeper -> oracle -> AUC.
The control that shows whether the multi-agent architecture earned its complexity."""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chem_core import Gatekeeper  # noqa: E402
from eval.pmo_auc import report  # noqa: E402
from llm import HAIKU  # noqa: E402
from oracle import DRD2Oracle  # noqa: E402

PROMPT = "Generate 50 diverse high-affinity SMILES targeting DRD2. Return one SMILES per line, nothing else."


def parse_smiles(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(\d+[.)]\s*|[-*]\s*)", "", line).strip().strip("`")
        tok = line.split()[0] if line.split() else ""
        if tok:
            out.append(tok)
    return out


def run(client=None, out="data/trajectories/single_call_llm.csv", model=HAIKU):
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    resp = client.messages.create(model=model, max_tokens=4096, temperature=0.7,
                                  messages=[{"role": "user", "content": PROMPT}])
    text = "".join(b.text for b in resp.content if b.type == "text")
    tokens = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    gk, oracle = Gatekeeper(), DRD2Oracle(traj_path=out)
    for s in parse_smiles(text):
        mol, _ = gk.check(s, "single_call_llm")  # general checks only; no parent
        if mol is not None:
            oracle(s, "single_call_llm")
    oracle.close()
    print(json.dumps({"rejections": gk.rejection_report(), "tokens": tokens, "auc": report(out)}, indent=2))


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    run()
