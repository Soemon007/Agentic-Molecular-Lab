"""Evidence specialist: the one agent whose tool does real work. `lookup_chembl` is an Omnigent FunctionTool with a real
callable (an HTTP lookup), so the policy gate is checking something that can actually touch the outside world. It has no
LLM: its turn is the tool call, made by the orchestrator for the final top hits (see orchestrator.evidence_step)."""
from omnigent.inner.datamodel import AgentDef, ExecutorSpec
from omnigent.inner.tools import FunctionTool

from chembl import lookup_chembl
from llm import HAIKU

TOOL = FunctionTool(
    name="lookup_chembl",
    description="Find the nearest ChEMBL compounds to a SMILES and their measured DRD2 activity, with citations. Read-only.",
    input_schema={"type": "object", "required": ["smiles"], "properties": {"smiles": {"type": "string"}}},
    callable=lookup_chembl)

AGENT = AgentDef(
    name="evidence",
    prompt="Evidence specialist (no LLM turn): looks up the lab's best molecules in ChEMBL and reports whether their "
           "nearest neighbours have measured DRD2 activity. Read-only; one fixed host.",
    executor=ExecutorSpec(model=HAIKU),  # spec holder only, like the coordinator: this agent never calls a model
    tools={TOOL.name: TOOL})
