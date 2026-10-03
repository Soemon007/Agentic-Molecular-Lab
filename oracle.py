"""DRD2 oracle wrapper: global call counter, failure accounting, trajectory logger.

Rules enforced here:
  * the counter increments on EVERY call, including failures and invalid SMILES —
    budget is budget, otherwise the @N columns are not comparable to baselines;
  * exceptions -> return 0.0 and log to oracle_failures.log; failure rate is printed at
    exit (above 2% means the Gatekeeper is leaking);
  * one CSV row per call: call_n, smiles, score, origin_branch, sa, ad_similarity, alerts;
  * this wrapper is the ONLY writer of Candidate.score.
"""
from __future__ import annotations

import atexit
import csv
import os
import time
import warnings
from pathlib import Path

import domain
import chem_core

REPO = Path(__file__).resolve().parent
DEFAULT_TRAJ = REPO / "data" / "trajectories" / "trajectory.csv"
DEFAULT_FAIL_LOG = REPO / "oracle_failures.log"
FAILURE_ALARM = 0.02
FIELDS = ["call_n", "smiles", "score", "origin_branch", "sa", "ad_similarity", "alerts"]


def ensure_tdc_assets() -> None:
    """TDC downloads/loads oracle weights relative to the CWD; pin that to data/ so the
    model location never depends on where the script is launched from."""
    from tdc import Oracle
    from tdc.chem_utils.oracle import oracle as tdc_oracle_mod

    (domain.DATA).mkdir(exist_ok=True)
    cwd = os.getcwd()
    os.chdir(domain.DATA)
    try:
        Oracle(name="DRD2")  # downloads data/oracle/drd2_current.pkl if missing
        if "drd2_model" not in vars(tdc_oracle_mod):  # preload so later calls don't depend on CWD
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tdc_oracle_mod.drd2_model = tdc_oracle_mod.load_drd2_model()
    finally:
        os.chdir(cwd)


class DRD2Oracle:
    def __init__(self, traj_path=DEFAULT_TRAJ, fail_log=DEFAULT_FAIL_LOG, report_at_exit=True):
        ensure_tdc_assets()
        from tdc import Oracle
        self._oracle = Oracle(name="DRD2")  # cwd-independent now: model preloaded above
        self.calls = 0
        self.failures = 0
        self.traj_path = Path(traj_path)
        self.fail_log = Path(fail_log)
        self.traj_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.traj_path.exists() or self.traj_path.stat().st_size == 0
        self._fh = self.traj_path.open("a", newline="")
        self._w = csv.writer(self._fh)
        if new:
            self._w.writerow(FIELDS)
        if report_at_exit:
            atexit.register(self.report)

    # -- core call -----------------------------------------------------------------
    def __call__(self, smiles: str, origin_branch: str = "unknown") -> float:
        self.calls += 1  # count first: failures are budget too
        n = self.calls
        score, sa, ad, alerts = 0.0, "", "", []
        try:
            mol = chem_core.mol_from_smiles(smiles)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = float(self._oracle(smiles))
            if mol is not None:  # annotations are telemetry; never allowed to fail the call
                try:
                    sa = round(chem_core.sa_score(mol), 4)
                    ad = round(domain.ad_similarity(mol), 4)
                    alerts = chem_core.get_alerts(mol)
                except Exception:
                    pass
        except Exception as e:
            score = 0.0
            self.failures += 1
            with self.fail_log.open("a") as f:
                f.write(f"{time.strftime('%F %T')}\tcall={n}\t{smiles!r}\t{type(e).__name__}: {e}\n")
        self._w.writerow([n, smiles, f"{score:.6f}", origin_branch, sa, ad, ";".join(alerts)])
        self._fh.flush()
        return score

    def score_candidate(self, cand: chem_core.Candidate) -> chem_core.Candidate:
        """The only path by which a Candidate gets a score (policy: branches cannot self-score)."""
        cand.score = self(cand.smiles, cand.origin_branch)
        mol = chem_core.mol_from_smiles(cand.smiles)
        if mol is not None:
            cand.core_scaffold = chem_core.scaffold(mol)
            cand.sa_score = chem_core.sa_score(mol)
            cand.ad_similarity = domain.ad_similarity(mol)
            cand.alerts = chem_core.get_alerts(mol)
        return cand

    # -- accounting ----------------------------------------------------------------
    @property
    def failure_rate(self) -> float:
        return self.failures / self.calls if self.calls else 0.0

    def report(self) -> None:
        if self._fh.closed:
            return
        rate = self.failure_rate
        flag = "  << ABOVE 2%: Gatekeeper is leaking" if rate > FAILURE_ALARM else ""
        print(f"[oracle] calls={self.calls} failures={self.failures} rate={rate:.2%}{flag}")

    def close(self) -> None:
        self.report()
        self._fh.close()
