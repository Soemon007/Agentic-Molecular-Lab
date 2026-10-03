"""Random-sampling baseline: ZINC subset -> DRD2 oracle, up to 10,000 calls."""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle import DRD2Oracle  # noqa: E402


def main(n=10_000, seed=0, out=None):
    from tdc.generation import MolGen
    smiles = MolGen(name="ZINC", path=str(Path(__file__).resolve().parents[1] / "data")).get_data()["smiles"].tolist()
    picks = random.Random(seed).sample(smiles, n)
    oracle = DRD2Oracle(traj_path=out or f"data/trajectories/random_seed{seed}.csv")
    for s in picks:
        oracle(s, "random")
    oracle.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    main(a.n, a.seed, a.out)
