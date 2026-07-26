"""CLI wrapper around `common.sizing.solve_for_budget`.

`scripts/` is not an importable package in this repo (pyproject.toml sets
`pythonpath = ["src"]` only, no `scripts/__init__.py`) so the actual solver
lives in `src/common/sizing.py`; this module just exposes it as a CLI, the
same pattern used by `scripts/make_ember_subset.py` / `src/common/ember_subset.py`.
"""
from __future__ import annotations

import argparse

from common.sizing import solve_for_budget


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Solve kernel scaling for a budget")
    p.add_argument("--budget-s", type=float, required=True,
                   help="target seconds for ONE Gram build")
    p.add_argument("--ref-n", type=int, default=400)
    p.add_argument("--ref-s", type=float, default=11.4)
    p.add_argument("--exponent", type=float, default=2.0)
    args = p.parse_args(argv)
    n = solve_for_budget(args.budget_s, args.ref_n, args.ref_s, args.exponent)
    print(f"max n for {args.budget_s}s/Gram: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
