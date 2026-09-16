"""Subprocess entry point that runs ONE candidate solution against ONE problem's fixtures.

Runs in its own process so a crash, an infinite loop, or a memory blow-up in model-generated
code is contained (the parent enforces a wall-clock timeout and just kills us). Only the pickled
result crosses back to the parent, which is where the trusted checker runs.

Trust model: this executes model-generated code on your machine. That is acceptable here because
it is YOUR model and YOUR box, and the isolation is for robustness, not adversaries. Do not point
this harness at an untrusted endpoint without a real sandbox (container / seccomp).

Usage: python -m dsbench.harness._worker <problem_module> <code_file> <result_file>
"""

from __future__ import annotations

import importlib
import pickle
import sys
import traceback


def main() -> int:
    module_path, code_file, result_file = sys.argv[1], sys.argv[2], sys.argv[3]
    try:
        prob = importlib.import_module(module_path).PROBLEM
        inputs = prob.make_inputs()
        with open(code_file) as fh:
            code = fh.read()

        if prob.mode == "sql":
            import duckdb

            con = duckdb.connect()
            for name, df in inputs["tables"].items():
                con.register(name, df)
            result = con.execute(code).fetchdf()
        else:
            import numpy as np
            import pandas as pd

            ns: dict = {"pd": pd, "np": np}  # lenient: notebooks always have these
            exec(code, ns)  # noqa: S102 - intentional: this is the benchmark's whole point
            fn = ns.get(prob.entrypoint)
            if not callable(fn):
                raise NameError(f"expected a function named {prob.entrypoint!r} to be defined")
            result = fn(**inputs)

        with open(result_file, "wb") as fh:
            pickle.dump({"ok": True, "result": result}, fh)
        return 0
    except Exception:
        with open(result_file, "wb") as fh:
            pickle.dump({"ok": False, "error": traceback.format_exc()}, fh)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
