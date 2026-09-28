"""GPU check of the seq-4096 MMQ keys against torch-ggml-ops' external oracle (ADR-001 retrain).

For each new key (ordinary M=4096, top-8 grouped R=32768) and, as a control, its seq-2048 twin
(M=2048, R=16384: the same kernel spec, already trusted in training), this runs the per-route checks
of tests/test_public_deployment_correctness.py without pytest (not in the training venv):
  - the public API route and the direct GGTensile route against the external reference;
  - repeatability: two public runs are identical;
  - dependence: negating the first input row (forward) or cotangent row (backward) changes the
    output.

    cd ~/src/torch-ggml-ops && python validate_seq4096_keys.py OUT.json [Ordinary|Grouped]

The optional family limits the run (the grouped keys were re-run alone once the oracle's own
AITER tables knew R = 32768).
"""
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, ".")  # the torch-ggml-ops checkout, for its tools/ package
from tools.mmq_correctness import (  # noqa: E402
    assert_changed,
    assert_external_reference,
    assert_repeat,
    external_reference,
    prepare_case,
)
from tools.mmq_deployment_cases import hip_control_root, public_deployment_cases  # noqa: E402
from tools.mmq_deployment_runner import run_implementation  # noqa: E402

NEW = {"Ordinary": 4096, "Grouped": 32768}
ONLY = sys.argv[2] if len(sys.argv) > 2 else None
TWIN = {"Ordinary": 2048, "Grouped": 16384}


def family(case):
    return next((f for f in NEW if case.operation.startswith(f)), None)


def shape(case):
    return (case.operation, case.quant_type, case.out_features, case.in_features)


def pairwise(check, actual, other, label):
    if isinstance(actual, tuple):
        for index, (a, b) in enumerate(zip(actual, other, strict=True)):
            check(a, b, f"{label}[{index}]")
    else:
        check(actual, other, label)


def run(case, prepared, implementation):
    return run_implementation(case, prepared, implementation, hip_root=hip_control_root())


def check_case(case):
    prepared = prepare_case(case, device=torch.device("cuda"))
    expected = external_reference(prepared)
    actual = run(case, prepared, "public")
    pairwise(assert_external_reference, actual, expected, f"public/{case.identity}")
    pairwise(assert_external_reference, run(case, prepared, "ggtensile"), expected,
             f"ggtensile/{case.identity}")
    pairwise(assert_repeat, actual, run(case, prepared, "public"), f"repeat/{case.identity}")
    if case.operation.endswith("Forward") or case.operation == "GroupedForwardPair":
        saved = prepared.input
        prepared.input = saved.clone()
        prepared.input[0].neg_()
        pairwise(assert_changed, actual, run(case, prepared, "public"), f"dep/{case.identity}")
        prepared.input = saved
    else:
        saved = prepared.grad_outputs
        mutated = list(saved)
        mutated[0] = mutated[0].clone()
        mutated[0][0].neg_()
        prepared.grad_outputs = tuple(mutated)
        assert_changed(actual, run(case, prepared, "public"), f"dep/{case.identity}")
        prepared.grad_outputs = saved


cases = public_deployment_cases()
new = [c for c in cases if family(c) and c.rows == NEW[family(c)] and ONLY in (None, family(c))]
new_shapes = {shape(c) for c in new}
twins = [c for c in cases if family(c) and c.rows == TWIN[family(c)] and shape(c) in new_shapes]
print(f"new keys: {len(new)}, seq-2048 twins: {len(twins)}", flush=True)
results = []
for role, case in [("twin", c) for c in twins] + [("new", c) for c in new]:
    t0 = time.time()
    try:
        check_case(case)
        status, error = "pass", ""
    except Exception as exc:  # noqa: BLE001 - record every failure, keep checking the rest
        # a missing tensor source (the DeepSeek-V4 GGUF behind the DeepSeek-shaped keys is not on
        # this box) is a key that could not be checked, not one that failed
        prerequisite = type(exc).__name__ == "CorrectnessPrerequisite"
        status = "skip" if prerequisite else "FAIL"
        error = f"{type(exc).__name__}: {exc}"[:400]
        if not prerequisite:
            traceback.print_exc()
    results.append({"role": role, "operation": case.operation, "quant": case.quant_type,
                    "rows": case.rows, "n": case.out_features, "k": case.in_features,
                    "symbol": case.symbol, "status": status, "error": error,
                    "seconds": round(time.time() - t0, 2)})
    print(f"{status} {role} {case.operation} {case.quant_type} rows={case.rows} "
          f"n={case.out_features} k={case.in_features} ({results[-1]['seconds']}s) {error}",
          flush=True)
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
summary = {role: {s: sum(r["role"] == role and r["status"] == s for r in results)
                  for s in ("pass", "FAIL", "skip")} for role in ("new", "twin")}
print("summary:", summary, flush=True)
Path(sys.argv[1]).write_text(json.dumps({"summary": summary, "results": results}, indent=1) + "\n")
# The exit status is informational only: on this box a process that initialized HIP exits 0
# whatever sys.exit says, so callers read the summary in OUT.json instead.
sys.exit(0 if not any(r["status"] == "FAIL" for r in results) else 1)
