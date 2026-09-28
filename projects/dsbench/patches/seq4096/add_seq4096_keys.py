"""Add seq-4096 exact keys to torch-ggml-ops' public MMQ catalogs (ADR-001 retrain, 2026-09-28).

WHY: the bundle is exact-key only. Ordinary MMQ is keyed on M = tokens per step (2048, 8192, 32768
= batch 1/4/16 at seq 2048) and grouped (MoE) MMQ on R = routed rows = tokens x top-k. A batch-1
seq-4096 step needs M = 4096 and, for Qwen's top-8 routing, R = 32768; neither exists, so training
fails closed ("unsupported exact deployment key ... M=4096").

WHAT: each new key reuses the kernel spec (the tuned winner) of its seq-2048 twin, the M=2048 or
R=16384 key with the same N and K. That is a starting point, not a guarantee: a spec validated
at one row count is not proven at another. validate_seq4096_keys.py checks every new key against
the library's external oracle, and on 2026-09-28 one failed: paired-backward Q3_K at R=32768
with spec 0 (NRMSE 0.38, limit 0.04). fix_pair_bwd_q3k.py moves it to spec 1 (the R=65536
winner), which passes. Run both scripts, in that order.

NOT extended: the top-6 grouped families (R = 12288 x ..., DeepSeek) and the fixed-grouped Q8_0
family (DeepSeek); no Qwen3.6 path calls them. The LM-head keys (M in 32..512) are chunk sizes,
not sequence lengths.

    python add_seq4096_keys.py [--apply]     # dry run without --apply
"""
import json
import sys
from pathlib import Path

CONFIGS = Path.home() / "src/torch-ggml-ops/tools/ggtensile/configs"
TWIN = {2048: 4096, 16384: 32768}  # seq-2048 token dimension -> its seq-4096 value
apply = "--apply" in sys.argv[1:]
added = 0
for path in sorted(CONFIGS.glob("mmq_*_catalog.json")):
    catalog = json.loads(path.read_text())
    if catalog["KernelFamily"].startswith("FixedGrouped"):
        continue
    logic = catalog["ExactLogic"]
    present = {json.dumps(e["Problem"], sort_keys=True) for e in logic}
    new = []
    for entry in logic:
        problem = entry["Problem"]
        if problem["M"] not in TWIN:
            continue
        twin = dict(problem, M=TWIN[problem["M"]])
        if json.dumps(twin, sort_keys=True) in present:
            continue
        new.append({"Problem": twin, "KernelSpecIndex": entry["KernelSpecIndex"]})
    if not new:
        continue
    added += len(new)
    print(f"{path.name}: +{len(new)} " + ", ".join(
        f"M={e['Problem']['M']} N={e['Problem']['N']} K={e['Problem']['K']} "
        f"spec={e['KernelSpecIndex']}"
        for e in new))
    if apply:
        # appended: load_catalog checks spec indices, duplicate problems and unused specs, not order
        catalog["ExactLogic"] = logic + new
        path.write_text(json.dumps(catalog, indent=2) + "\n")
print(f"{'added' if apply else 'would add'} {added} keys")
