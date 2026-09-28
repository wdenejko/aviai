"""Summarize audit_long_seq.py reports: the warmed (second) step, memory, loss, gradient checks.

    python audit_summary.py REPORT.json [REPORT.json ...]
"""
import json
import sys

GIB = 2**30
rows = []
for path in sys.argv[1:]:
    d = json.load(open(path))
    seq = d["arguments"]["sequence_length"]
    phases = {p["phase"]: p for p in d["performance_summary"]["phases_by_total_time"]}
    secs = {k: round(v["total_seconds"], 3) for k, v in phases.items()}
    peak = max(p["peak_reserved_bytes"] for p in phases.values()) / GIB
    first, second = d.get("first_backward", {}), d.get("second_backward", {})
    update = d.get("first_update", {})
    print(f"== seq {seq}: status {d.get('status')}")
    print("  phases (s):", secs)
    print(f"  peak reserved {peak:.2f} GiB; "
          f"loss first {first.get('loss')} second {second.get('loss')}")
    print(f"  LoRA-B updated {update.get('changed_tensors')}/{update.get('tensors')}")
    extra = d.get("extra_steps")
    if extra:
        print("  extra_steps:", json.dumps(extra)[:400])
    rows.append((seq, secs, peak))
print("\nphase names:", sorted({k for _, s, _ in rows for k in s}))
