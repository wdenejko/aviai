"""Point the R=32768 paired-backward Q3_K key at kernel spec 1 (seq4096 branch).

Validation (validate_seq4096_keys.py, 2026-09-28) failed this key alone with the rule's choice,
spec 0, the R=16384 winner: NRMSE 0.38 against the external reference (limit 0.04), 42% of the
[32768, 2048] input gradient differing. Its twin at R=16384 passes with spec 0, and the other nine
new grouped keys pass with their R=16384 specs. Spec 1 is the R=65536 winner (MIWaveTile [2, 4]
instead of [1, 4]; everything else equal).
"""
import json
from pathlib import Path

CONFIGS = Path.home() / "src/torch-ggml-ops/tools/ggtensile/configs"
p = CONFIGS / "mmq_grouped_bwd_pair_q3_k_catalog.json"
d = json.loads(p.read_text())
hits = [e for e in d["ExactLogic"] if e["Problem"]["M"] == 32768]
assert len(hits) == 1 and hits[0]["KernelSpecIndex"] == 0, hits
hits[0]["KernelSpecIndex"] = 1
p.write_text(json.dumps(d, indent=2) + "\n")
print("R=32768 paired-backward Q3_K -> spec 1")
