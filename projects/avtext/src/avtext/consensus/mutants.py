"""Mutant-injection audit (Phase 2, Skill B).

Plant a KNOWN error in a clean report, run it through the whole pipeline, and check
the error is caught — by a parse failure, an invariant violation, or a consensus
disagreement. It's judgment-free: we corrupt a good record in a defined way, so we
already know the truth. This measures the pipeline's DETECTION POWER — the missing
half of "the parsers agree", because they can agree on something wrong.

The lesson the audit makes concrete: consensus is BLIND to errors in the raw itself
(every parser reads the same corrupted string and agrees), so it only catches parser
*disagreement*. Physical impossibilities are caught by *invariants*. A plausible-but-
wrong value — a temperature that's 4° off, an altimeter 6 hPa high — slips through
everything short of an independent authority. That gap is precisely why the trust
ladder needs a Tier-2 gold seed.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from avtext.consensus.vote import consensus
from avtext.oracles import ORACLES
from avtext.quality import check_invariants

# Group matchers (IEM raw has no METAR/SPECI keyword). Space-anchored so RVR groups
# like "R28/0600" (R-prefixed) and station ids don't match.
_TEMP = re.compile(r"(?<= )(M?\d{2})/(M?\d{2})(?=\s|$)")
_WIND = re.compile(r"(?<= )(\d{3})(\d{2})KT(?=\s|$)")  # dir+speed, no gust
_Q = re.compile(r"(?<= )Q(\d{4})(?=\s|$)")
_A = re.compile(r"(?<= )A(\d{4})(?=\s|$)")


def _val(s: str) -> int:
    return -int(s[1:]) if s.startswith("M") else int(s)


def _enc(v: int) -> str:
    return f"M{abs(v):02d}" if v < 0 else f"{v:02d}"


# ── injectors: raw -> mutated raw (or None if the group isn't present) ────────
def dewpoint_exceeds_temp(raw: str) -> str | None:
    m = _TEMP.search(raw)
    if not m:
        return None
    return raw[: m.start(2)] + _enc(_val(m.group(1)) + 3) + raw[m.end(2) :]


def gust_below_wind(raw: str) -> str | None:
    m = _WIND.search(raw)
    if not m or int(m.group(2)) < 5:
        return None
    return raw[: m.end(2)] + f"G{int(m.group(2)) - 3:02d}" + raw[m.end(2) :]


def implausible_altimeter(raw: str) -> str | None:
    if (m := _Q.search(raw)) is not None:
        return raw[: m.start(1)] + "1200" + raw[m.end(1) :]  # 1200 hPa > 1100
    if (m := _A.search(raw)) is not None:
        return raw[: m.start(1)] + "3500" + raw[m.end(1) :]  # 35.00 inHg ≈ 1185 hPa
    return None


def perturb_temp_plausible(raw: str) -> str | None:
    m = _TEMP.search(raw)
    if not m:
        return None
    return raw[: m.start(1)] + _enc(_val(m.group(1)) + 4) + raw[m.end(1) :]


def perturb_altimeter_plausible(raw: str) -> str | None:
    if (m := _Q.search(raw)) is not None:
        return raw[: m.start(1)] + f"{int(m.group(1)) + 6:04d}" + raw[m.end(1) :]
    if (m := _A.search(raw)) is not None:
        return raw[: m.start(1)] + f"{int(m.group(1)) + 20:04d}" + raw[m.end(1) :]
    return None


def garble(raw: str) -> str | None:
    toks = raw.split()
    if len(toks) < 4:
        return None
    toks.insert(3, "QX7Z")  # a nonsense group past the header
    return " ".join(toks)


@dataclass
class Mutant:
    name: str
    inject: Callable[[str], str | None]
    expected: str  # "invariant" | "parse" | "undetectable" (the point we're proving)


MUTANTS = [
    Mutant("dewpoint>temp", dewpoint_exceeds_temp, "invariant"),
    Mutant("gust<wind", gust_below_wind, "invariant"),
    Mutant("altimeter out-of-range", implausible_altimeter, "invariant"),
    Mutant("garble", garble, "parse"),
    Mutant("temp +4 (plausible)", perturb_temp_plausible, "undetectable"),
    Mutant("altimeter +6 (plausible)", perturb_altimeter_plausible, "undetectable"),
]


def detect(raw: str) -> str | None:
    """The channel that flags `raw` as suspect, or None if the pipeline accepts it."""
    results = [o.decode(raw) for o in ORACLES]
    obs = [r.obs for r in results if r.ok]
    if len(obs) < len(results):
        return "parse"
    if any(check_invariants(o) for o in obs):
        return "invariant"
    if consensus(obs).disagreements:
        return "consensus"
    return None
