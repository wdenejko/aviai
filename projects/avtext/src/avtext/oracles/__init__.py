"""oracles — parser adapters (Skill B, Phase 2).

Three *independent, different-lineage* parsers behind one interface, each mapping
raw text into the canonical schema:

    python-metar (`metar`)         — long-standing, regex-based
    mivek metar-taf-parser         — separate lineage, strong TAF support
    avwx-engine (`avwx`)           — different assumptions again

A 4th "voice" — the AWC official decoded JSON — joins at the consensus layer.
Lineage diversity matters: three parsers that share an upstream assumption will
agree *and* be wrong together, inflating consensus agreement (α). Adapter bugs
are caught by the quality/ and consensus/ layers, not trusted here.
"""
