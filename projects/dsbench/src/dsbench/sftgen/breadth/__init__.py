"""Breadth + replay acquisition for the Gate-1 pilot mixture (ADR-001 §B.4).

The targeted slices (Targets A/C, `../`) are OURS and execution-verified. The remaining ~86% of the
pilot is PUBLIC breadth (DS notebooks, text-to-SQL, SWE, general code) + replay. This package
pulls those from HuggingFace under two hard ADR-001 rules -- a teacher-licence allow-list and
decontamination vs dsbench -- and normalises every source into the one training record shape
(`messages` [+ `tools`] + assistant-only loss mask + provenance `meta`).
"""
