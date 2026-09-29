"""Target A asks for the SQL only (ADR-004 Revision 2).

Revision 1's prompt also asked for "the numeric result", which the model can't know without the
data; the Gate-2 adapter learned to invent one. The verified number stays in the row's verification
record, never in the prompt or the answer.
"""

from __future__ import annotations

import numpy as np
from dsbench.sftgen import synth
from dsbench.sftgen.conventions import ALL_CONVENTIONS, DIALECT_DISPLAY
from dsbench.sftgen.dialect_conventions import generate
from dsbench.sftgen.schema import row_to_dict


def test_no_prompt_asks_for_a_number():
    domain = synth.build("web_sessions", seed=5, n=50)
    rng = np.random.default_rng(0)
    for conv in ALL_CONVENTIONS:
        for dialect in DIALECT_DISPLAY:
            assert conv.system(domain, dialect).endswith("in a ```sql code block.")
        for _ in range(20):
            question = conv.question(domain, conv.params(rng))
            assert not any(ask in question for ask in ("Reply with", "Give the", "numeric result"))


def test_generated_rows_answer_with_the_sql_and_keep_the_number_in_verification():
    rows, report = generate(seed=11, reps=1, n=300, dialects=["duckdb"])
    assert report["emitted"] == len(rows) > 0 and report["rejected"] == 0
    for row in map(row_to_dict, rows):
        answer = next(t for t in row["turns"] if t["role"] == "assistant")["content"]
        assert answer.startswith("```sql\n") and answer.endswith("\n```")
        assert "Answer:" not in answer
        assert row["verification"]["agrees"] and row["verification"]["truth"] is not None
