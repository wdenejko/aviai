"""Render the benchmark dashboard — one self-contained web app for all findings.

Reads reports/gemma-4/benchmarks.json (from `collect`) and emits a single HTML file that shows,
per product, the base -> finetuned -> combined lift plus a full comparison matrix. Output is a
self-contained fragment (inline CSS/JS, theme-aware) so it renders both as a claude.ai Artifact
and when opened straight off disk. Re-run after new runs land to refresh — products with no run
yet render as "running on dashi", so the app is useful before the NOTAM numbers arrive.

CLI:  uv run python -m avtext.report.dashboard   ->   reports/gemma-4/dashboard.html
"""
# ruff: noqa: E501 — this module is mostly an HTML/CSS template + narrative prose; hard-wrapping
# CSS rules and finding sentences at 100 cols hurts readability more than the long lines do.

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

# Canonical product order + one-line "what the task is". Drives both the cards and the matrix;
# a product with no runs yet still gets a card (so the layout is stable as results trickle in).
CANON: list[tuple[str, str, str, str]] = [
    ("metar", "METAR", "decode", "Raw METAR → 23-field structured decode."),
    ("taf", "TAF", "decode", "TAF → nested change-group forecast."),
    ("notam_ext", "NOTAM · extraction", "decode", "NOTAM E-field → category-specific rows."),
    ("notam_cls", "NOTAM · classification", "classification", "NOTAM → 1 of 13 op. classes."),
]
VARIANT_LABEL = {
    "base": "base",
    "single": "single-task FT",
    "combined": "combined FT (METAR+TAF)",
    "all": "all-products FT",
}
VARIANT_ORDER = {"base": 0, "single": 1, "combined": 2, "all": 3}

# Curated narrative — the findings, in the project's own voice. Kept here (not auto-derived) so
# the story stays deliberate; update alongside the data.
FINDINGS = [
    (
        "Finetuning is transformational, not marginal",
        "On every decode task the rank-16 LoRA turns a barely-usable base model into a near-solved "
        "one. Whole-record exact match jumps from single/low-double digits to ~90%. The base model "
        "knows the vocabulary but cannot hold the whole structured schema; the adapter teaches the "
        "format, not new meteorology.",
    ),
    (
        "One adapter for all products is free (slightly synergistic)",
        "A single combined adapter matches or beats each single-task adapter on its own eval "
        "(matched-N): METAR 91.1%→93.1% EM, TAF 89.1%→90.1% EM. Multi-task training does not cost "
        "accuracy here — so NOTAM folds into the same adapter rather than needing its own.",
    ),
    (
        "The reported hallucination rate overstates real fabrication",
        "METAR's ~7% field-level hallucination is ≥70% mislabelled parser-defeat *wins* (the model "
        "reads glued/doubled tokens the 3 gold parsers choke on, so gold is empty and a correct read "
        "scores as fabrication). True fabrication is ≤2.1%, dominated by slash-masked fields. Fix is "
        "targeted masking augmentation, not blanket abstention.",
    ),
    (
        "rank-16 is the deployment choice",
        "rank-64 bought only +0.2% value / +1.8% EM over rank-16 on METAR — capacity is not the "
        "bottleneck. rank-16 is the only rank carried forward.",
    ),
    (
        "No Chinese datapoints anywhere in the NOTAM corpus",
        "Hard constraint honoured: raw text and labels are English-only (recursive CJK guard). "
        "NOTAM-Evolve was rejected (88% Chinese labels); OpenNOTAM (label-identical to the Knots "
        "expert gold) + DEEL-AI (MIT) supply extraction (11,340) and classification (8,212).",
    ),
]


def _pct(x: float | None) -> str:
    return f"{x * 100:.1f}%" if isinstance(x, int | float) else "—"


def _e4b(runs: list[dict], product: str) -> list[dict]:
    rows = [r for r in runs if r["product"] == product and r["model_size"] == "E4B"]
    return sorted(
        rows,
        key=lambda r: (VARIANT_ORDER.get(r["variant"], 9), -(r["metrics"].get("n_records") or 0)),
    )


def _pick(rows: list[dict], variant: str) -> dict | None:
    cand = [r for r in rows if r["variant"] == variant]
    return max(cand, key=lambda r: r["metrics"].get("n_records") or 0) if cand else None


def _delta_badge(base: float | None, ft: float | None, *, good_up: bool = True) -> str:
    if not isinstance(base, int | float) or not isinstance(ft, int | float):
        return ""
    d = (ft - base) * 100
    up = d >= 0
    good = up == good_up
    arrow = "▲" if up else "▼"
    cls = "up" if good else "down"
    return f'<span class="delta {cls}">{arrow} {abs(d):.1f}pt</span>'


def _card(product: str, label: str, kind: str, blurb: str, rows: list[dict]) -> str:
    if not rows:
        return (
            f'<article class="card pending"><h3>{html.escape(label)}</h3>'
            f'<p class="blurb">{html.escape(blurb)}</p>'
            f'<div class="running">⏳ running on dashi</div></article>'
        )
    base = _pick(rows, "base")
    ft = _pick(rows, "all") or _pick(rows, "combined") or _pick(rows, "single")
    if kind == "classification":
        b_acc = base["metrics"].get("accuracy") if base else None
        f_acc = ft["metrics"].get("accuracy") if ft else None
        f_f1 = ft["metrics"].get("macro_f1") if ft else None
        big = _pct(f_acc)
        rowsub = (
            f'<div class="metric"><span>accuracy</span><b>{_pct(b_acc)} → {_pct(f_acc)}</b>'
            f"{_delta_badge(b_acc, f_acc)}</div>"
            f'<div class="metric"><span>macro-F1 (FT)</span><b>{_pct(f_f1)}</b></div>'
        )
    else:
        b_em = base["metrics"].get("exact_match") if base else None
        f_em = ft["metrics"].get("exact_match") if ft else None
        b_val = base["metrics"].get("value_acc") if base else None
        f_val = ft["metrics"].get("value_acc") if ft else None
        b_hal = base["metrics"].get("hallucination_rate") if base else None
        f_hal = ft["metrics"].get("hallucination_rate") if ft else None
        big = _pct(f_em)
        rowsub = (
            f'<div class="metric"><span>exact match</span><b>{_pct(b_em)} → {_pct(f_em)}</b>'
            f"{_delta_badge(b_em, f_em)}</div>"
            f'<div class="metric"><span>value acc</span><b>{_pct(b_val)} → {_pct(f_val)}</b>'
            f"{_delta_badge(b_val, f_val)}</div>"
            f'<div class="metric"><span>halluc</span><b>{_pct(b_hal)} → {_pct(f_hal)}</b>'
            f"{_delta_badge(b_hal, f_hal, good_up=False)}</div>"
        )
    ftname = VARIANT_LABEL.get(ft["variant"], ft["variant"]) if ft else "—"
    return (
        f'<article class="card"><h3>{html.escape(label)}</h3>'
        f'<p class="blurb">{html.escape(blurb)}</p>'
        f'<div class="big">{big}<span class="big-label">best FT ({ftname})</span></div>'
        f'<div class="metrics">{rowsub}</div></article>'
    )


def _matrix_row(r: dict) -> str:
    m = r["metrics"]
    v = VARIANT_LABEL.get(r["variant"], r["variant"])
    rank = f"r{r['rank']}" if r["rank"] else "—"
    n = m.get("n_records") or 0
    if m["kind"] == "classification":
        cells = f"<td>{_pct(m.get('accuracy'))}</td><td>{_pct(m.get('macro_f1'))}</td><td>—</td>"
    else:
        cells = (
            f"<td>{_pct(m.get('value_acc'))}</td><td class='em'>{_pct(m.get('exact_match'))}</td>"
            f"<td>{_pct(m.get('hallucination_rate'))}</td>"
        )
    return (
        f'<tr class="v-{r["variant"]}"><td>{html.escape(v)}</td><td>{rank}</td>'
        f'<td class="num">{n:,}</td>{cells}</tr>'
    )


def _matrix(runs: list[dict]) -> str:
    blocks = []
    for product, label, _kind, _blurb in CANON:
        rows = _e4b(runs, product)
        head = (
            "<tr><th>variant</th><th>rank</th><th>n</th>"
            "<th>value acc / accuracy</th><th>EM / macro-F1</th><th>halluc</th></tr>"
        )
        if not rows:
            body = '<tr><td colspan="6" class="running">⏳ running on dashi</td></tr>'
        else:
            body = "".join(_matrix_row(r) for r in rows)
        blocks.append(
            f'<section class="pblock"><h3>{html.escape(label)}</h3>'
            f'<div class="tw"><table class="matrix">{head}{body}</table></div></section>'
        )
    return "".join(blocks)


def _all_runs(runs: list[dict]) -> str:
    head = (
        "<tr><th>date</th><th>model</th><th>product</th><th>variant</th>"
        "<th>n</th><th>value/acc</th><th>EM/F1</th><th>halluc</th></tr>"
    )
    body = []
    for r in sorted(runs, key=lambda r: r.get("created_utc") or ""):
        m = r["metrics"]
        if m["kind"] == "classification":
            a, b, c = _pct(m.get("accuracy")), _pct(m.get("macro_f1")), "—"
        else:
            a, b, c = (
                _pct(m.get("value_acc")),
                _pct(m.get("exact_match")),
                _pct(m.get("hallucination_rate")),
            )
        date = (r.get("created_utc") or "")[:10]
        body.append(
            f"<tr><td class='num'>{date}</td><td>{html.escape(r['model_id'])}</td>"
            f"<td>{html.escape(r['product_label'])}</td><td>{html.escape(r['variant'])}</td>"
            f"<td class='num'>{(m.get('n_records') or 0):,}</td>"
            f"<td>{a}</td><td>{b}</td><td>{c}</td></tr>"
        )
    return f'<div class="tw"><table class="allruns">{head}{"".join(body)}</table></div>'


def _findings() -> str:
    items = "".join(
        f"<li><b>{html.escape(t)}.</b> {html.escape(body)}</li>" for t, body in FINDINGS
    )
    return f"<ol class='findings'>{items}</ol>"


def render(data: dict) -> str:
    runs = data["runs"]
    done = {r["product"] for r in runs}
    cards = "".join(_card(p, label, kind, blurb, _e4b(runs, p)) for p, label, kind, blurb in CANON)
    n_products_done = len([p for p, *_ in CANON if p in done])
    chips = (
        f'<span class="chip">{len(runs)} runs</span>'
        f'<span class="chip">{n_products_done}/{len(CANON)} products</span>'
        f'<span class="chip">Gemma-4-E4B · rank-16 LoRA</span>'
    )
    payload = json.dumps(data)
    return _TEMPLATE.format(
        chips=chips,
        cards=cards,
        matrix=_matrix(runs),
        findings=_findings(),
        allruns=_all_runs(runs),
        payload=html.escape(payload),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the benchmark dashboard web app.")
    ap.add_argument("--data", type=Path, default=Path("reports/gemma-4/benchmarks.json"))
    ap.add_argument("--out", type=Path, default=Path("reports/gemma-4/dashboard.html"))
    args = ap.parse_args()
    data = json.loads(args.data.read_text())
    args.out.write_text(render(data), encoding="utf-8")
    print(f"wrote dashboard ({len(data['runs'])} runs) -> {args.out}")


_TEMPLATE = """<meta charset="utf-8">
<title>avtext — Gemma-4 finetuning benchmarks</title>
<style>
:root {{
  --bg:#f4f6f9; --surface:#ffffff; --surface2:#eef1f6; --ink:#111821; --muted:#5b6673;
  --border:#dde3ec; --amber:#c8842a; --cyan:#1f8fa3; --green:#2f9e5f; --red:#cf5147;
  --shadow:0 1px 2px rgba(16,24,40,.06),0 8px 24px rgba(16,24,40,.05);
}}
@media (prefers-color-scheme:dark) {{
  :root {{ --bg:#0d1117; --surface:#161b22; --surface2:#1b2129; --ink:#e6edf3; --muted:#8b98a6;
    --border:#232b35; --amber:#e0a13c; --cyan:#3fb6c4; --green:#3fae6b; --red:#e0655b;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.3); }}
}}
:root[data-theme="light"] {{ --bg:#f4f6f9; --surface:#fff; --surface2:#eef1f6; --ink:#111821;
  --muted:#5b6673; --border:#dde3ec; --amber:#c8842a; --cyan:#1f8fa3; --green:#2f9e5f; --red:#cf5147; }}
:root[data-theme="dark"] {{ --bg:#0d1117; --surface:#161b22; --surface2:#1b2129; --ink:#e6edf3;
  --muted:#8b98a6; --border:#232b35; --amber:#e0a13c; --cyan:#3fb6c4; --green:#3fae6b; --red:#e0655b; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  line-height:1.5; -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:1080px; margin:0 auto; padding:32px 24px 80px; }}
header.top {{ border-bottom:1px solid var(--border); padding-bottom:20px; margin-bottom:28px; }}
.eyebrow {{ font-size:12px; letter-spacing:.14em; text-transform:uppercase; color:var(--amber);
  font-weight:700; }}
h1 {{ font-size:clamp(26px,4vw,38px); margin:.15em 0 .1em; letter-spacing:-.02em;
  text-wrap:balance; }}
.sub {{ color:var(--muted); max-width:60ch; }}
.chips {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:14px; }}
.chip {{ font-size:12px; background:var(--surface2); border:1px solid var(--border);
  color:var(--muted); padding:4px 10px; border-radius:999px; font-variant-numeric:tabular-nums; }}
h2 {{ font-size:13px; letter-spacing:.12em; text-transform:uppercase; color:var(--muted);
  margin:44px 0 16px; font-weight:700; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:16px; }}
.card {{ background:var(--surface); border:1px solid var(--border); border-radius:14px;
  padding:18px; box-shadow:var(--shadow); display:flex; flex-direction:column; }}
.card h3 {{ margin:0 0 2px; font-size:16px; }}
.card .blurb {{ color:var(--muted); font-size:12.5px; margin:0 0 14px; min-height:2.6em; }}
.big {{ font-size:38px; font-weight:750; letter-spacing:-.03em; line-height:1;
  font-variant-numeric:tabular-nums; }}
.big-label {{ display:block; font-size:11px; font-weight:600; color:var(--muted);
  letter-spacing:.02em; margin-top:4px; text-transform:none; }}
.metrics {{ margin-top:14px; display:flex; flex-direction:column; gap:7px; }}
.metric {{ display:flex; align-items:center; gap:8px; font-size:13px;
  font-variant-numeric:tabular-nums; }}
.metric span {{ color:var(--muted); width:78px; flex:none; }}
.metric b {{ font-weight:650; }}
.delta {{ font-size:11px; font-weight:700; margin-left:auto; padding:1px 6px; border-radius:6px; }}
.delta.up {{ color:var(--green); background:color-mix(in srgb,var(--green) 14%,transparent); }}
.delta.down {{ color:var(--red); background:color-mix(in srgb,var(--red) 14%,transparent); }}
.card.pending {{ opacity:.85; }}
.running {{ color:var(--amber); font-weight:650; font-size:13px; padding:14px 0; }}
.pblock {{ margin-bottom:18px; }}
.pblock h3 {{ font-size:15px; margin:0 0 8px; }}
.tw {{ overflow-x:auto; border:1px solid var(--border); border-radius:12px; background:var(--surface); }}
table {{ border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums; }}
th {{ text-align:right; font-weight:600; color:var(--muted); font-size:11.5px;
  text-transform:uppercase; letter-spacing:.04em; padding:10px 14px; border-bottom:1px solid var(--border); }}
th:first-child {{ text-align:left; }}
td {{ text-align:right; padding:9px 14px; border-bottom:1px solid var(--border); }}
td:first-child {{ text-align:left; font-weight:600; }}
tr:last-child td {{ border-bottom:none; }}
td.em {{ font-weight:750; }}
td.num {{ color:var(--muted); }}
tr.v-combined td:first-child {{ color:var(--cyan); }}
tr.v-all td:first-child {{ color:var(--green); font-weight:750; }}
tr.v-single td:first-child {{ color:var(--amber); }}
.matrix td.running, table td.running {{ text-align:left; }}
.findings {{ padding-left:20px; margin:0; display:flex; flex-direction:column; gap:12px; }}
.findings li {{ padding-left:4px; }}
.findings b {{ color:var(--ink); }}
details {{ margin-top:14px; }}
summary {{ cursor:pointer; color:var(--muted); font-size:13px; font-weight:600;
  padding:8px 0; user-select:none; }}
.allruns td, .allruns th {{ padding:7px 12px; font-size:12px; }}
footer {{ margin-top:48px; color:var(--muted); font-size:12px; border-top:1px solid var(--border);
  padding-top:16px; }}
</style>
<div class="wrap">
  <header class="top">
    <div class="eyebrow">avtext · aviation-text LLM lab</div>
    <h1>Gemma-4 finetuning benchmarks</h1>
    <p class="sub">Does a rank-16 LoRA on Gemma-4-E4B beat the base model at decoding aviation
      weather &amp; NOTAM text — and can one adapter do all of it? Every frozen eval, base vs
      finetuned vs combined, in one view.</p>
    <div class="chips">{chips}</div>
  </header>

  <h2>Headline — base → best finetune</h2>
  <div class="cards">{cards}</div>

  <h2>Comparison matrix · Gemma-4-E4B</h2>
  {matrix}

  <h2>Findings</h2>
  {findings}

  <h2>All runs</h2>
  <details><summary>Every recorded run (incl. E2B &amp; legacy v1 evals)</summary>
  {allruns}</details>

  <footer>Generated from <code>reports/gemma-4/benchmarks.json</code> ·
    5-way scoring (HIT / WRONG / ABSTAIN / HALLUCINATE / TRUE_ABSTAIN), whole-record exact match ·
    hallucination on decode tasks is field-level and over-counts parser-defeat wins (see Findings).</footer>
</div>
<script type="application/json" id="benchmarks">{payload}</script>
"""


if __name__ == "__main__":
    main()
