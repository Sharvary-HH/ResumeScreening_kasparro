"""Self-contained HTML report.

Stdlib only, one file, no CDN links - it has to open from file:// with no
network. Reads the results dict (i.e. results.json), never pipeline internals,
so it can be regenerated without re-running the batch:

    python -m src.report output/results.json

Everything interpolated goes through html.escape(): resume text is untrusted
input and a candidate named "<script>" must not execute.
"""
import html
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from src import config

STYLE = """
:root { color-scheme: light; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 0 auto; padding: 32px 20px; max-width: 1100px; color: #1c1c1e;
       background: #fbfbfd; line-height: 1.5; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 36px 0 10px; text-transform: uppercase;
     letter-spacing: .06em; color: #55555c; }
.sub { color: #6a6a72; margin: 0 0 24px; font-size: 13px; }
.summary { display: flex; flex-wrap: wrap; gap: 10px; }
.stat { background: #fff; border: 1px solid #e2e2e8; border-radius: 8px;
        padding: 10px 14px; min-width: 104px; }
.stat b { display: block; font-size: 20px; }
.stat span { font-size: 11px; color: #6a6a72; text-transform: uppercase;
             letter-spacing: .05em; }
table { border-collapse: collapse; width: 100%; font-size: 13px; background: #fff;
        border: 1px solid #e2e2e8; border-radius: 8px; overflow: hidden; }
th, td { padding: 7px 10px; text-align: left; border-bottom: 1px solid #eeeef2; }
th { background: #f4f4f7; font-weight: 600; cursor: pointer; user-select: none;
     white-space: nowrap; }
tbody tr:nth-child(even) { background: #fafafc; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.bar { background-image: linear-gradient(to right, #dce7f5 var(--pct), transparent var(--pct)); }
details { background: #fff; border: 1px solid #e2e2e8; border-radius: 8px;
          margin-bottom: 8px; padding: 10px 14px; }
summary { cursor: pointer; font-weight: 600; font-size: 14px; }
details dl { margin: 12px 0 2px; font-size: 13px; }
details dt { font-weight: 600; color: #55555c; margin-top: 9px; font-size: 12px;
             text-transform: uppercase; letter-spacing: .04em; }
details dd { margin: 3px 0 0; }
ul { margin: 3px 0; padding-left: 20px; }
.tag { display: inline-block; background: #eef1f6; border-radius: 4px;
       padding: 1px 7px; margin: 2px 3px 2px 0; font-size: 12px; }
.penalty { color: #a3402f; }
.muted { color: #6a6a72; }
"""

# Click a header to sort. Kept deliberately small.
SCRIPT = """
document.querySelectorAll('table.sortable th').forEach(function (th, index) {
  th.addEventListener('click', function () {
    var body = th.closest('table').tBodies[0];
    var rows = Array.prototype.slice.call(body.rows);
    var asc = th.dataset.asc !== 'true';
    th.dataset.asc = asc;
    rows.sort(function (a, b) {
      var x = a.cells[index].dataset.sort || a.cells[index].textContent;
      var y = b.cells[index].dataset.sort || b.cells[index].textContent;
      var diff = isNaN(x) || isNaN(y) ? String(x).localeCompare(String(y)) : x - y;
      return asc ? diff : -diff;
    });
    rows.forEach(function (row) { body.appendChild(row); });
  });
});
"""

SUMMARY_LABELS = [
    ("total_resumes", "Resumes"),
    ("successfully_parsed", "Parsed"),
    ("eligible", "Eligible"),
    ("rejected", "Rejected"),
    ("failed_unreadable", "Failed"),
    ("duplicates_skipped", "Duplicates"),
    ("github_enriched", "GitHub OK"),
    ("github_failed", "GitHub failed"),
    ("llm_failures", "LLM failures"),
    ("run_seconds", "Seconds"),
]

SCORE_COLUMNS = [
    ("ai_project_depth", "AI", config.AI_MAX),
    ("python_backend", "Python", config.PYTHON_MAX),
    ("cloud_fullstack", "Cloud", config.CLOUD_MAX),
    ("github", "GitHub", config.GITHUB_MAX),
    ("engineering_depth", "Eng", config.ENGINEERING_MAX),
]


def e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _tags(items: List[str]) -> str:
    if not items:
        return '<span class="muted">none</span>'
    return "".join(f'<span class="tag">{e(item)}</span>' for item in items)


def _bullets(items: List[str]) -> str:
    if not items:
        return '<span class="muted">none</span>'
    return "<ul>" + "".join(f"<li>{e(item)}</li>" for item in items) + "</ul>"


def _score_cell(value: Any, maximum: int) -> str:
    try:
        pct = max(0, min(100, round(float(value) / maximum * 100)))
    except (TypeError, ValueError, ZeroDivisionError):
        return f'<td class="num">{e(value)}</td>'
    return (
        f'<td class="num bar" style="--pct:{pct}%" data-sort="{e(value)}">'
        f"{e(value)}</td>"
    )


def _summary_block(summary: Dict[str, Any]) -> str:
    stats = "".join(
        f"<div class='stat'><b>{e(summary.get(key, 0))}</b><span>{e(label)}</span></div>"
        for key, label in SUMMARY_LABELS
    )
    return f"<h2>Batch summary</h2><div class='summary'>{stats}</div>"


def _ranking_table(ranked: List[Dict[str, Any]]) -> str:
    if not ranked:
        return "<h2>Ranked candidates</h2><p class='muted'>No eligible candidates.</p>"

    headers = "".join(
        f"<th>{e(label)}/{maximum}</th>" for _, label, maximum in SCORE_COLUMNS
    )
    rows = []
    for candidate in ranked:
        breakdown = candidate.get("score_breakdown") or {}
        cells = "".join(
            _score_cell(breakdown.get(key, 0), maximum)
            for key, _, maximum in SCORE_COLUMNS
        )
        penalties = breakdown.get("penalties", 0)
        rows.append(
            "<tr>"
            f"<td class='num'>{e(candidate.get('rank'))}</td>"
            f"<td>{e(candidate.get('candidate_name'))}</td>"
            f"<td class='muted'>{e(candidate.get('source_file'))}</td>"
            f"{_score_cell(candidate.get('total_score', 0), config.TOTAL_MAX)}"
            f"{cells}"
            f"<td class='num penalty'>{e(penalties)}</td>"
            "</tr>"
        )

    return (
        "<h2>Ranked candidates</h2>"
        "<table class='sortable'><thead><tr>"
        "<th>#</th><th>Candidate</th><th>File</th>"
        f"<th>Total/{config.TOTAL_MAX}</th>{headers}<th>Penalty</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _detail_blocks(ranked: List[Dict[str, Any]]) -> str:
    blocks = []
    for candidate in ranked:
        breakdown = candidate.get("score_breakdown") or {}
        penalties = breakdown.get("penalties", 0)
        penalty_line = (
            f"<dt>Penalties</dt><dd class='penalty'>{e(penalties)} points</dd>"
            if penalties
            else ""
        )
        errors = candidate.get("errors") or []
        error_line = (
            f"<dt>Errors</dt><dd class='penalty'>{_bullets(errors)}</dd>"
            if errors
            else ""
        )
        blocks.append(
            "<details><summary>"
            f"#{e(candidate.get('rank'))} &nbsp; {e(candidate.get('candidate_name'))}"
            f" &nbsp; <span class='muted'>{e(candidate.get('total_score'))}/"
            f"{config.TOTAL_MAX}</span></summary><dl>"
            f"<dt>Project summary</dt><dd>{e(candidate.get('project_summary'))}</dd>"
            f"<dt>Matched skills</dt><dd>{_tags(candidate.get('matched_skills') or [])}</dd>"
            f"<dt>Strengths</dt><dd>{_bullets(candidate.get('strengths') or [])}</dd>"
            f"<dt>Concerns</dt><dd>{_bullets(candidate.get('concerns') or [])}</dd>"
            f"<dt>GitHub</dt><dd>{e(candidate.get('github_summary'))}</dd>"
            f"{penalty_line}{error_line}"
            "</dl></details>"
        )
    return "<h2>Candidate detail</h2>" + "".join(blocks) if blocks else ""


def _rejected_table(rejected: List[Dict[str, Any]]) -> str:
    if not rejected:
        return ""
    rows = "".join(
        "<tr>"
        f"<td>{e(candidate.get('candidate_name'))}</td>"
        f"<td class='muted'>{e(candidate.get('source_file'))}</td>"
        f"<td>{e('; '.join(candidate.get('rejection_reasons') or []))}</td>"
        f"<td>{_tags(candidate.get('matched_skills') or [])}</td>"
        "</tr>"
        for candidate in rejected
    )
    return (
        f"<h2>Rejected ({len(rejected)})</h2>"
        "<table class='sortable'><thead><tr><th>Candidate</th><th>File</th>"
        "<th>Reasons</th><th>Matched skills</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _failed_table(failed: List[Dict[str, Any]]) -> str:
    if not failed:
        return ""
    rows = "".join(
        f"<tr><td class='muted'>{e(item.get('source_file'))}</td>"
        f"<td>{e(item.get('error'))}</td></tr>"
        for item in failed
    )
    return (
        f"<h2>Failed files ({len(failed)})</h2>"
        "<table><thead><tr><th>File</th><th>Error</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def render_html(results: Dict[str, Any]) -> str:
    ranked = results.get("ranked_candidates", [])
    summary = results.get("batch_summary", {})

    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Resume Screening Results</title>"
        f"<style>{STYLE}</style></head><body>"
        "<h1>Resume Screening Results</h1>"
        f"<p class='sub'>{e(summary.get('total_resumes', 0))} resumes screened in "
        f"{e(summary.get('run_seconds', 0))}s &middot; model "
        f"{e(config.LLM_MODEL)}</p>"
        f"{_summary_block(summary)}"
        f"{_ranking_table(ranked)}"
        f"{_detail_blocks(ranked)}"
        f"{_rejected_table(results.get('rejected_candidates', []))}"
        f"{_failed_table(results.get('failed_files', []))}"
        f"<script>{SCRIPT}</script></body></html>"
    )


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    source = Path(argv[0]) if argv else Path("output/results.json")
    if not source.exists():
        print(f"No results file at {source.resolve()}", file=sys.stderr)
        return 2

    results = json.loads(source.read_text(encoding="utf-8"))
    target = source.with_suffix(".html")
    target.write_text(render_html(results), encoding="utf-8")
    print(target.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
