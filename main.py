"""CLI entry point.

    python main.py --input ./resumes --output ./output/results.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from src import config
from src.llm import client as llm_client
from src.pipeline import PipelineOptions, run
from src.report import render_html


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Screen and rank resumes for an AI/Python intern role.",
    )
    parser.add_argument("--input", default="./resumes", help="Folder of resumes")
    parser.add_argument(
        "--output", default="./output/results.json", help="Where to write results JSON"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only the first N resumes"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=config.DEFAULT_TOP_N,
        help="Rows in the terminal ranking table",
    )
    parser.add_argument(
        "--no-github", action="store_true", help="Skip GitHub enrichment"
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Ignore the LLM and GitHub caches"
    )
    parser.add_argument(
        "--no-html", action="store_true", help="Skip writing the HTML report"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Serve LLM results from cache only; never call the network",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print a line per resume while running"
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Input folder not found: {input_dir.resolve()}", file=sys.stderr)
        return 2

    llm_client.configure(offline=args.offline, use_cache=not args.no_cache)
    try:
        llm_client.require_credentials()
    except llm_client.LLMUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 2

    options = PipelineOptions(
        use_github=not args.no_github,
        use_cache=not args.no_cache,
        offline=args.offline,
        progress=print if args.verbose else None,
    )

    print(f"Screening resumes in {input_dir.resolve()} ...")
    if not args.verbose:
        print("(run with --verbose for a line per resume)")

    results = run(input_dir, options, limit=args.limit)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    html_path = None
    if not args.no_html:
        html_path = output_path.with_suffix(".html")
        html_path.write_text(render_html(results), encoding="utf-8")

    print_report(results, top_n=args.top)

    print(f"\nJSON:  {output_path.resolve()}")
    if html_path:
        print(f"HTML:  {html_path.resolve()}")
    return 0


# --- Terminal report --------------------------------------------------------
# str.ljust/rjust only - plain ASCII renders in any terminal and pastes cleanly
# into a README.

COLUMNS = [
    ("RANK", 4, "rjust"),
    ("CANDIDATE", 22, "ljust"),
    ("SCORE", 5, "rjust"),
    ("AI/40", 5, "rjust"),
    ("PY/30", 5, "rjust"),
    ("CLOUD/15", 8, "rjust"),
    ("GH/10", 5, "rjust"),
    ("ENG/5", 5, "rjust"),
]


def _row(values) -> str:
    cells = []
    for (_, width, align), value in zip(COLUMNS, values):
        text = str(value)
        if len(text) > width:
            text = text[: width - 1] + "."
        cells.append(getattr(text, align)(width))
    return "  ".join(cells)


def print_report(results: Dict[str, Any], top_n: int = config.DEFAULT_TOP_N) -> None:
    ranked = results.get("ranked_candidates", [])
    summary = results.get("batch_summary", {})

    print(f"\nTOP {min(top_n, len(ranked))} CANDIDATES")
    print(_row([label for label, _, _ in COLUMNS]))
    print("-" * len(_row([label for label, _, _ in COLUMNS])))

    for candidate in ranked[:top_n]:
        breakdown = candidate.get("score_breakdown") or {}
        print(
            _row(
                [
                    candidate.get("rank") or "-",
                    candidate.get("candidate_name", ""),
                    candidate.get("total_score", "-"),
                    breakdown.get("ai_project_depth", "-"),
                    breakdown.get("python_backend", "-"),
                    breakdown.get("cloud_fullstack", "-"),
                    breakdown.get("github", "-"),
                    breakdown.get("engineering_depth", "-"),
                ]
            )
        )

    if not ranked:
        print("(no eligible candidates)")

    print("\nBATCH SUMMARY")
    for key, value in summary.items():
        if key == "llm_provider_error":
            continue
        print(f"  {key.replace('_', ' '):<22} {value}")

    provider_error = summary.get("llm_provider_error")
    if provider_error:
        print(f"\n!! LLM provider stopped the run: {provider_error}")
        print("   Fix the credential or credit issue, then re-run.")
        print("   Cached candidates are replayed for free, so only the")
        print("   unprocessed resumes will cost anything.")

    _print_rejection_summary(results.get("rejected_candidates", []))

    failed = results.get("failed_files", [])
    if failed:
        print(f"\nFAILED FILES ({len(failed)})")
        for item in failed:
            print(f"  {item['source_file']:<24} {item['error']}")


def _print_rejection_summary(rejected) -> None:
    if not rejected:
        return

    counts = Counter()
    for candidate in rejected:
        reasons = candidate.get("rejection_reasons") or []
        counts["both" if len(reasons) > 1 else (reasons[0] if reasons else "unknown")] += 1

    label = {
        config.REJECT_NO_PYTHON: "no Python evidence",
        config.REJECT_NO_AI: "no AI evidence",
        "both": "both",
    }
    parts = [f"{count} {label.get(reason, reason)}" for reason, count in counts.most_common()]
    print(f"\n{len(rejected)} rejected -- " + ", ".join(parts))


if __name__ == "__main__":
    raise SystemExit(main())
