"""Batch orchestration. Every per-candidate stage is wrapped: one unreadable
resume, one refused LLM call, or one dead GitHub link must never end the batch.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src import config
from src.github.client import enrich
from src.llm import client as llm_client
from src.llm.extractor import extract_profile
from src.llm.scorer import score_profile
from src.models import CandidateProfile, CandidateResult, GitHubEnrichment
from src.parser import collect_resumes, is_usable, parse_resume
from src.screening.eligibility import check_eligibility
from src.screening.ranking import build_breakdown, compute_total, rank_candidates


@dataclass
class Outcome:
    source_file: str
    result: Optional[CandidateResult] = None
    failure: Optional[str] = None
    github_status: str = "missing"
    llm_failed: bool = False


@dataclass
class PipelineOptions:
    use_github: bool = True
    use_cache: bool = True
    offline: bool = False
    progress: Optional[Callable[[str], None]] = None
    max_workers: int = field(default_factory=lambda: config.LLM_MAX_WORKERS)


def _fallback_name(path: Path) -> str:
    return path.stem.replace("_", " ").title()


def process_one(path: Path, options: PipelineOptions) -> Outcome:
    outcome = Outcome(source_file=path.name)
    errors: List[str] = []

    parsed = parse_resume(path)
    usable, reason = is_usable(parsed)
    if not usable:
        outcome.failure = reason or "Unreadable file"
        return outcome

    # --- Extraction ---------------------------------------------------------
    try:
        profile = extract_profile(parsed, errors)
    except Exception as exc:  # noqa: BLE001
        profile = None
        errors.append(f"extraction_error: {exc}")

    if profile is None:
        # A failed file, not a rejection: never claim someone is ineligible when
        # the truth is we could not read them.
        outcome.llm_failed = True
        outcome.failure = errors[-1] if errors else "llm_unavailable: no result"
        return outcome

    if not (profile.candidate_name or "").strip():
        profile.candidate_name = _fallback_name(path)

    # --- Eligibility (deterministic) ----------------------------------------
    eligibility = check_eligibility(profile)

    result = CandidateResult(
        candidate_name=profile.candidate_name,
        source_file=path.name,
        eligible=eligibility.eligible,
        matched_skills=eligibility.matched_skills,
        rejection_reasons=eligibility.rejection_reasons,
        errors=errors,
    )
    outcome.result = result

    if not eligibility.eligible:
        return outcome

    # --- Scoring (eligible only) --------------------------------------------
    try:
        scores = score_profile(profile, result.errors)
    except Exception as exc:  # noqa: BLE001
        scores = None
        result.errors.append(f"scoring_error: {exc}")

    if scores is None:
        # Eligible but unscored: kept in the output with rank=null so the
        # reviewer sees who we failed to score, and why.
        outcome.llm_failed = True
        result.project_summary = "Not scored - LLM scoring unavailable."
        return outcome

    # --- GitHub (eligible only, fail-soft) ----------------------------------
    github: Optional[GitHubEnrichment] = None
    if options.use_github:
        try:
            github = enrich(profile.github_url, use_cache=options.use_cache)
        except Exception as exc:  # noqa: BLE001 - defence in depth
            result.errors.append(f"github_error: {exc}")
    outcome.github_status = github.status if github else "skipped"

    result.total_score = compute_total(scores, github)
    result.score_breakdown = build_breakdown(scores, github)
    result.project_summary = scores.project_summary
    result.strengths = scores.strengths
    result.concerns = list(scores.concerns)
    result.github_summary = github.summary if github else "GitHub enrichment skipped."

    if eligibility.weak_ai_only and config.WEAK_AI_CONCERN not in result.concerns:
        result.concerns.append(config.WEAK_AI_CONCERN)

    return outcome


def run(input_dir, options: Optional[PipelineOptions] = None, limit: Optional[int] = None
        ) -> Dict[str, Any]:
    options = options or PipelineOptions()
    started = time.time()

    files, duplicates = collect_resumes(input_dir)
    if limit is not None:
        files = files[:limit]

    total = len(files)
    outcomes: List[Outcome] = []

    def task(indexed):
        index, path = indexed
        try:
            outcome = process_one(path, options)
        except Exception as exc:  # noqa: BLE001 - last line of defence
            outcome = Outcome(source_file=path.name, failure=f"unexpected_error: {exc}")

        if options.progress:
            options.progress(_progress_line(index, total, outcome))
        return outcome

    # Network-bound work; 4 concurrent requests is enough without tripping
    # provider rate limits.
    with ThreadPoolExecutor(max_workers=options.max_workers) as pool:
        outcomes = list(pool.map(task, enumerate(files, start=1)))

    return _assemble(outcomes, duplicates, total, time.time() - started)


def _progress_line(index: int, total: int, outcome: Outcome) -> str:
    prefix = f"[{index}/{total}] {outcome.source_file:<24}"
    if outcome.failure:
        return f"{prefix} -> failed    {outcome.failure[:60]}"
    result = outcome.result
    if result is None or not result.eligible:
        reasons = ", ".join(result.rejection_reasons) if result else "unknown"
        return f"{prefix} -> rejected  {reasons}"
    if result.total_score is None:
        return f"{prefix} -> eligible  score=n/a (llm unavailable)"
    return f"{prefix} -> eligible  score={result.total_score}"


def _assemble(
    outcomes: List[Outcome], duplicates: List[Path], total: int, elapsed: float
) -> Dict[str, Any]:
    eligible = [o for o in outcomes if o.result and o.result.eligible]
    rejected = [o.result for o in outcomes if o.result and not o.result.eligible]
    failed = [o for o in outcomes if o.failure]

    ranked = rank_candidates([o.result for o in eligible])

    github_enriched = sum(1 for o in eligible if o.github_status == "ok")
    github_failed = sum(
        1 for o in eligible if o.github_status in ("not_found", "rate_limited", "error")
    )

    summary = {
        "total_resumes": total + len(duplicates),
        "successfully_parsed": total - len(failed),
        "eligible": len(eligible),
        "rejected": len(rejected),
        "failed_unreadable": len(failed),
        "duplicates_skipped": len(duplicates),
        "github_enriched": github_enriched,
        "github_failed": github_failed,
        "llm_failures": sum(1 for o in outcomes if o.llm_failed),
        "run_seconds": round(elapsed, 1),
    }

    # A dead credential is one run-level problem, not 50 candidate problems.
    provider_error = llm_client.breaker_reason()
    if provider_error:
        summary["llm_provider_error"] = provider_error

    return {
        "batch_summary": summary,
        "ranked_candidates": [
            _public_fields(r, ranked=True) for r in ranked
        ],
        "rejected_candidates": [
            _public_fields(r, ranked=False)
            for r in sorted(rejected, key=lambda r: r.source_file)
        ],
        "failed_files": [
            {"source_file": o.source_file, "error": o.failure}
            for o in sorted(failed, key=lambda o: o.source_file)
        ],
        "duplicate_files": [p.name for p in duplicates],
    }


def _public_fields(result: CandidateResult, *, ranked: bool) -> Dict[str, Any]:
    if not ranked:
        return {
            "candidate_name": result.candidate_name,
            "source_file": result.source_file,
            "eligible": False,
            "rejection_reasons": result.rejection_reasons,
            "matched_skills": result.matched_skills,
        }

    payload = {
        "rank": result.rank,
        "candidate_name": result.candidate_name,
        "source_file": result.source_file,
        "eligible": True,
        "total_score": result.total_score,
        "score_breakdown": result.score_breakdown,
        "matched_skills": result.matched_skills,
        "project_summary": result.project_summary,
        "github_summary": result.github_summary,
        "strengths": result.strengths,
        "concerns": result.concerns,
    }
    if result.errors:
        payload["errors"] = result.errors
    return payload
