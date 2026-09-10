"""Scoring arithmetic and ranking. Pure Python, no LLM, no I/O."""
from typing import Dict, List, Optional, Tuple

from src import config
from src.models import CandidateResult, GitHubEnrichment, LLMScores


def compute_total(scores: LLMScores, github: Optional[GitHubEnrichment]) -> int:
    """Sum the five categories, subtract penalties, clamp to 0-100."""
    github_points = github.total if github else 0
    raw = (
        scores.ai_project_depth
        + scores.python_backend
        + scores.cloud_fullstack
        + scores.engineering_depth
        + github_points
    )
    deductions = sum(penalty.points for penalty in scores.penalties)
    return max(0, min(config.TOTAL_MAX, raw - deductions))


def build_breakdown(
    scores: LLMScores, github: Optional[GitHubEnrichment]
) -> Dict[str, int]:
    return {
        "ai_project_depth": scores.ai_project_depth,
        "python_backend": scores.python_backend,
        "cloud_fullstack": scores.cloud_fullstack,
        "github": github.total if github else 0,
        "engineering_depth": scores.engineering_depth,
        "penalties": -sum(penalty.points for penalty in scores.penalties),
    }


def _sort_key(result: CandidateResult) -> Tuple:
    """Descending score, then AI depth, then Python, then name.

    Name last so the order is stable and a re-run produces byte-identical
    output for the same inputs.
    """
    breakdown = result.score_breakdown or {}
    return (
        -(result.total_score or 0),
        -breakdown.get("ai_project_depth", 0),
        -breakdown.get("python_backend", 0),
        result.candidate_name.lower(),
    )


def rank_candidates(results: List[CandidateResult]) -> List[CandidateResult]:
    """Sort scored candidates and assign ranks starting at 1.

    Eligible-but-unscored candidates (an LLM failure) sort to the bottom and
    keep rank = None, same as rejected ones.
    """
    scored = [r for r in results if r.eligible and r.total_score is not None]
    unscored = [r for r in results if r.eligible and r.total_score is None]

    scored.sort(key=_sort_key)
    for position, result in enumerate(scored, start=1):
        result.rank = position

    for result in unscored:
        result.rank = None

    return scored + sorted(unscored, key=lambda r: r.candidate_name.lower())
