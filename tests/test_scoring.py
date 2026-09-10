"""Scoring arithmetic, GitHub thresholds, and ranking order.

All pure functions - no network, no LLM.
"""
from datetime import datetime, timedelta, timezone

import pytest

from src.github.client import is_relevant_repo, score_activity, score_repos
from src.models import CandidateResult, GitHubEnrichment, LLMScores, Penalty
from src.screening.ranking import build_breakdown, compute_total, rank_candidates

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def scores(**kwargs) -> LLMScores:
    defaults = dict(
        ai_project_depth=30,
        python_backend=20,
        cloud_fullstack=10,
        engineering_depth=3,
    )
    defaults.update(kwargs)
    return LLMScores(**defaults)


def github(total: int) -> GitHubEnrichment:
    return GitHubEnrichment(
        status="ok", activity_score=min(total, 5), repo_score=max(total - 5, 0),
        total=total,
    )


# --- Totals -----------------------------------------------------------------


def test_total_is_the_sum_of_all_five_categories():
    assert compute_total(scores(), github(8)) == 30 + 20 + 10 + 3 + 8


def test_missing_github_contributes_zero():
    assert compute_total(scores(), None) == 63


def test_penalties_are_subtracted():
    penalised = scores(penalties=[Penalty(reason="Thin LLM wrapper", points=15)])
    assert compute_total(penalised, github(8)) == 71 - 15


@pytest.mark.parametrize("points", [5, 15])
def test_penalties_at_both_bounds_apply(points):
    penalised = scores(penalties=[Penalty(reason="Tutorial-style project", points=points)])
    assert compute_total(penalised, None) == 63 - points


def test_multiple_penalties_stack():
    penalised = scores(
        penalties=[
            Penalty(reason="Thin wrapper", points=10),
            Penalty(reason="No ownership evidence", points=5),
        ]
    )
    assert compute_total(penalised, None) == 63 - 15


def test_score_clamps_at_zero():
    low = scores(
        ai_project_depth=2,
        python_backend=1,
        cloud_fullstack=0,
        engineering_depth=0,
        penalties=[Penalty(reason="Thin wrapper", points=15)],
    )
    assert compute_total(low, None) == 0


def test_score_clamps_at_one_hundred():
    perfect = scores(
        ai_project_depth=40, python_backend=30, cloud_fullstack=15, engineering_depth=5
    )
    assert compute_total(perfect, github(10)) == 100


def test_penalty_outside_the_allowed_range_fails_validation():
    # Out-of-range values must fail validation so the LLM retry path fires.
    with pytest.raises(Exception):
        Penalty(reason="Made-up scale", points=40)


def test_breakdown_reports_penalties_as_a_negative_number():
    breakdown = build_breakdown(
        scores(penalties=[Penalty(reason="Thin wrapper", points=10)]), github(6)
    )
    assert breakdown == {
        "ai_project_depth": 30,
        "python_backend": 20,
        "cloud_fullstack": 10,
        "github": 6,
        "engineering_depth": 3,
        "penalties": -10,
    }


# --- GitHub activity thresholds ---------------------------------------------


@pytest.mark.parametrize(
    "days, expected",
    [(0, 5), (10, 5), (30, 5), (31, 4), (90, 4), (100, 3), (180, 3), (181, 2),
     (365, 2), (400, 1), (None, 0)],
)
def test_activity_thresholds(days, expected):
    assert score_activity(days) == expected


# --- GitHub repo thresholds -------------------------------------------------


@pytest.mark.parametrize(
    "relevant, total, expected",
    [(6, 10, 5), (5, 10, 5), (4, 10, 4), (3, 10, 4), (2, 10, 3), (1, 10, 2),
     (0, 10, 1), (0, 0, 0)],
)
def test_repo_thresholds(relevant, total, expected):
    assert score_repos(relevant, total) == expected


def test_python_repo_pushed_recently_is_relevant():
    repo = {
        "fork": False,
        "language": "Python",
        "pushed_at": (NOW - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    assert is_relevant_repo(repo, NOW)


def test_forks_and_stale_repos_do_not_count():
    recent = (NOW - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    stale = (NOW - timedelta(days=500)).strftime("%Y-%m-%dT%H:%M:%SZ")

    assert not is_relevant_repo(
        {"fork": True, "language": "Python", "pushed_at": recent}, NOW
    )
    assert not is_relevant_repo(
        {"fork": False, "language": "Python", "pushed_at": stale}, NOW
    )


def test_non_python_repo_counts_when_it_looks_like_ai_work():
    repo = {
        "fork": False,
        "language": "TypeScript",
        "name": "rag-chatbot",
        "description": "Retrieval chatbot",
        "pushed_at": (NOW - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    assert is_relevant_repo(repo, NOW)


def test_unrelated_repo_does_not_count():
    repo = {
        "fork": False,
        "language": "CSS",
        "name": "portfolio-site",
        "description": "Personal website",
        "pushed_at": (NOW - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    assert not is_relevant_repo(repo, NOW)


# --- Ranking ----------------------------------------------------------------


def candidate(name, total, ai=0, python=0) -> CandidateResult:
    return CandidateResult(
        candidate_name=name,
        source_file=f"{name}.pdf",
        eligible=True,
        total_score=total,
        score_breakdown={"ai_project_depth": ai, "python_backend": python},
    )


def test_ranking_sorts_by_score_descending():
    ranked = rank_candidates(
        [candidate("Low", 40), candidate("High", 90), candidate("Mid", 65)]
    )
    assert [c.candidate_name for c in ranked] == ["High", "Mid", "Low"]
    assert [c.rank for c in ranked] == [1, 2, 3]


def test_ties_break_on_ai_depth_then_python_then_name():
    ranked = rank_candidates(
        [
            candidate("Carol", 70, ai=20, python=25),
            candidate("Alice", 70, ai=30, python=20),
            candidate("Bob", 70, ai=20, python=30),
        ]
    )
    assert [c.candidate_name for c in ranked] == ["Alice", "Bob", "Carol"]


def test_ranking_is_deterministic_for_identical_scores():
    people = [candidate("Zoe", 50), candidate("Adam", 50)]
    first = [c.candidate_name for c in rank_candidates(list(people))]
    second = [c.candidate_name for c in rank_candidates(list(reversed(people)))]
    assert first == second == ["Adam", "Zoe"]


def test_eligible_but_unscored_candidates_sort_last_with_no_rank():
    unscored = CandidateResult(
        candidate_name="Unscored", source_file="x.pdf", eligible=True
    )
    ranked = rank_candidates([unscored, candidate("Scored", 10)])

    assert ranked[0].candidate_name == "Scored"
    assert ranked[0].rank == 1
    assert ranked[-1].candidate_name == "Unscored"
    assert ranked[-1].rank is None
