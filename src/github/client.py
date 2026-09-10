"""GitHub enrichment, worth up to 10 points.

Scoring is deterministic against fixed thresholds, so the same profile always
yields the same points and the boundaries are unit-testable. Every failure path
returns total=0 with a descriptive status; nothing here raises.
"""
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from src import config
from src.models import GitHubEnrichment
from src.parser.urls import github_username

_ISO_FORMATS = ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z")


# --- Deterministic scoring --------------------------------------------------


def score_activity(days_since_last_push: Optional[int]) -> int:
    """0-5 from how recently the user pushed public code."""
    if days_since_last_push is None:
        return config.GITHUB_ACTIVITY_NO_EVENTS_SCORE
    for max_days, points in config.GITHUB_ACTIVITY_THRESHOLDS:
        if days_since_last_push <= max_days:
            return points
    return config.GITHUB_ACTIVITY_STALE_SCORE


def score_repos(relevant_count: int, total_repos: int) -> int:
    """0-5 from the number of relevant, recently maintained repositories."""
    for minimum, points in config.GITHUB_REPO_THRESHOLDS:
        if relevant_count >= minimum:
            return points
    if total_repos > 0:
        return config.GITHUB_REPO_HAS_REPOS_SCORE
    return config.GITHUB_REPO_NO_REPOS_SCORE


def is_relevant_repo(repo: Dict[str, Any], now: datetime) -> bool:
    """A non-fork repo, pushed within the last year, that is either Python or
    named/described as AI work."""
    if repo.get("fork"):
        return False

    updated = _parse_time(repo.get("pushed_at") or repo.get("updated_at"))
    if updated is None or (now - updated).days > config.GITHUB_REPO_FRESH_DAYS:
        return False

    if (repo.get("language") or "").lower() == "python":
        return True

    haystack = " ".join(
        [
            repo.get("name") or "",
            repo.get("description") or "",
            " ".join(repo.get("topics") or []),
        ]
    ).lower()
    return any(
        keyword in haystack.replace("_", "-")
        for keyword in config.GITHUB_AI_REPO_KEYWORDS
    )


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    for fmt in _ISO_FORMATS:
        try:
            parsed = datetime.strptime(value, fmt)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            continue
    return None


def _days_since_last_push(events: List[Dict[str, Any]], now: datetime) -> Optional[int]:
    pushes = [
        _parse_time(event.get("created_at"))
        for event in events
        if event.get("type") == "PushEvent"
    ]
    timestamps = [t for t in pushes if t is not None]
    if not timestamps:
        return None
    return max(0, (now - max(timestamps)).days)


# --- HTTP + cache -----------------------------------------------------------


def _headers() -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if config.GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {config.GITHUB_TOKEN}"
    return headers


def _cache_path(username: str):
    return config.GITHUB_CACHE_DIR / f"{username.lower()}.json"


def _cache_read(username: str) -> Optional[Dict[str, Any]]:
    path = _cache_path(username)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# Only these fields feed the score. Caching the raw API responses instead costs
# ~140x the disk for nothing, and the cache is committed to the repo.
_REPO_FIELDS = ("name", "description", "language", "fork", "pushed_at", "updated_at",
                "topics")
_EVENT_FIELDS = ("type", "created_at")


def _slim(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": payload.get("status", "ok"),
        "user": {"public_repos": (payload.get("user") or {}).get("public_repos")},
        "repos": [
            {key: repo.get(key) for key in _REPO_FIELDS}
            for repo in payload.get("repos") or []
        ],
        "events": [
            {key: event.get(key) for key in _EVENT_FIELDS}
            for event in payload.get("events") or []
            if event.get("type") == "PushEvent"
        ],
    }


def _cache_write(username: str, payload: Dict[str, Any]) -> None:
    try:
        config.GITHUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(username).write_text(
            json.dumps(_slim(payload), indent=2), encoding="utf-8"
        )
    except OSError:
        pass


def _fetch(username: str, use_cache: bool) -> Tuple[str, Dict[str, Any]]:
    """Returns (status, payload): ok | not_found | rate_limited | error."""
    if use_cache:
        cached = _cache_read(username)
        if cached is not None:
            return cached.get("status", "ok"), cached

    endpoints = {
        "user": f"{config.GITHUB_API}/users/{username}",
        "repos": f"{config.GITHUB_API}/users/{username}/repos"
        "?sort=updated&per_page=100",
        "events": f"{config.GITHUB_API}/users/{username}/events/public?per_page=100",
    }

    payload: Dict[str, Any] = {"status": "ok"}
    try:
        with httpx.Client(
            headers=_headers(), timeout=config.GITHUB_TIMEOUT, follow_redirects=True
        ) as client:
            for name, url in endpoints.items():
                response = client.get(url)

                if response.status_code == 404:
                    return "not_found", {"status": "not_found"}
                if response.status_code in (403, 429):
                    return "rate_limited", {"status": "rate_limited"}
                if response.status_code != 200:
                    return "error", {"status": "error"}

                payload[name] = response.json()
    except httpx.HTTPError:
        return "error", {"status": "error"}

    if use_cache:
        _cache_write(username, payload)
    return "ok", payload


# --- Public API -------------------------------------------------------------


_STATUS_SUMMARY = {
    "missing": "No GitHub profile link found on the resume.",
    "not_found": "GitHub profile link is dead (user not found).",
    "rate_limited": "GitHub rate limit reached - set GITHUB_TOKEN to enrich.",
    "error": "GitHub lookup failed (network or API error).",
}


def enrich(github_url: Optional[str], *, use_cache: bool = True) -> GitHubEnrichment:
    username = github_username(github_url or "")
    if not username:
        return GitHubEnrichment(status="missing", summary=_STATUS_SUMMARY["missing"])

    status, payload = _fetch(username, use_cache)
    if status != "ok":
        return GitHubEnrichment(
            username=username, status=status, summary=_STATUS_SUMMARY[status]
        )

    now = datetime.now(timezone.utc)
    repos = payload.get("repos") or []
    events = payload.get("events") or []
    public_repos = (payload.get("user") or {}).get("public_repos")

    days = _days_since_last_push(events, now)
    relevant = [repo for repo in repos if is_relevant_repo(repo, now)]

    activity = score_activity(days)
    repo_points = score_repos(len(relevant), len(repos))

    return GitHubEnrichment(
        username=username,
        status="ok",
        activity_score=activity,
        repo_score=repo_points,
        total=activity + repo_points,
        summary=_summarize(username, days, len(relevant), len(repos)),
        public_repos=public_repos,
        last_activity_days=days,
    )


def _summarize(
    username: str, days: Optional[int], relevant: int, total_repos: int
) -> str:
    if days is None:
        recency = "no public push activity in the last 90 days"
    elif days <= 30:
        recency = f"pushed {days} day(s) ago"
    else:
        recency = f"last public push {days} days ago"

    return (
        f"@{username}: {recency}; {relevant} relevant "
        f"(Python/AI) of {total_repos} public repos."
    )
