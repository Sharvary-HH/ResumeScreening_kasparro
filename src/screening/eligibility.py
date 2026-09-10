"""Hard filtering. Deterministic, pure, and deliberately free of any LLM.

This module must never import from src.llm. Who is eligible is a business rule,
and business rules belong in code we can unit test without a network call.
"""
import re
from typing import List, Tuple

from src import config
from src.models import CandidateProfile, EligibilityResult


def _build_corpus(profile: CandidateProfile) -> str:
    """Everything the candidate claims, flattened and lowercased - so Python
    buried in a project's tech_stack counts as much as Python in the skills list."""
    parts: List[str] = list(profile.skills)

    for project in profile.projects:
        parts.append(project.name)
        parts.append(project.description)
        parts.extend(project.tech_stack)
        if project.role:
            parts.append(project.role)

    for job in profile.experience:
        parts.append(job.title)
        parts.append(job.description)
        parts.extend(job.tech_stack)

    return " | ".join(part for part in parts if part).lower()


def _find_keywords(corpus: str, keywords: List[str]) -> List[str]:
    """Word-boundary matched: a substring search for "ai" hits "email"."""
    hits = []
    for keyword in keywords:
        pattern = r"\b" + re.escape(keyword)
        # Only require a closing boundary for whole words - prefixes like
        # "fine-tun" are meant to match "fine-tuned"/"fine-tuning".
        if not keyword.endswith("-") and keyword.isalnum():
            pattern += r"\b"
        if re.search(pattern, corpus):
            hits.append(keyword)
    return hits


def _match_skills(profile: CandidateProfile) -> List[str]:
    """Intersection with the curated list, reported in config's display casing so
    output is consistent however the resume wrote it."""
    claimed = {skill.strip().lower() for skill in profile.skills if skill.strip()}
    matched = []
    for skill in config.RELEVANT_SKILLS:
        needle = skill.lower()
        if any(needle == c or needle in c.split(", ") for c in claimed):
            matched.append(skill)
    return matched


def check_eligibility(profile: CandidateProfile) -> EligibilityResult:
    """Apply the two hard requirements: a Python stack and AI project evidence."""
    corpus = _build_corpus(profile)

    python_evidence = _find_keywords(corpus, config.PYTHON_KEYWORDS)
    strong_ai = _find_keywords(corpus, config.AI_KEYWORDS_STRONG)
    weak_ai = _find_keywords(corpus, config.AI_KEYWORDS_WEAK)
    ai_evidence = strong_ai + [k for k in weak_ai if k not in strong_ai]

    rejection_reasons: List[str] = []
    if not python_evidence:
        rejection_reasons.append(config.REJECT_NO_PYTHON)
    if not ai_evidence:
        rejection_reasons.append(config.REJECT_NO_AI)

    return EligibilityResult(
        eligible=not rejection_reasons,
        rejection_reasons=rejection_reasons,
        matched_skills=_match_skills(profile),
        python_evidence=python_evidence,
        ai_evidence=ai_evidence,
        # Classical ML alone still passes; the 40-point AI category demotes it.
        weak_ai_only=bool(ai_evidence) and not strong_ai,
    )
