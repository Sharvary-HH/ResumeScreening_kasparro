"""CandidateProfile -> LLMScores. Eligible candidates only.

The model assigns sub-scores with evidence; all arithmetic happens in
screening/ranking.py. The model is a judge of substance, not a calculator.
"""
import json
from typing import List, Optional

from src import config
from src.llm.client import chat_json, schema_instructions
from src.models import CandidateProfile, LLMScores

SYSTEM_PROMPT = """You are a senior engineer screening resumes for an AI/Python \
intern role. Score the candidate on each category below using only the evidence \
in the profile.

SCORING CATEGORIES

ai_project_depth (0-{ai_max}) - the heaviest category. Reward real AI systems: \
agents, RAG and retrieval, tool/function calling, state management, \
orchestration, evaluation, and meaningful business logic around the model. \
Reward depth of implementation, not name-dropping.
  {ai_max}-32: production-grade agentic or RAG system with real complexity
  31-20: a solid working AI project with some depth
  19-10: a basic AI project or a thin LLM integration
  9-0: classical ML coursework only, or AI mentioned without a project

python_backend (0-{python_max}) - Python depth and backend engineering: Python, \
FastAPI/Django/Flask, async, PostgreSQL, Redis, API design. Evidence in projects \
and internships outweighs a keyword-only skills list.

cloud_fullstack (0-{cloud_max}) - GCP/AWS/Azure, Docker, CI/CD, real deployment. \
React/Next.js count as supporting signals when they are part of an end-to-end \
system the candidate actually shipped.

engineering_depth (0-{eng_max}) - testing, architecture, caching, queues, \
observability, concurrency, failure handling.

PENALTIES - a list of {{reason, points}} with points between {pen_min} and \
{pen_max}. Apply one when:
- an "AI project" is a thin wrapper around a single LLM or API call with no \
workflow, data processing, retrieval, state, backend logic, or evaluation
- projects are tutorial-style with no implementation detail or evidence of \
ownership
Return an empty list when neither applies. Do not invent penalties.

Do not award high marks merely because a framework name appears in a skills \
section. Cite what the candidate actually built.

Also return:
- project_summary: one or two sentences describing what they built, backed by \
evidence from the profile
- strengths: 2 to 4 short, specific items
- concerns: 0 to 4 short, specific items

Return only JSON matching this schema:
{schema}"""


def _profile_digest(profile: CandidateProfile) -> str:
    """Contact details are irrelevant to scoring and only add tokens."""
    payload = {
        "skills": profile.skills,
        "projects": [p.model_dump() for p in profile.projects],
        "experience": [e.model_dump() for e in profile.experience],
        "education": [e.model_dump() for e in profile.education],
        "has_github": bool(profile.github_url),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def score_profile(
    profile: CandidateProfile, errors: Optional[List[str]] = None
) -> Optional[LLMScores]:
    system = SYSTEM_PROMPT.format(
        ai_max=config.AI_MAX,
        python_max=config.PYTHON_MAX,
        cloud_max=config.CLOUD_MAX,
        eng_max=config.ENGINEERING_MAX,
        pen_min=config.PENALTY_MIN,
        pen_max=config.PENALTY_MAX,
        schema=schema_instructions(LLMScores),
    )

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                "Score this candidate profile.\n\n" + _profile_digest(profile)
            ),
        },
    ]

    return chat_json(messages, LLMScores, errors=errors)
