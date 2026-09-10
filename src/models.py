"""Pydantic schemas for everything that crosses a module boundary.

Targets Python 3.10, where Pydantic v2 evaluates annotations at runtime and
`str | None` raises TypeError - hence Optional[...] / List[...] throughout.
"""
from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, BeforeValidator, Field

from src import config


def _none_to_blank(value):
    return "" if value is None else value


def _none_to_empty_list(value):
    return [] if value is None else value


# The extraction prompt tells the model to use null for anything absent, so a
# null here is expected output, not a fault. Coercing beats spending a retry.
Text = Annotated[str, BeforeValidator(_none_to_blank)]
StrList = Annotated[List[str], BeforeValidator(_none_to_empty_list)]


# --- What the LLM extracts from a resume (neutral facts only) ---------------


class Project(BaseModel):
    name: Text = ""
    description: Text = ""
    tech_stack: StrList = Field(default_factory=list)
    role: Optional[str] = None


class Experience(BaseModel):
    company: Text = ""
    title: Text = ""
    description: Text = ""
    tech_stack: StrList = Field(default_factory=list)
    duration: Optional[str] = None


class Education(BaseModel):
    institution: Text = ""
    degree: Text = ""
    year: Optional[str] = None


class CandidateProfile(BaseModel):
    candidate_name: Text = ""
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: StrList = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    github_url: Optional[str] = None
    linkedin_url: Optional[str] = None


# --- Parsing ----------------------------------------------------------------


class ParsedResume(BaseModel):
    source_path: str
    text: str = ""
    urls: List[str] = Field(default_factory=list)
    page_count: int = 0
    parser: str = ""
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# --- Deterministic screening ------------------------------------------------


class EligibilityResult(BaseModel):
    eligible: bool
    rejection_reasons: List[str] = Field(default_factory=list)
    matched_skills: List[str] = Field(default_factory=list)
    python_evidence: List[str] = Field(default_factory=list)
    ai_evidence: List[str] = Field(default_factory=list)
    # True when the only AI signal is classical ML rather than LLM/agentic work.
    weak_ai_only: bool = False


# --- What the LLM scores ----------------------------------------------------


class Penalty(BaseModel):
    reason: Text
    points: int = Field(ge=config.PENALTY_MIN, le=config.PENALTY_MAX)


class LLMScores(BaseModel):
    """Sub-scores only; Python does the arithmetic. The ge/le bounds are
    load-bearing - an out-of-range score fails validation and triggers a retry."""

    ai_project_depth: int = Field(ge=0, le=config.AI_MAX)
    python_backend: int = Field(ge=0, le=config.PYTHON_MAX)
    cloud_fullstack: int = Field(ge=0, le=config.CLOUD_MAX)
    engineering_depth: int = Field(ge=0, le=config.ENGINEERING_MAX)
    penalties: List[Penalty] = Field(default_factory=list)
    project_summary: Text = ""
    strengths: StrList = Field(default_factory=list)
    concerns: StrList = Field(default_factory=list)


# --- GitHub -----------------------------------------------------------------


class GitHubEnrichment(BaseModel):
    username: Optional[str] = None
    # ok | missing | not_found | rate_limited | error
    status: str = "missing"
    activity_score: int = Field(default=0, ge=0, le=5)
    repo_score: int = Field(default=0, ge=0, le=5)
    total: int = Field(default=0, ge=0, le=config.GITHUB_MAX)
    summary: str = ""
    public_repos: Optional[int] = None
    last_activity_days: Optional[int] = None


# --- Final per-candidate output --------------------------------------------


class CandidateResult(BaseModel):
    rank: Optional[int] = None
    candidate_name: str = ""
    source_file: str = ""
    eligible: bool = False
    total_score: Optional[int] = None
    score_breakdown: Optional[Dict[str, Any]] = None
    matched_skills: List[str] = Field(default_factory=list)
    project_summary: str = ""
    github_summary: str = ""
    strengths: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)
    rejection_reasons: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
