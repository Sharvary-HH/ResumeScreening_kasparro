"""Schema behaviour that the LLM path depends on."""
import pytest
from pydantic import ValidationError

from src.models import CandidateProfile, Experience, LLMScores


def test_nulls_are_absorbed_into_blanks_and_empty_lists():
    # The extraction prompt tells the model to use null for anything absent, so
    # nulls in these fields are expected output rather than a fault.
    job = Experience.model_validate(
        {"company": "Acme", "title": None, "description": None, "tech_stack": None}
    )

    assert job.title == ""
    assert job.description == ""
    assert job.tech_stack == []


def test_profile_survives_a_sparse_response():
    profile = CandidateProfile.model_validate(
        {"candidate_name": "Asha Rao", "skills": None, "projects": []}
    )

    assert profile.skills == []
    assert profile.email is None


@pytest.mark.parametrize(
    "field, value",
    [
        ("ai_project_depth", 41),
        ("python_backend", 31),
        ("cloud_fullstack", 16),
        ("engineering_depth", 6),
        ("ai_project_depth", -1),
    ],
)
def test_out_of_range_subscores_fail_validation(field, value):
    # These bounds are what sends a bad response back through the retry path,
    # so they need to actually bite.
    payload = {
        "ai_project_depth": 10,
        "python_backend": 10,
        "cloud_fullstack": 5,
        "engineering_depth": 2,
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        LLMScores.model_validate(payload)


def test_scores_at_the_upper_bound_are_accepted():
    scores = LLMScores.model_validate(
        {
            "ai_project_depth": 40,
            "python_backend": 30,
            "cloud_fullstack": 15,
            "engineering_depth": 5,
        }
    )

    assert scores.ai_project_depth == 40
    assert scores.penalties == []
