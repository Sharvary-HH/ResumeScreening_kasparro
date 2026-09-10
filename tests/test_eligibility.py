"""Eligibility is the highest-value thing to test: it decides who gets rejected,
and it does so without any network call, so every case here is cheap and exact.
"""
from src import config
from src.models import CandidateProfile, Experience, Project
from src.screening.eligibility import check_eligibility


def profile(**kwargs) -> CandidateProfile:
    return CandidateProfile(candidate_name="Test Candidate", **kwargs)


def test_python_plus_agentic_project_is_eligible():
    result = check_eligibility(
        profile(
            skills=["Python", "FastAPI"],
            projects=[
                Project(
                    name="Doc assistant",
                    description="RAG pipeline over internal docs",
                    tech_stack=["LangGraph", "FAISS"],
                )
            ],
        )
    )

    assert result.eligible
    assert result.rejection_reasons == []
    assert "python" in result.python_evidence
    assert "rag" in result.ai_evidence
    assert not result.weak_ai_only


def test_java_react_stack_is_rejected_for_no_python():
    result = check_eligibility(
        profile(
            skills=["Java", "React", "Spring Boot"],
            projects=[
                Project(
                    name="Inventory portal",
                    description="CRUD app for warehouse stock",
                    tech_stack=["Spring Boot", "MySQL"],
                )
            ],
        )
    )

    assert not result.eligible
    assert config.REJECT_NO_PYTHON in result.rejection_reasons


def test_python_without_any_ai_is_rejected():
    result = check_eligibility(
        profile(
            skills=["Python", "Django", "PostgreSQL"],
            projects=[
                Project(
                    name="Blog platform",
                    description="Django site with user auth and comments",
                    tech_stack=["Django", "PostgreSQL"],
                )
            ],
        )
    )

    assert not result.eligible
    assert config.REJECT_NO_AI in result.rejection_reasons
    assert config.REJECT_NO_PYTHON not in result.rejection_reasons


def test_langchain_without_python_is_rejected():
    # LangChain has a JS SDK, so an AI keyword alone does not imply Python.
    result = check_eligibility(
        profile(
            skills=["JavaScript", "TypeScript"],
            projects=[
                Project(
                    name="Chat widget",
                    description="LangChain agent in a Next.js app",
                    tech_stack=["LangChain", "Next.js"],
                )
            ],
        )
    )

    assert not result.eligible
    assert config.REJECT_NO_PYTHON in result.rejection_reasons


def test_supporting_javascript_skills_do_not_cause_rejection():
    # Regression guard: the filter screens for what is present, never against
    # extra frontend skills.
    result = check_eligibility(
        profile(
            skills=["Python", "React", "Next.js", "TypeScript"],
            projects=[
                Project(
                    name="Research copilot",
                    description="RAG over PDFs with a Next.js frontend",
                    tech_stack=["Python", "FastAPI", "React"],
                )
            ],
        )
    )

    assert result.eligible
    assert "React" in result.matched_skills
    assert "Next.js" in result.matched_skills


def test_python_found_only_in_project_tech_stack():
    result = check_eligibility(
        profile(
            skills=["Communication", "Teamwork"],
            projects=[
                Project(
                    name="Support agent",
                    description="Multi-agent triage bot",
                    tech_stack=["Python", "CrewAI"],
                )
            ],
        )
    )

    assert result.eligible
    assert "python" in result.python_evidence


def test_python_evidence_can_come_from_a_framework_alone():
    result = check_eligibility(
        profile(
            skills=["Django"],
            experience=[
                Experience(
                    company="Acme",
                    title="Intern",
                    description="Built an LLM summarizer",
                    tech_stack=["Django"],
                )
            ],
        )
    )

    assert result.eligible
    assert "django" in result.python_evidence


def test_classical_ml_only_passes_but_is_flagged_weak():
    result = check_eligibility(
        profile(
            skills=["Python", "TensorFlow"],
            projects=[
                Project(
                    name="Digit classifier",
                    description="CNN trained on MNIST",
                    tech_stack=["TensorFlow", "NumPy"],
                )
            ],
        )
    )

    assert result.eligible
    assert result.weak_ai_only
    assert "tensorflow" in result.ai_evidence


def test_word_boundary_prevents_substring_false_positives():
    # "email" must not register as AI evidence, "available" must not either.
    result = check_eligibility(
        profile(
            skills=["Java"],
            projects=[
                Project(
                    name="Mailer",
                    description="Available email templating service",
                    tech_stack=["Java"],
                )
            ],
        )
    )

    assert result.ai_evidence == []
    assert not result.eligible


def test_empty_profile_is_rejected_for_both_reasons():
    result = check_eligibility(profile())

    assert not result.eligible
    assert result.rejection_reasons == [config.REJECT_NO_PYTHON, config.REJECT_NO_AI]
