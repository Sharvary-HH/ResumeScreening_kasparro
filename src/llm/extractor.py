"""Resume text -> CandidateProfile.

The model is asked for neutral facts and nothing else. It is never asked whether
a candidate is eligible, qualified, or good - that decision belongs to
src/screening/eligibility.py.
"""
from typing import List, Optional

from src.llm.client import chat_json, schema_instructions
from src.models import CandidateProfile, ParsedResume
from src.parser.urls import find_github_url, find_linkedin_url

SYSTEM_PROMPT = """You extract structured data from resumes. Return only JSON \
matching the schema. Do not infer, judge, or evaluate. If a field is absent, use \
null or an empty list.

Rules:
- Copy what the resume says. Do not add skills or technologies the candidate did \
not mention.
- tech_stack should list the concrete technologies named for that project or role.
- description should be one or two sentences drawn from the resume's own wording.
- Return every project and every work/internship entry you find.

JSON schema:
{schema}"""

# Truncating keeps the request inside a sane token budget. The corpus tops out
# around 12k characters, so in practice this only trims the longest resumes.
MAX_RESUME_CHARS = 14000


def _user_prompt(text: str, urls: List[str]) -> str:
    body = text[:MAX_RESUME_CHARS]
    known_links = "\n".join(urls) if urls else "(none found)"
    return (
        "Extract the candidate profile from this resume.\n\n"
        f"RESUME TEXT:\n{body}\n\n"
        # Most GitHub URLs in this corpus exist only as PDF link annotations,
        # so the model cannot see them in the text above.
        f"KNOWN_LINKS (extracted from the file's hyperlinks):\n{known_links}\n"
    )


def extract_profile(
    parsed: ParsedResume, errors: Optional[List[str]] = None
) -> Optional[CandidateProfile]:
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(
                schema=schema_instructions(CandidateProfile)
            ),
        },
        {"role": "user", "content": _user_prompt(parsed.text, parsed.urls)},
    ]

    profile = chat_json(messages, CandidateProfile, errors=errors)
    if profile is None:
        return None

    # URLs are a parsing problem, not a judgment call. Whatever the model
    # returned, the ones we pulled out of the file itself win.
    github = find_github_url(parsed.urls)
    if github:
        profile.github_url = github
    linkedin = find_linkedin_url(parsed.urls)
    if linkedin:
        profile.linkedin_url = linkedin

    return profile
