"""Parsing, URL normalization, and the quality gate.

The PDF cases need real files: the behaviour that matters here - GitHub URLs
hiding in link annotations - only shows up in a real PDF.
"""
import pytest

from src.models import ParsedResume
from src.parser import parse_resume
from src.parser.quality import is_usable
from src.parser.urls import (
    extract_urls_from_text,
    find_github_url,
    github_username,
    merge_urls,
    normalize_url,
)
from tests.conftest import RESUMES

# The corpus is personal data and is not committed, so the tests that read it
# skip on a clean clone rather than fail. Everything else still runs.
needs_corpus = pytest.mark.skipif(
    not any(RESUMES.glob("*.pdf")), reason="resumes/ corpus not available"
)


# --- URL normalization ------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("https://github.com/octocat/", "https://github.com/octocat"),
        ("https://github.com/octocat.git", "https://github.com/octocat"),
        ("https://www.github.com/octocat", "https://github.com/octocat"),
        ("https://GitHub.COM/OctoCat", "https://github.com/OctoCat"),
        ("github.com/octocat", "https://github.com/octocat"),
        ("https://github.com/octocat?tab=repositories", "https://github.com/octocat"),
        ("https://github.com/octocat),", "https://github.com/octocat"),
        ("mailto:someone@example.com", ""),
    ],
)
def test_url_normalization(raw, expected):
    assert normalize_url(raw) == expected


def test_merge_urls_dedupes_across_sources():
    merged = merge_urls(
        ["https://github.com/octocat/"], ["github.com/octocat", "www.github.com/octocat"]
    )
    assert merged == ["https://github.com/octocat"]


def test_merge_urls_prefers_the_complete_form_of_a_wrapped_url():
    # A URL split across a line wrap yields a truncated username; the longer
    # form is the real one.
    merged = merge_urls(
        ["https://github.com/annishasaravan", "https://github.com/annishasaravanan"]
    )
    assert merged == ["https://github.com/annishasaravanan"]


def test_a_repo_path_is_not_treated_as_a_truncated_profile():
    merged = merge_urls(["https://github.com/octocat", "https://github.com/octocat/hello"])
    assert len(merged) == 2


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://github.com/octocat", "octocat"),
        ("https://github.com/octocat/hello-world", ""),  # a repo, not a profile
        ("https://github.com/login", ""),
        ("https://linkedin.com/in/octocat", ""),
    ],
)
def test_github_username_extraction(url, expected):
    assert github_username(url) == expected


def test_wrapped_url_is_rejoined_from_text():
    text = "Portfolio\nhttps://github.com/annishasaravan\nan\nSKILLS\nPython"
    assert "https://github.com/annishasaravanan" in merge_urls(
        extract_urls_from_text(text)
    )


def test_unrelated_next_line_is_not_glued_onto_a_url():
    text = "https://github.com/octocat\nEDUCATION"
    urls = merge_urls(extract_urls_from_text(text))
    assert "https://github.com/octocat" in urls
    assert not any("EDUCATION" in url for url in urls)


# --- PDF --------------------------------------------------------------------


@needs_corpus
def test_github_url_recovered_from_a_pdf_link_annotation():
    # candidate_07 hyperlinks the word "GitHub" - the URL is nowhere in the
    # visible text, so a text-only parser finds nothing.
    parsed = parse_resume(RESUMES / "candidate_07.pdf")

    assert parsed.ok

    url = find_github_url(parsed.urls)
    assert url.startswith("https://github.com/")

    # The proof that it came from an annotation: the username appears nowhere
    # in the extracted text.
    username = url.rsplit("/", 1)[-1]
    assert username.lower() not in parsed.text.lower()


@needs_corpus
def test_pdf_with_corrupt_fonts_still_parses():
    # MuPDF logs zlib/FreeType errors for this file; the text extracts fine and
    # it must not be treated as a failure.
    parsed = parse_resume(RESUMES / "candidate_07.pdf")

    assert parsed.ok
    assert len(parsed.text) > 1000
    assert is_usable(parsed) == (True, None)


@needs_corpus
def test_multi_column_resume_extracts_readable_text():
    parsed = parse_resume(RESUMES / "candidate_04.pdf")

    assert parsed.ok
    assert len(parsed.text) > 1000


# --- DOCX -------------------------------------------------------------------


def test_docx_parsing_returns_text_tables_and_hyperlinks(synthetic_docx):
    parsed = parse_resume(synthetic_docx)

    assert parsed.ok
    assert parsed.parser == "docx"
    assert "LangGraph" in parsed.text
    assert "Redis-backed job queues" in parsed.text  # from a table cell
    assert find_github_url(parsed.urls) == "https://github.com/priya-nair"


# --- Dispatch and quality gate ---------------------------------------------


def test_unknown_extension_returns_a_failed_result_instead_of_raising(tmp_path):
    path = tmp_path / "resume.rtf"
    path.write_text("not a supported format")

    parsed = parse_resume(path)

    assert not parsed.ok
    assert "Unsupported file type" in parsed.error


def test_missing_file_returns_a_failed_result(tmp_path):
    parsed = parse_resume(tmp_path / "nope.pdf")

    assert not parsed.ok
    assert parsed.error


def test_quality_gate_rejects_a_near_empty_blob():
    usable, reason = is_usable(
        ParsedResume(source_path="x.pdf", text="Name\nEmail\n", parser="pdf")
    )

    assert not usable
    assert "below minimum length" in reason


def test_quality_gate_rejects_symbol_soup():
    usable, reason = is_usable(
        ParsedResume(source_path="x.pdf", text="#@$%^&*" * 100, parser="pdf")
    )

    assert not usable
    assert "garbage" in reason


@needs_corpus
def test_quality_gate_accepts_a_real_resume():
    usable, reason = is_usable(parse_resume(RESUMES / "candidate_01.pdf"))

    assert usable
    assert reason is None
