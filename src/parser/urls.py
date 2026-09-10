"""URL collection and normalization, shared by the PDF and DOCX parsers."""
import re
from typing import Iterable, List

# Broad sweep first, then the two shapes we specifically care about in case the
# resume printed them without a scheme.
URL_PATTERNS = [
    re.compile(r"https?://\S+", re.IGNORECASE),
    re.compile(r"(?<!//)\bgithub\.com/[\w.-]+(?:/[\w.-]+)?", re.IGNORECASE),
    re.compile(r"(?<!//)\blinkedin\.com/in/[\w.-]+", re.IGNORECASE),
]

_TRAILING_JUNK = ".,;:)]}>\"'|*"


def normalize_url(url: str) -> str:
    """Canonical form so the same profile from text and from a link annotation
    collapses to one entry."""
    url = (url or "").strip().strip(_TRAILING_JUNK)
    if not url:
        return ""

    # mailto:/tel: targets are contact details, not links we score on.
    if re.match(r"^(mailto|tel|fax):", url, flags=re.IGNORECASE):
        return ""

    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        url = "https://" + url

    # Only the host gets lowercased; GitHub paths are case-sensitive.
    match = re.match(r"^(https?)://([^/?#]+)([^?#]*)", url, flags=re.IGNORECASE)
    if not match:
        return url
    scheme, host, path = match.groups()

    host = host.lower().lstrip(".")
    if host.startswith("www."):
        host = host[4:]

    path = path.rstrip("/")
    if path.lower().endswith(".git"):
        path = path[: -len(".git")]
    path = path.rstrip(_TRAILING_JUNK)

    return f"{scheme.lower()}://{host}{path}"


def extract_urls_from_text(text: str) -> List[str]:
    """Regex sweep over the visible text. A fallback - link annotations are the
    reliable source."""
    if not text:
        return []

    found: List[str] = []
    for haystack in (text, _repair_wrapped_urls(text)):
        for pattern in URL_PATTERNS:
            found.extend(pattern.findall(haystack))
    return found


def _repair_wrapped_urls(text: str) -> str:
    """Re-join URLs broken by a line wrap.

    One resume splits a github.com/<user> across two lines, which otherwise
    yields a truncated username. At most one continuation line is pulled up;
    chaining further would glue unrelated content together.
    """
    lines = text.split("\n")
    repaired = []
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        continuation = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if (
            re.search(r"https?://\S+$|\b(?:github|linkedin)\.com/\S*$", line, re.I)
            and re.match(r"^[\w-]+$", continuation)
            and len(continuation) <= 24
            # An all-caps line is a section heading ("EDUCATION"), not the tail
            # of a wrapped URL.
            and continuation != continuation.upper()
        ):
            line = line.rstrip("-") + continuation
            index += 1
        repaired.append(line)
        index += 1
    return "\n".join(repaired)


def merge_urls(*sources: Iterable[str]) -> List[str]:
    """Normalize and de-duplicate, keeping first-seen order. Where one URL just
    extends another's last segment, the longer one wins - the short form is a
    line-wrap truncation."""
    seen: List[str] = []
    for source in sources:
        for raw in source or []:
            url = normalize_url(raw)
            if url and url not in seen:
                seen.append(url)

    result: List[str] = []
    for url in seen:
        superseded = any(
            other != url and other.startswith(url) and _same_path_depth(url, other)
            for other in seen
        )
        if not superseded:
            result.append(url)
    return result


def _same_path_depth(shorter: str, longer: str) -> bool:
    """True when `longer` extends the last segment rather than adding a new one -
    a truncated username, not a repo path."""
    return "/" not in longer[len(shorter) :]


def github_username(url: str) -> str:
    """Username for a profile URL, or '' for repo paths: github.com/foo -> 'foo',
    but github.com/foo/bar is a repository."""
    match = re.match(
        r"^https?://github\.com/([\w-]+)/?$", normalize_url(url), flags=re.IGNORECASE
    )
    if not match:
        return ""

    username = match.group(1)
    if username.lower() in {"login", "signup", "about", "features", "explore", "topics"}:
        return ""
    return username


def find_github_url(urls: List[str]) -> str:
    """First URL in the list that is a GitHub profile."""
    for url in urls:
        if github_username(url):
            return normalize_url(url)
    return ""


def find_linkedin_url(urls: List[str]) -> str:
    for url in urls:
        if re.match(r"^https?://linkedin\.com/in/", url, flags=re.IGNORECASE):
            return url
    return ""
