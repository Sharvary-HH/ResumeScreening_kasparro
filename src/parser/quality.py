"""Text quality gate.

Cheap guard that runs before we spend an LLM call on a file. A resume that
fails here is recorded as a failed file and the batch carries on.
"""
from typing import Optional, Tuple

from src import config
from src.models import ParsedResume


def is_usable(parsed: ParsedResume) -> Tuple[bool, Optional[str]]:
    """Return (usable, reason_if_not)."""
    if parsed.error:
        return False, parsed.error

    text = (parsed.text or "").strip()
    if not text:
        return False, "No text could be extracted from the file"

    if len(text) < config.MIN_TEXT_CHARS:
        return False, (
            f"Extracted text below minimum length ({len(text)} chars, "
            f"need {config.MIN_TEXT_CHARS})"
        )

    non_alnum = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    ratio = non_alnum / len(text)
    if ratio > config.MAX_NON_ALNUM_RATIO:
        return False, (
            f"Extracted text looks like garbage ({ratio:.0%} non-alphanumeric)"
        )

    return True, None
