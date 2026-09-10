"""Resume parsing: dispatch on file extension, never raise."""
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

from src import config
from src.models import ParsedResume
from src.parser.docx_parser import parse_docx
from src.parser.pdf_parser import parse_pdf
from src.parser.quality import is_usable
from src.parser.urls import extract_urls_from_text, merge_urls

__all__ = ["parse_resume", "is_usable", "collect_resumes", "file_sha256"]


def parse_resume(path) -> ParsedResume:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix == ".docx":
        return parse_docx(path)
    if suffix == ".txt":
        return _parse_txt(path)

    return ParsedResume(
        source_path=str(path),
        parser="unknown",
        error=f"Unsupported file type '{suffix or path.name}'",
    )


def _parse_txt(path: Path) -> ParsedResume:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return ParsedResume(
            source_path=str(path), parser="txt", error=f"Text read failed: {exc}"
        )

    return ParsedResume(
        source_path=str(path),
        text=text,
        urls=merge_urls(extract_urls_from_text(text)),
        page_count=1,
        parser="txt",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_resumes(input_dir) -> Tuple[List[Path], List[Path]]:
    """Returns (files_to_process, duplicates). Identical bytes mean the same
    candidate submitted twice: process the first, report the rest."""
    input_dir = Path(input_dir)
    candidates = sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in config.SUPPORTED_EXTENSIONS
    )

    seen: Dict[str, Path] = {}
    unique: List[Path] = []
    duplicates: List[Path] = []

    for path in candidates:
        try:
            digest = file_sha256(path)
        except OSError:
            # Let parse_resume report the error rather than swallowing it here.
            unique.append(path)
            continue

        if digest in seen:
            duplicates.append(path)
        else:
            seen[digest] = path
            unique.append(path)

    return unique, duplicates
