"""PDF text + link extraction with PyMuPDF.

The important part here is get_links(). Most resumes in this corpus hyperlink
the word "GitHub" instead of printing the URL, so a text-only regex sweep misses
the majority of the profiles. See README > Design Decisions.
"""
from pathlib import Path

import pymupdf

from src.models import ParsedResume
from src.parser.urls import extract_urls_from_text, merge_urls

# One resume in the corpus has corrupt embedded fonts and makes MuPDF write
# zlib/FreeType warnings straight to C-level stderr. Text still extracts fine,
# so silence the noise rather than treating it as a failure.
pymupdf.TOOLS.mupdf_display_errors(False)


def parse_pdf(path: Path) -> ParsedResume:
    path = Path(path)
    try:
        with pymupdf.open(path) as doc:
            pages = []
            link_uris = []
            for page in doc:
                pages.append(page.get_text())
                for link in page.get_links():
                    uri = link.get("uri")
                    if uri:
                        link_uris.append(uri)
            page_count = doc.page_count
    except Exception as exc:  # noqa: BLE001 - a bad file must not end the batch
        return ParsedResume(
            source_path=str(path),
            parser="pdf",
            error=f"PDF read failed: {exc}",
        )

    text = "\n".join(pages)
    return ParsedResume(
        source_path=str(path),
        text=text,
        urls=merge_urls(link_uris, extract_urls_from_text(text)),
        page_count=page_count,
        parser="pdf",
    )
