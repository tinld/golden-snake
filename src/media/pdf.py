"""PDF text extraction helpers.

The functions here favor predictable, clean text output and work page-by-page
to cope with large documents.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional

import fitz

__all__ = [
      "PDFReadingTransformer",
      "read_pdf_pages",
      "read_pdf_text",
      "read_pdf_page_records",
]


class PDFReadingTransformer:
      """Read PDF text with optional cleaning and page limits."""

      def __init__(self, pdf_path: str | Path, max_pages: Optional[int] = None):
            self.pdf_path = self._normalize_path(pdf_path)
            self.max_pages = max_pages

      @staticmethod
      def _clean_text(text: str) -> str:
            """Lightly clean extracted text for downstream NLP use."""

            text = text.replace("\u00a0", " ").replace("\ufeff", "")
            text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
            text = re.sub(r"[\t ]+", " ", text)
            text = re.sub(r"\s*\n\s*", "\n", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            return text.strip()

      @staticmethod
      def _normalize_path(pdf_path: str | Path) -> Path:
            path = Path(pdf_path).expanduser().resolve()
            if not path.exists():
                  raise FileNotFoundError(f"PDF not found: {path}")
            if path.suffix.lower() != ".pdf":
                  raise ValueError(f"Expected a PDF file, got: {path.suffix}")
            return path

      @staticmethod
      def _selected_pages(total_pages: int, max_pages: Optional[int]) -> Iterable[int]:
            if max_pages is None:
                  return range(total_pages)
            if max_pages <= 0:
                  raise ValueError("max_pages must be positive when provided")
            return range(min(total_pages, max_pages))

      def read_pdf_pages(
            self,
            pdf_path: Optional[str | Path] = None,
            *,
            max_pages: Optional[int] = None,
            clean: bool = True,
      ) -> List[str]:
            """Extract text from each page of a PDF."""

            path = self._normalize_path(pdf_path or self.pdf_path)
            limit = max_pages if max_pages is not None else self.max_pages
            page_texts: List[str] = []

            with fitz.open(path) as doc:
                  for page_index in self._selected_pages(doc.page_count, limit):
                        raw_text = doc.load_page(page_index).get_text("text")
                        page_texts.append(self._clean_text(raw_text) if clean else raw_text)

            return page_texts

      def read_pdf_page_records(
            self,
            pdf_path: Optional[str | Path] = None,
            *,
            max_pages: Optional[int] = None,
            clean: bool = True,
      ) -> List[dict]:
            """Return per-page records with page numbers for database storage."""

            pages = self.read_pdf_pages(pdf_path=pdf_path, max_pages=max_pages, clean=clean)
            return [
                  {"page": page_number, "text": text}
                  for page_number, text in enumerate(pages, start=1)
            ]

      def read_pdf_text(
            self,
            pdf_path: Optional[str | Path] = None,
            *,
            max_pages: Optional[int] = None,
            clean: bool = True,
            join_with: str = "\n\n",
      ) -> str:
            """Extract full text from a PDF, optionally limiting page count."""

            pages = self.read_pdf_pages(pdf_path=pdf_path, max_pages=max_pages, clean=clean)
            return join_with.join(pages)


def read_pdf_pages(
      pdf_path: str | Path,
      *,
      max_pages: Optional[int] = None,
      clean: bool = True,
) -> List[str]:
      """Functional wrapper around PDFReadingTransformer.read_pdf_pages."""

      return PDFReadingTransformer(pdf_path, max_pages=max_pages).read_pdf_pages(
            clean=clean
      )


def read_pdf_page_records(
      pdf_path: str | Path,
      *,
      max_pages: Optional[int] = None,
      clean: bool = True,
) -> List[dict]:
      """Functional wrapper returning page-numbered records."""

      return PDFReadingTransformer(pdf_path, max_pages=max_pages).read_pdf_page_records(
            clean=clean
      )


def read_pdf_text(
      pdf_path: str | Path,
      *,
      max_pages: Optional[int] = None,
      clean: bool = True,
      join_with: str = "\n\n",
) -> str:
      """Functional wrapper around PDFReadingTransformer.read_pdf_text."""

      return PDFReadingTransformer(pdf_path, max_pages=max_pages).read_pdf_text(
            clean=clean, join_with=join_with
      )
