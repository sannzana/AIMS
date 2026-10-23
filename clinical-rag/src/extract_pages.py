"""Extract page-level text and table metadata from protocol PDFs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import fitz
except ImportError:  # pragma: no cover - exercised in dependency-free installs
    fitz = None

try:
    import pdfplumber
except ImportError:  # pragma: no cover
    pdfplumber = None


def _table_text(tables: list[list[list[Any]]]) -> str:
    rows: list[str] = []
    for table in tables:
        for row in table:
            cells = [str(cell).strip() if cell is not None else "" for cell in row]
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def extract_pdf(pdf_path: Path, source_title: str | None = None) -> dict[str, Any]:
    """Extract one-based page records, including table text where available."""
    if fitz is None:
        raise RuntimeError("PyMuPDF is required to extract PDF files")
    pages: list[dict[str, Any]] = []
    with fitz.open(pdf_path) as document:
        plumber = pdfplumber.open(pdf_path) if pdfplumber is not None else None
        try:
            for index, page in enumerate(document):
                text = page.get_text("text")
                table_text = ""
                if plumber is not None and index < len(plumber.pages):
                    table_text = _table_text(plumber.pages[index].extract_tables())
                if table_text:
                    text = f"{text.rstrip()}\n{table_text}" if text.strip() else table_text
                review_flags = []
                if not text.strip():
                    review_flags.append("unreadable_or_empty_text")
                if table_text:
                    review_flags.append("table_extracted_review_required")
                pages.append(
                    {
                        "page": index + 1,
                        "text": text,
                        "has_table": bool(table_text),
                        "manual_review_flags": review_flags,
                    }
                )
        finally:
            if plumber is not None:
                plumber.close()
    return {
        "pdf": pdf_path.name,
        "title": source_title or pdf_path.stem,
        "pages": pages,
    }


def extract_pages(
    pdf_dir: Path,
    output_path: Path,
    source_titles: dict[str, str] | None = None,
    source_text_additions: dict[str, dict[str, str]] | None = None,
) -> None:
    documents = []
    source_titles = source_titles or {}
    source_text_additions = source_text_additions or {}
    if pdf_dir.exists():
        for path in sorted(pdf_dir.glob("*.pdf")):
            try:
                document = extract_pdf(path, source_titles.get(path.name))
                additions = source_text_additions.get(path.name, {})
                for page in document["pages"]:
                    addition = additions.get(str(page["page"]))
                    if addition:
                        page["text"] = f"{page['text'].rstrip()}\n{addition}"
                        page["manual_review_flags"].append(
                            "source_text_addition_manual_review"
                        )
                documents.append(document)
            except (OSError, RuntimeError) as exc:
                documents.append(
                    {
                        "pdf": path.name,
                        "title": source_titles.get(path.name, path.stem),
                        "pages": [],
                        "manual_review_flags": [f"extraction_failed: {exc}"],
                    }
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"documents": documents}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_page_config(config_path: Path) -> dict[str, Any]:
    return json.loads(config_path.read_text(encoding="utf-8"))


def assigned_pages(config: dict[str, Any]) -> list[int]:
    pages: list[int] = []
    for assignment in config.get("assignments", []):
        pages.extend(int(page) for page in assignment.get("pages", []))
        for start, end in assignment.get("ranges", []):
            pages.extend(range(int(start), int(end) + 1))
    return pages


def assigned_page_keys(config: dict[str, Any]) -> list[tuple[str, int]]:
    """Return unique page identities, scoped to each assigned PDF."""
    keys: list[tuple[str, int]] = []
    for assignment in config.get("assignments", []):
        pdf = str(assignment.get("pdf", assignment.get("protocol", "")))
        keys.extend((pdf, int(page)) for page in assignment.get("pages", []))
        for start, end in assignment.get("ranges", []):
            keys.extend((pdf, page) for page in range(int(start), int(end) + 1))
    return keys


def render_flagged_pages(
    pdf_dir: Path, pages_path: Path, output_dir: Path
) -> list[Path]:
    """Render pages carrying manual-review flags as PNGs."""
    pages_data = json.loads(pages_path.read_text(encoding="utf-8"))
    rendered: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    if fitz is None:
        raise RuntimeError("PyMuPDF is required to render PDF pages")
    for document in pages_data.get("documents", []):
        flagged = [
            page["page"]
            for page in document.get("pages", [])
            if page.get("manual_review_flags")
        ]
        if not flagged:
            continue
        pdf_path = pdf_dir / document["pdf"]
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found for rendering: {pdf_path}")
        with fitz.open(pdf_path) as source:
            for number in flagged:
                image = source[number - 1].get_pixmap(matrix=fitz.Matrix(2, 2))
                target = output_dir / f"{Path(document['pdf']).stem}_page_{number}.png"
                image.save(target)
                rendered.append(target)
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/pdfs"))
    parser.add_argument("--output", type=Path, default=Path("data/pages.json"))
    parser.add_argument("--config", type=Path, default=Path("data/pages_config.json"))
    parser.add_argument("--render-flagged", type=Path)
    args = parser.parse_args()
    if not args.pdf_dir.exists():
        print(f"ERROR: PDF directory not found: {args.pdf_dir}", file=sys.stderr)
        raise SystemExit(2)
    config = load_page_config(args.config) if args.config.exists() else {}
    source_titles = {
        str(assignment["pdf"]): str(assignment["source_title"])
        for assignment in config.get("assignments", [])
        if assignment.get("pdf") and assignment.get("source_title")
    }
    extract_pages(
        args.pdf_dir,
        args.output,
        source_titles,
        config.get("source_text_additions", {}),
    )
    if args.config.exists():
        pages = assigned_page_keys(config)
        if len(pages) != len(set(pages)):
            print("ERROR: assigned page ranges contain duplicate pages", file=sys.stderr)
            raise SystemExit(1)
        print(f"Assigned pages: {len(pages)}")
    if args.render_flagged is not None:
        rendered = render_flagged_pages(args.pdf_dir, args.output, args.render_flagged)
        print(f"Rendered {len(rendered)} flagged page(s)")
    print(f"Extracted {len(list(args.pdf_dir.glob('*.pdf')))} PDF(s) to {args.output}")


if __name__ == "__main__":
    main()
