from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import fitz

from .config import DEFAULT_PAPERS_DIR


ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
DEFAULT_QUERY = 'all:"large language model" OR all:"LLM"'


@dataclass
class PaperRecord:
    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    updated: str
    pdf_url: str
    abs_url: str
    pdf_path: str
    text_path: str
    page_count: int
    text_chars: int


def _clean_arxiv_id(raw_id: str) -> str:
    value = raw_id.rstrip("/").split("/")[-1]
    value = re.sub(r"v\d+$", "", value)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _text(node: ET.Element, name: str) -> str:
    found = node.find(f"{ATOM_NS}{name}")
    if found is None or found.text is None:
        return ""
    return " ".join(found.text.split())


def search_arxiv(query: str, max_results: int) -> list[dict]:
    params = {
        "search_query": query,
        "start": "0",
        "max_results": str(max_results),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "PaperMem-eval/0.1 (mailto:papermem-eval@example.com)",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
    root = ET.fromstring(body)
    entries = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        paper_id = _clean_arxiv_id(_text(entry, "id"))
        pdf_url = ""
        abs_url = _text(entry, "id")
        for link in entry.findall(f"{ATOM_NS}link"):
            attrs = dict(link.attrib)
            if attrs.get("title") == "pdf" or attrs.get("type") == "application/pdf":
                pdf_url = attrs.get("href", "")
            elif attrs.get("rel") == "alternate":
                abs_url = attrs.get("href", abs_url)
        if not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        entries.append(
            {
                "paper_id": paper_id,
                "title": _text(entry, "title"),
                "authors": [
                    _text(author, "name")
                    for author in entry.findall(f"{ATOM_NS}author")
                    if _text(author, "name")
                ],
                "abstract": _text(entry, "summary"),
                "published": _text(entry, "published"),
                "updated": _text(entry, "updated"),
                "pdf_url": pdf_url,
                "abs_url": abs_url,
                "primary_category": (
                    entry.find(f"{ARXIV_NS}primary_category").attrib.get("term", "")
                    if entry.find(f"{ARXIV_NS}primary_category") is not None
                    else ""
                ),
            }
        )
    return entries


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 1024:
        return
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "PaperMem-eval/0.1 (mailto:papermem-eval@example.com)",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    output_path.write_bytes(data)


def extract_pdf_text(pdf_path: Path, text_path: Path) -> tuple[str, int]:
    text_path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open(pdf_path)
    parts: list[str] = []
    try:
        for page_index, page in enumerate(document, start=1):
            text = " ".join(page.get_text("text").split())
            if text:
                parts.append(f"\n\n[Page {page_index}]\n{text}")
        page_count = document.page_count
    finally:
        document.close()
    text = "\n".join(parts).strip()
    text_path.write_text(text, encoding="utf-8")
    return text, page_count


def load_metadata(papers_dir: Path = DEFAULT_PAPERS_DIR) -> list[PaperRecord]:
    metadata_path = papers_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    return [PaperRecord(**item) for item in raw]


def load_paper_text(record: PaperRecord) -> str:
    return Path(record.text_path).read_text(encoding="utf-8", errors="ignore")


def sample_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 6000:
        return text[:limit]
    head = limit // 3
    tail = limit // 5
    middle_budget = limit - head - tail
    middle_parts = []
    spans = 3
    step = max(1, (len(text) - head - tail) // spans)
    for index in range(spans):
        start = head + index * step
        middle_parts.append(text[start : start + middle_budget // spans])
    return (
        text[:head]
        + "\n\n[... sampled middle sections ...]\n\n"
        + "\n\n".join(middle_parts)
        + "\n\n[... sampled final section ...]\n\n"
        + text[-tail:]
    )


def paper_packet(record: PaperRecord, text: str, sample_chars: int) -> str:
    authors = ", ".join(record.authors[:8])
    if len(record.authors) > 8:
        authors += ", et al."
    return (
        f"Paper ID: {record.paper_id}\n"
        f"Title: {record.title}\n"
        f"Authors: {authors}\n"
        f"Published: {record.published}\n"
        f"Pages: {record.page_count}\n"
        f"Abstract: {record.abstract}\n\n"
        f"Representative full-paper text:\n{sample_text(text, sample_chars)}"
    )


def download_llm_papers(
    output_dir: Path,
    query: str,
    count: int,
    max_candidates: int,
    min_pages: int,
    sleep_seconds: float,
) -> list[PaperRecord]:
    output_dir = output_dir.resolve()
    pdf_dir = output_dir / "pdfs"
    text_dir = output_dir / "texts"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = search_arxiv(query, max_candidates)
    selected: list[PaperRecord] = []
    seen: set[str] = set()

    for candidate in candidates:
        if len(selected) >= count:
            break
        paper_id = candidate["paper_id"]
        if paper_id in seen:
            continue
        seen.add(paper_id)
        pdf_path = pdf_dir / f"{paper_id}.pdf"
        text_path = text_dir / f"{paper_id}.txt"
        try:
            print(f"Downloading {paper_id}: {candidate['title'][:90]}")
            download_file(candidate["pdf_url"], pdf_path)
            text, page_count = extract_pdf_text(pdf_path, text_path)
        except Exception as exc:
            print(f"Skipping {paper_id}: {exc}")
            continue
        if page_count < min_pages:
            print(f"Skipping {paper_id}: only {page_count} pages (< {min_pages})")
            pdf_path.unlink(missing_ok=True)
            text_path.unlink(missing_ok=True)
            continue
        record = PaperRecord(
            paper_id=paper_id,
            title=candidate["title"],
            authors=candidate["authors"],
            abstract=candidate["abstract"],
            published=candidate["published"],
            updated=candidate["updated"],
            pdf_url=candidate["pdf_url"],
            abs_url=candidate["abs_url"],
            pdf_path=str(pdf_path),
            text_path=str(text_path),
            page_count=page_count,
            text_chars=len(text),
        )
        selected.append(record)
        print(f"Selected {paper_id}: {page_count} pages, {len(text)} chars")
        time.sleep(sleep_seconds)

    if len(selected) < count:
        raise RuntimeError(f"Only selected {len(selected)} papers; try lowering --min-pages or raising --max-candidates.")

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps([asdict(item) for item in selected], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return selected


def iter_pdf_paths(records: Iterable[PaperRecord]) -> list[Path]:
    return [Path(record.pdf_path).resolve() for record in records]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download 10 long LLM-related papers from arXiv.")
    parser.add_argument("--out", type=Path, default=DEFAULT_PAPERS_DIR)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--min-pages", type=int, default=8)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    args = parser.parse_args()
    records = download_llm_papers(
        output_dir=args.out,
        query=args.query,
        count=args.count,
        max_candidates=args.max_candidates,
        min_pages=args.min_pages,
        sleep_seconds=args.sleep_seconds,
    )
    print(f"Wrote {len(records)} papers to {args.out.resolve()}")


if __name__ == "__main__":
    main()
