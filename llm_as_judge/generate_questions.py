from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from .config import DEFAULT_PAPERS_DIR, DEFAULT_RESULTS_DIR, load_settings
from .llm_client import EvalLLMClient, gather_with_concurrency
from .papers import load_metadata, load_paper_text, paper_packet


QUESTION_SYSTEM_PROMPT = """You create rigorous evaluation questions for paper-reading systems.
Each question must be answerable from the supplied paper text, must require paper-specific details,
and must not be a generic LLM trivia question. Golden answers should be concise but complete.
Return valid JSON only."""


def _question_prompt(packet: str, questions_per_paper: int) -> str:
    return f"""Create exactly {questions_per_paper} evaluation questions for the paper below.

Requirements:
- Questions should cover methods, experimental setup, results, limitations, ablations, or definitions.
- At least one question should require a specific numeric or named result when available.
- Avoid yes/no questions.
- Golden answers must be directly grounded in the paper text.
- Include the paper_id in every item.
- Use stable ids q_<paper_id>_01, q_<paper_id>_02, ...

Return JSON in this shape:
{{
  "questions": [
    {{
      "id": "q_<paper_id>_01",
      "paper_id": "<paper_id>",
      "question": "...",
      "golden_answer": "...",
      "evidence_hint": "short phrase identifying where the answer comes from",
      "difficulty": "easy|medium|hard"
    }}
  ]
}}

Paper:
{packet}
"""


def _extract_items(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, dict):
        return []
    for key in ("questions", "items", "qa_pairs", "qas", "data"):
        value = raw.get(key)
        if isinstance(value, list):
            return value
    dict_values = list(raw.values())
    if dict_values and all(isinstance(value, dict) for value in dict_values):
        return dict_values
    return []


def _first_text(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if value:
            return str(value).strip()
    return ""


def _normalize_questions(raw: Any, paper_id: str, count: int) -> list[dict[str, Any]]:
    items = _extract_items(raw)
    normalized = []
    for index, item in enumerate(items[:count], start=1):
        if not isinstance(item, dict):
            continue
        question = _first_text(item, ("question", "question_text", "q", "prompt"))
        answer = _first_text(item, ("golden_answer", "gold_answer", "reference_answer", "answer", "a"))
        if not question or not answer:
            continue
        normalized.append(
            {
                "id": str(item.get("id") or f"q_{paper_id}_{index:02d}"),
                "paper_id": str(item.get("paper_id") or paper_id),
                "question": question,
                "golden_answer": answer,
                "evidence_hint": _first_text(item, ("evidence_hint", "evidence", "source_hint")),
                "difficulty": str(item.get("difficulty", "medium")).strip() or "medium",
            }
        )
    if len(normalized) != count:
        raise RuntimeError(f"Expected {count} questions for {paper_id}, got {len(normalized)}")
    return normalized


async def generate_questions(
    papers_dir: Path,
    output_path: Path,
    questions_per_paper: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    settings = load_settings()
    client = EvalLLMClient(settings)
    records = load_metadata(papers_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async def one_paper(record):
        print(f"Generating questions for {record.paper_id}: {record.title[:80]}", flush=True)
        text = load_paper_text(record)
        packet = paper_packet(record, text, settings.paper_sample_chars)
        raw: Any = {}
        usage: dict[str, int] = {}
        last_error: Exception | None = None
        for attempt in range(1, 3):
            try:
                raw, usage = await client.chat_json(
                    model=settings.question_model,
                    messages=[
                        {"role": "system", "content": QUESTION_SYSTEM_PROMPT},
                        {"role": "user", "content": _question_prompt(packet, questions_per_paper)},
                    ],
                    temperature=0.2,
                    max_completion_tokens=3200,
                )
                questions = _normalize_questions(raw, record.paper_id, questions_per_paper)
                break
            except Exception as exc:
                last_error = exc
                debug_path = output_path.parent / f"raw_questions_{record.paper_id}_attempt_{attempt}.json"
                debug_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
                if attempt == 2:
                    raise RuntimeError(f"Failed to generate {questions_per_paper} questions for {record.paper_id}: {exc}") from exc
        else:
            raise RuntimeError(f"Failed to generate questions for {record.paper_id}: {last_error}")
        for item in questions:
            item["question_model"] = settings.question_model
            item["generation_usage"] = usage
        print(f"Generated {len(questions)} questions for {record.paper_id}", flush=True)
        return questions

    if concurrency <= 1:
        nested = []
        questions_so_far = []
        for record in records:
            group = await one_paper(record)
            nested.append(group)
            questions_so_far.extend(group)
            output_path.write_text(json.dumps(questions_so_far, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        nested = await gather_with_concurrency(concurrency, *(one_paper(record) for record in records))
    questions = [item for group in nested for item in group]
    output_path.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 50 golden Q/A items with GPT-5.4.")
    parser.add_argument("--papers-dir", type=Path, default=DEFAULT_PAPERS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS_DIR / "questions_golden.json")
    parser.add_argument("--questions-per-paper", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    questions = asyncio.run(
        generate_questions(
            papers_dir=args.papers_dir,
            output_path=args.out,
            questions_per_paper=args.questions_per_paper,
            concurrency=args.concurrency,
        )
    )
    print(f"Wrote {len(questions)} questions to {args.out.resolve()}")


if __name__ == "__main__":
    main()
