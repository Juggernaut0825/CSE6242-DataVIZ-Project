from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .bare_service import BareGPTService
from .config import DEFAULT_PAPERS_DIR, DEFAULT_RESULTS_DIR, load_settings
from .generate_questions import generate_questions
from .judge import judge_answers_for_question, summarize_results
from .llm_client import gather_with_concurrency
from .papers import DEFAULT_QUERY, download_llm_papers, load_metadata
from .papermem_service import PaperMemService
from .rag_service import RagAgentService


def load_questions(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    if limit:
        return questions[:limit]
    return questions


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def run_system_answers(
    *,
    papers_dir: Path,
    results_dir: Path,
    questions: list[dict[str, Any]],
    systems: list[str],
    papermem_ingest: bool,
    reuse_papermem_state: bool,
    papermem_project_id: str | None,
    papermem_session_id: str | None,
    force_rag_rebuild: bool,
    concurrency: int,
    replace_answers: bool,
) -> list[dict[str, Any]]:
    settings = load_settings()
    services: dict[str, Any] = {}
    if "rag" in systems:
        rag = RagAgentService(settings, papers_dir, results_dir / "rag_index.json")
        await rag.prepare(force_rebuild=force_rag_rebuild)
        services["rag_gpt4o_mini"] = rag
    if "bare" in systems:
        services["bare_gpt4o_mini"] = BareGPTService(settings)
    if "papermem" in systems:
        papermem = PaperMemService(
            settings=settings,
            papers_dir=papers_dir,
            state_path=results_dir / "papermem_project.json",
            project_id=papermem_project_id,
            session_id=papermem_session_id,
        )
        await papermem.prepare(ingest=papermem_ingest, reuse_state=reuse_papermem_state)
        services["papermem"] = papermem

    async def one(service_name: str, service: Any, question: dict[str, Any]) -> dict[str, Any]:
        print(f"Answering {question['id']} with {service_name}", flush=True)
        try:
            return await service.answer(question)
        except Exception as exc:
            return {
                "system": service_name,
                "question_id": question["id"],
                "answer": f"[ERROR] {type(exc).__name__}: {exc}",
                "latency_seconds": 0.0,
                "usage": {},
                "retrieved_context_chars": 0,
                "retrieved_chunks": [],
            }

    tasks = [
        one(service_name, service, question)
        for question in questions
        for service_name, service in services.items()
    ]
    new_answers = await gather_with_concurrency(concurrency, *tasks)
    answers_path = results_dir / "answers.json"
    if answers_path.exists() and not replace_answers:
        existing = json.loads(answers_path.read_text(encoding="utf-8"))
    else:
        existing = []
    merged: dict[tuple[str, str], dict[str, Any]] = {
        (item["question_id"], item["system"]): item for item in existing
    }
    for item in new_answers:
        merged[(item["question_id"], item["system"])] = item
    answers = list(merged.values())
    write_json(answers_path, answers)
    return answers


async def run_judge(
    *,
    questions: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    results_dir: Path,
    concurrency: int,
) -> list[dict[str, Any]]:
    settings = load_settings()
    answers_by_question: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for answer in answers:
        answers_by_question[answer["question_id"]][answer["system"]] = answer

    async def one(question: dict[str, Any]) -> dict[str, Any]:
        print(f"Judging {question['id']}", flush=True)
        return await judge_answers_for_question(settings, question, answers_by_question[question["id"]])

    semaphore = asyncio.Semaphore(concurrency)
    judgments: list[dict[str, Any]] = []

    async def guarded(question: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await one(question)

    tasks = [asyncio.create_task(guarded(question)) for question in questions]
    for task in asyncio.as_completed(tasks):
        judgments.append(await task)
        write_json(results_dir / "judgments.json", judgments)
    write_json(results_dir / "judgments.json", judgments)
    summarize_results(judgments, results_dir)
    return judgments


async def main_async(args: argparse.Namespace) -> None:
    papers_dir = args.papers_dir.resolve()
    results_dir = args.results_dir.resolve()
    questions_path = args.questions.resolve()

    if args.download:
        download_llm_papers(
            output_dir=papers_dir,
            query=args.query,
            count=args.paper_count,
            max_candidates=args.max_candidates,
            min_pages=args.min_pages,
            sleep_seconds=args.sleep_seconds,
        )

    if args.generate_questions or not questions_path.exists():
        await generate_questions(
            papers_dir=papers_dir,
            output_path=questions_path,
            questions_per_paper=args.questions_per_paper,
            concurrency=args.question_concurrency,
        )

    load_metadata(papers_dir)
    questions = load_questions(questions_path, args.limit_questions)
    write_json(results_dir / "questions_used.json", questions)

    answers_path = results_dir / "answers.json"
    if args.run_systems or not answers_path.exists():
        systems = [item.strip() for item in args.systems.split(",") if item.strip()]
        answers = await run_system_answers(
            papers_dir=papers_dir,
            results_dir=results_dir,
            questions=questions,
            systems=systems,
            papermem_ingest=args.papermem_ingest,
            reuse_papermem_state=not args.no_reuse_papermem_state,
            papermem_project_id=args.papermem_project_id,
            papermem_session_id=args.papermem_session_id,
            force_rag_rebuild=args.force_rag_rebuild,
            concurrency=args.answer_concurrency,
            replace_answers=args.replace_answers,
        )
    else:
        answers = json.loads(answers_path.read_text(encoding="utf-8"))

    if args.judge or not (results_dir / "judgments.json").exists():
        judgments = await run_judge(
            questions=questions,
            answers=answers,
            results_dir=results_dir,
            concurrency=args.judge_concurrency,
        )
        print(f"Wrote judgments for {len(judgments)} questions to {results_dir}")

    summary_path = results_dir / "summary.md"
    if summary_path.exists():
        print(summary_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PaperMem LLM-as-judge evaluation.")
    parser.add_argument("--papers-dir", type=Path, default=DEFAULT_PAPERS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--questions", type=Path, default=DEFAULT_RESULTS_DIR / "questions_golden.json")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--paper-count", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--min-pages", type=int, default=8)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--generate-questions", action="store_true")
    parser.add_argument("--questions-per-paper", type=int, default=5)
    parser.add_argument("--question-concurrency", type=int, default=2)
    parser.add_argument("--run-systems", action="store_true")
    parser.add_argument("--systems", default="papermem,rag,bare", help="Comma list: papermem,rag,bare")
    parser.add_argument("--answer-concurrency", type=int, default=2)
    parser.add_argument("--papermem-ingest", action="store_true", help="Ingest eval_paper2 PDFs into a PaperMem project.")
    parser.add_argument("--no-reuse-papermem-state", action="store_true")
    parser.add_argument("--papermem-project-id")
    parser.add_argument("--papermem-session-id")
    parser.add_argument("--force-rag-rebuild", action="store_true")
    parser.add_argument("--replace-answers", action="store_true", help="Replace answers.json instead of merging by question/system.")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-concurrency", type=int, default=2)
    parser.add_argument("--limit-questions", type=int)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
