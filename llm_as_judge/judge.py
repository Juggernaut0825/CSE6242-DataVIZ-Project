from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .config import EvalSettings
from .llm_client import EvalLLMClient


JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator for paper-question answering systems.
Grade each answer against the golden answer, not against your outside knowledge.
Do not reward unsupported extra claims. Return valid JSON only."""

METRICS = ["accuracy", "recall", "faithfulness", "completeness", "specificity"]


def _judge_prompt(question: dict[str, Any], answers: dict[str, dict[str, Any]]) -> str:
    packed_answers = {
        name: {
            "answer": payload.get("answer", ""),
            "latency_seconds": payload.get("latency_seconds"),
            "retrieved_context_chars": payload.get("retrieved_context_chars"),
        }
        for name, payload in answers.items()
    }
    return f"""Evaluate the systems on a 0-5 scale for each metric:
- accuracy: factual correctness versus the golden answer.
- recall: coverage of key facts from the golden answer.
- faithfulness: absence of contradictions, hallucinations, or unsupported additions.
- completeness: answers all parts of the question.
- specificity: includes paper-specific details rather than generic statements.

Also provide a short rationale for each system. Do not judge latency here; it is measured separately.

Return JSON:
{{
  "scores": {{
    "system_name": {{
      "accuracy": 0,
      "recall": 0,
      "faithfulness": 0,
      "completeness": 0,
      "specificity": 0,
      "overall": 0,
      "rationale": "..."
    }}
  }}
}}

Question ID: {question["id"]}
Paper ID: {question.get("paper_id")}
Question: {question["question"]}
Golden answer: {question["golden_answer"]}

System answers:
{json.dumps(packed_answers, ensure_ascii=False, indent=2)}
"""


async def judge_answers_for_question(
    settings: EvalSettings,
    question: dict[str, Any],
    answers: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    client = EvalLLMClient(settings)
    raw, usage = await client.chat_json(
        model=settings.judge_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": _judge_prompt(question, answers)},
        ],
        temperature=0.0,
        max_completion_tokens=2000,
    )
    scores = raw.get("scores", {})
    normalized: dict[str, Any] = {}
    for system_name, answer in answers.items():
        item = scores.get(system_name, {})
        metric_scores = {metric: _score(item.get(metric)) for metric in METRICS}
        overall = _score(item.get("overall"))
        if overall == 0 and any(metric_scores.values()):
            overall = mean(metric_scores.values())
        normalized[system_name] = {
            **metric_scores,
            "overall": overall,
            "rationale": str(item.get("rationale", "")).strip(),
            "judge_usage": usage,
            "latency_seconds": float(answer.get("latency_seconds") or 0.0),
            "retrieved_context_chars": int(answer.get("retrieved_context_chars") or 0),
        }
    return {
        "question_id": question["id"],
        "paper_id": question.get("paper_id"),
        "scores": normalized,
    }


def _score(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(5.0, numeric))


def summarize_results(judgments: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    rows = []
    by_question_fastest: dict[str, float] = {}
    for judgment in judgments:
        latencies = [
            payload.get("latency_seconds", 0.0)
            for payload in judgment.get("scores", {}).values()
            if payload.get("latency_seconds", 0.0) > 0
        ]
        by_question_fastest[judgment["question_id"]] = min(latencies) if latencies else 0.0

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for judgment in judgments:
        fastest = by_question_fastest.get(judgment["question_id"], 0.0)
        for system_name, payload in judgment.get("scores", {}).items():
            item = dict(payload)
            latency = float(item.get("latency_seconds") or 0.0)
            item["time_efficiency"] = 5.0 if latency <= 0 or fastest <= 0 else max(0.0, min(5.0, 5.0 * fastest / latency))
            grouped[system_name].append(item)

    for system_name, items in sorted(grouped.items()):
        row = {"system": system_name, "n": len(items)}
        for metric in [*METRICS, "overall", "time_efficiency"]:
            row[metric] = round(mean(item.get(metric, 0.0) for item in items), 3) if items else 0.0
        latencies = sorted(float(item.get("latency_seconds") or 0.0) for item in items)
        row["avg_latency_seconds"] = round(mean(latencies), 3) if latencies else 0.0
        row["p95_latency_seconds"] = round(latencies[int(0.95 * (len(latencies) - 1))], 3) if latencies else 0.0
        row["avg_context_chars"] = round(mean(int(item.get("retrieved_context_chars") or 0) for item in items), 1) if items else 0.0
        rows.append(row)

    summary = {"rows": rows}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(_summary_markdown(rows), encoding="utf-8")
    (output_dir / "summary.csv").write_text(_summary_csv(rows), encoding="utf-8")
    return summary


def _summary_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No results.\n"
    columns = [
        "system",
        "n",
        "accuracy",
        "recall",
        "faithfulness",
        "completeness",
        "specificity",
        "overall",
        "time_efficiency",
        "avg_latency_seconds",
        "avg_context_chars",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def _summary_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    columns = list(rows[0].keys())
    lines = [",".join(columns)]
    for row in rows:
        values = [str(row.get(column, "")).replace(",", ";") for column in columns]
        lines.append(",".join(values))
    return "\n".join(lines) + "\n"

