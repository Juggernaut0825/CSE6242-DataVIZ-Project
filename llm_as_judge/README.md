# PaperMem LLM-as-Judge Evaluation

This folder evaluates 10 long LLM-related arXiv papers in `../eval_paper2` with 50 GPT-5.4-generated golden Q/A items.

Systems compared:

1. `papermem`: the local PaperMem backend via `/projects`, `/chat_sessions`, `/files/ingest`, and `/chat/stream`.
2. `rag_gpt4o_mini`: a new lightweight vector RAG service using embeddings + GPT-4o mini.
3. `bare_gpt4o_mini`: GPT-4o mini with no retrieval context.

Judge metrics:

- `accuracy`: correctness against the golden answer.
- `recall`: coverage of key facts.
- `faithfulness`: no contradictions or unsupported additions.
- `completeness`: answers every part of the question.
- `specificity`: uses paper-specific details.
- `time_efficiency`: measured relative latency score per question, not LLM-judged.

## Setup

```bash
cd /Users/zijianwang/Desktop/GT/SPRING26/CSE6242_DataVIZ/Project
python3 -m pip install -r llm_as_judge/requirements.txt
cp llm_as_judge/.env.example llm_as_judge/.env
```

If `backend/.env` already has `LLM_API_KEY`, the scripts can reuse it. Override with `EVAL_OPENAI_API_KEY` if needed.

## Download 10 long arXiv papers

```bash
python3 -m llm_as_judge.papers --out eval_paper2 --count 10 --min-pages 8
```

Outputs:

- `eval_paper2/pdfs/*.pdf`
- `eval_paper2/texts/*.txt`
- `eval_paper2/metadata.json`

## Generate 50 golden questions

```bash
python3 -m llm_as_judge.generate_questions \
  --papers-dir eval_paper2 \
  --out llm_as_judge/results/questions_golden.json \
  --questions-per-paper 5
```

## Run all systems and judge

Start PaperMem first if you want the `papermem` system:

```bash
docker compose up -d
cd backend
python3 -m pip install --user -r requirements.txt  # if your backend venv is not active
uvicorn app.main:app --reload --port 8000
```

Then in another shell:

```bash
cd /Users/zijianwang/Desktop/GT/SPRING26/CSE6242_DataVIZ/Project
python3 -m llm_as_judge.run_eval \
  --run-systems \
  --papermem-ingest \
  --judge
```

If you already ran `rag,bare` while PaperMem was offline, start the backend and then append only the PaperMem column:

```bash
python3 -m llm_as_judge.run_eval \
  --systems papermem \
  --run-systems \
  --papermem-ingest \
  --judge
```

`answers.json` is merged by `(question_id, system)` by default. Add `--replace-answers` for a clean replacement run.

For a fast smoke test:

```bash
python3 -m llm_as_judge.run_eval \
  --systems rag,bare \
  --run-systems \
  --judge \
  --limit-questions 3
```

Final outputs are written under `llm_as_judge/results/`:

- `questions_golden.json`
- `answers.json`
- `judgments.json`
- `summary.md`
- `summary.csv`
- `summary.json`

The average metric table is in `summary.md` and `summary.csv`.
