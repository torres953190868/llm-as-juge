# Project Memory

## Project Overview

- Project name: `llm-judge-bias`
- Research focus: verify whether `LLM-as-a-judge` exhibits `position bias` and `length bias`
- Current benchmark/data source: `FastChat` MT-Bench questions and model answers
- Primary workflow: prepare question-answer pairs, run manipulation checks for padded answers, construct controlled judge prompts, run bias evaluation, summarize results

## Research Goal

This repository is for experiments on whether an LLM judge is systematically biased by:

- answer order / presentation position
- answer length / verbosity

The immediate use case is to work with MT-Bench `question` and `model_answer` data from FastChat.

## Important Data Paths

- MT-Bench questions:
  - `FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl`
- MT-Bench reference answers:
  - `FastChat/fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl`
- Downloaded model answers:
  - `FastChat/data/mt_bench/model_answer/*.jsonl`
- Downloaded GPT-4 judgments:
  - `FastChat/data/mt_bench/model_judgment/gpt-4_single.jsonl`
  - `FastChat/data/mt_bench/model_judgment/gpt-4_pair.jsonl`
- Local export script:
  - `00_export_mt_bench_pairs.py`

Important: the project currently uses two MT-Bench data roots.

- Questions live under `FastChat/fastchat/llm_judge/data/mt_bench/`
- Downloaded answers and judgments live under `FastChat/data/mt_bench/`

Do not assume these directories are interchangeable. Always verify which one a script reads from before running experiments.

## MT-Bench Data Schema

### Question file

Each row in `question.jsonl` contains:

- `question_id`
- `category`
- `turns`

Known MT-Bench categories:

- `writing`
- `roleplay`
- `reasoning`
- `math`
- `coding`
- `extraction`
- `stem`
- `humanities`

### Model answer file

Each row in `model_answer/*.jsonl` typically contains:

- `question_id`
- `answer_id`
- `model_id`
- `choices`
- `tstamp`

The actual answer text is usually in:

- `choices[0]["turns"]`

## Bias Experiment Guidelines

When designing judge-bias experiments:

- keep `question_id` as the primary key linking questions, answers, and judgments
- preserve the original question content before any prompt transformation
- explicitly control answer position when testing `position bias`
- explicitly control answer length when testing `length bias`
- do not mix position and length manipulations in the same experimental condition unless the condition is intentionally factorial
- record the exact judge prompt template used for every run
- record the evaluated judge model name and version
- keep raw outputs and parsed results separate
- run manipulation checks before length-bias trials so invalid padded answers do not enter judge evaluation

For `position bias` studies:

- compare the same two answers under swapped order
- keep all non-order prompt text fixed
- log both original order and swapped order results

For `length bias` studies:

- measure answer length with a single consistent metric per experiment
- preferred metrics: `token_count`, then `word_count`, then `char_count`
- report clearly which metric was used

## Coding and Project Conventions

- Favor high cohesion and low coupling.
- If a file grows beyond 400 lines, stop and consider refactoring it into smaller modules.
- Keep data preparation, evaluation logic, and result analysis in separate files.
- Use clear filenames that reflect one responsibility.
- Prefer deterministic scripts and explicit paths over hidden state.
- Save intermediate artifacts instead of recomputing expensive preprocessing.

## Validation Workflow

- After every code change, run a minimal validation immediately.
- Do not postpone verification until the end of a long edit sequence.
- For scripts, prefer the smallest meaningful check first:
  - syntax check
  - load a few rows
  - verify output file exists
  - confirm row counts or key fields

## Length Bias Pipeline

Current dedicated scripts:

- `01_screen_length_bias_eligibility.py`: screen MT-Bench samples for length-padding eligibility.
- `02_pad_answers_deepseek.py`: generate padded answer variants with DeepSeek and reject outputs outside the configured length ratio.
- `03_prepare_manipulation_check_trials.py`: build manipulation-check tasks for padded answers without calling judge APIs.
- `04_run_manipulation_check_judge.py`: run one or more OpenAI-compatible judge models on manipulation-check tasks and save raw plus parsed JSON checks.
- `05_filter_manipulation_check_results.py`: keep only padded rows that pass strict manipulation checks.
- `length_bias_manipulation_judge.py`: helper module for manipulation-check payloads, JSON parsing, trial identifiers, and strict pass policy.
- `06_prepare_length_bias_trials.py`: validate checked padded rows and build pairwise `long_A` / `long_B` trial records.
- `07_run_length_bias_judge.py`: run one or more OpenAI-compatible judge models and save raw plus parsed judgments.
- `08_analyze_length_bias_results.py`: summarize parsed judgments by judge, prompt condition, category, and swapped `long_A` / `long_B` pairs.
- `length_bias_statistics.py`: helper module for question-cluster statistics, swapped-position paired statistics, deterministic bootstrap confidence intervals, data-shape interpretation, and sample coverage/attrition metadata.
- `09_prepare_position_bias_trials.py`: build standalone position-bias swapped A/B trials from two original MT-Bench model-answer files; default pair is `gpt-4` vs `gpt-3.5-turbo`.
- `10_run_position_bias_judge.py`: run one or more OpenAI-compatible judge models on position-bias trials and save raw plus parsed judgments.
- `11_analyze_position_bias_results.py`: summarize parsed position-bias judgments by source-model preference and A/B position preference.
- `run_position_bias_experiment.py`: orchestration entrypoint for standalone position-bias prepare, judge, and analyze stages.
- `run_length_bias_experiment.py`: orchestration entrypoint for legacy prepare/judge/analyze and checked-all stages.
- `run_bias_judge.py`: shared helpers for bias-experiment orchestration scripts.

Default pilot constraints:

- pilot length ratio target: `1.3 <= padded_word_count / original_word_count <= 2.0`
- current padding model default: DeepSeek `deepseek-v4-pro` via `DEEPSEEK_API_KEY`
- padding generation default concurrency: `--parallel 3`
- current manipulation-check judge default: official DeepSeek `deepseek-v4-pro` via `DEEPSEEK_API_KEY`, with `--parallel 3`
- current length-bias judge concurrency: `07_run_length_bias_judge.py --parallel N` means up to `N` concurrent requests per judge model, not total global concurrency.
- `1.3x` is a pilot run-through threshold, not a strong-manipulation threshold for final claims.
- prompt conditions: `standard_anti_length` and `neutral_no_length`
- position control: every included sample should produce both `long_A` and `long_B`
- current legacy padded rows may lack structured answer turn fields; use `06_prepare_length_bias_trials.py --require-answer-turns` when strict turn-boundary validation is required.
- strict manipulation-check pass policy: `semantic_equivalence=true`, `new_facts=false`, `structure_improvement=false`, and `quality_improvement=false`.
- current length-bias and position-bias judge default suite: Gemini `gemini-3-flash-preview` directly, official DeepSeek `deepseek-v4-pro` via `DEEPSEEK_API_KEY`, plus official Xiaomi `mimo-v2-pro` via `XIAOMI_API_KEY`.
- OpenCode Go judge candidates are disabled in the built-in model list and should not be used for the current default runs.
- current parsed pilot shape: `252 rows = 21 questions x 2 prompts x 2 positions x 3 judges`.
- current pilot attrition: `80 -> 28 -> 21` from screened MT-Bench rows, to eligible rows, to analyzed questions with parsed judgments.
- current pilot category coverage is limited after screening/padding and should not be described as full MT-Bench coverage.
- length-bias and position-bias claims should remain separate; swapped `long_A` / `long_B` controls are not a full standalone position-bias experiment.
- standalone position-bias default source pair: `gpt-4` vs `gpt-3.5-turbo`.
- standalone position-bias default judge setting: Gemini plus official DeepSeek plus official Xiaomi (`--gemini 1 --deepseek 1 --xiaomi 1 --opencode-go 0`).
- standalone position-bias prepare now excludes MT-Bench question IDs `105`, `107`, `128`, and `136` by default because DeepSeek/Xiaomi repeatedly returned empty judge content for these prompts; use `09_prepare_position_bias_trials.py --include-known-empty-response-questions` only when intentionally rerunning those problematic cases.
- use `run_position_bias_experiment.py` for the dedicated position-bias prepare/judge/analyze pipeline.
- final claims require a manipulation check that padded answers preserve meaning while changing answer length.
- dry-run commands should not call paid APIs; paid API usage starts when padding or judging scripts run without `--dry-run`.

## Environment Notes

- Expected working environment: PowerShell on Windows
- Common virtual environment name: `llm-judge-env`
- The FastChat download script in this repository has been adapted to prefer `curl.exe` on Windows

## Useful Commands

From repo root:

```powershell
python -m py_compile "00_export_mt_bench_pairs.py"
python "00_export_mt_bench_pairs.py"
```

From `FastChat` repo root:

```powershell
python -m fastchat.llm_judge.download_mt_bench_pregenerated
```

To inspect question categories quickly:

```powershell
@'
import json
from collections import Counter
from pathlib import Path
path = Path("FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl")
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
print(Counter(row["category"] for row in rows))
'@ | python -
```

## Documentation Intent

This `CLAUDE.md` should be treated as the shared project memory for this repository:

- project purpose
- real data locations
- evaluation constraints
- code organization rules
- repeatable commands

Update this file whenever the experiment design, folder structure, or core workflow changes.
