# Project Memory

## Project Overview

- Project name: `llm-judge-bias`
- Research focus: verify whether `LLM-as-a-judge` exhibits `position bias` and `length bias`
- Current benchmark/data source: `FastChat` MT-Bench questions and model answers
- Primary workflow: prepare question-answer pairs, construct controlled judge prompts, run bias evaluation, summarize results

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
  - `import json.py`

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

- `pad_answers_deepseek.py`: generate padded answer variants with DeepSeek and reject outputs outside the configured length ratio.
- `prepare_length_bias_trials.py`: validate padded rows and build pairwise `long_A` / `long_B` trial records.
- `run_length_bias_judge.py`: run one or more OpenAI-compatible judge models and save raw plus parsed judgments.
- `analyze_length_bias_results.py`: summarize parsed judgments by judge, prompt condition, and category.
- `run_length_bias_experiment.py`: orchestration entrypoint for prepare, judge, and analysis stages.

Default pilot constraints:

- pilot length ratio target: `1.3 <= padded_word_count / original_word_count <= 2.0`
- current padding model default: DeepSeek `deepseek-v4-flash` via `DEEPSEEK_API_KEY`
- padding generation default concurrency: `--parallel 3`
- `1.3x` is a pilot run-through threshold, not a strong-manipulation threshold for final claims.
- prompt conditions: `standard_anti_length` and `neutral_no_length`
- position control: every included sample should produce both `long_A` and `long_B`
- current legacy padded rows may lack structured answer turn fields; use `prepare_length_bias_trials.py --require-answer-turns` when strict turn-boundary validation is required.

## Environment Notes

- Expected working environment: PowerShell on Windows
- Common virtual environment name: `llm-judge-env`
- The FastChat download script in this repository has been adapted to prefer `curl.exe` on Windows

## Useful Commands

From repo root:

```powershell
python -m py_compile "import json.py"
python "import json.py"
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
