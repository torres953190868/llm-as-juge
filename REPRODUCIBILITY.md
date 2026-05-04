# Reproducibility Notes

## Version Control Boundary

This repository tracks the experiment scripts, configuration examples, and project notes.
It intentionally ignores local-only or generated files:

- `.env` and other secret-bearing environment files
- `llm-judge-env/`, `.venv/`, and Python cache directories
- the upstream `FastChat/` checkout and downloaded MT-Bench data
- generated JSONL/TXT/SVG experiment artifacts and temporary API probes

Use `.env.example` as the template for required API key names. Keep real keys in `.env`.

## Environment

The root scripts currently require only Python standard-library modules.
Use Python 3.11+ and install future dependencies with:

```powershell
python -m pip install -r requirements.txt
```

## Minimal Validation

After changing scripts, run the smallest relevant check first:

```powershell
python -m py_compile "length_bias_common.py" "length_bias_samples.py"
python "01_screen_length_bias_eligibility.py" --dry-run
python "03_prepare_manipulation_check_trials.py" --dry-run --limit 2
python "04_run_manipulation_check_judge.py" --dry-run --limit 1 --deepseek 1 --gemini 0 --xiaomi 0
python "06_prepare_length_bias_trials.py" --dry-run
python "09_prepare_position_bias_trials.py" --dry-run --limit 2
python "11_run_position_bias_experiment.py" --dry-run --question-limit 2
python "08_analyze_length_bias_results.py" --dry-run
```

The `FastChat/` directory is treated as an external data/source checkout. If it is
missing, restore it separately before running data preparation commands that read
MT-Bench files.

## Current Pilot Scope

The current length-bias dataset is a pilot run. The observed attrition is
`80 -> 28 -> 17`: 80 screened MT-Bench rows, 28 eligible rows, and 17 questions
with parsed judge results. The current parsed file has 204 rows, interpreted as
`17 questions x 2 prompts x 2 positions x 3 judges`.

Category coverage is limited by the screening and padding stages. Current parsed
coverage is concentrated in humanities, roleplay, and STEM, so the results should
not be described as full MT-Bench coverage.

Length-bias and position-bias claims should remain separate. The length-bias
pipeline includes swapped `long_A` / `long_B` trials as a control, but a final
position-bias claim requires its own experiment and analysis plan.

Final length-bias claims require a manipulation check before length-bias trial
preparation: padded answers must be verified to preserve meaning while changing
answer length. The strict pass policy is semantic equivalence with no new facts,
structure improvement, or quality improvement. Dry-run commands do not call paid
APIs; paid API usage starts only when padding or judging scripts are run without
`--dry-run`.

Position-bias trial preparation is separate from length-bias trial preparation.
Use `11_run_position_bias_experiment.py` for the dedicated position-bias
prepare/judge/analyze pipeline. It defaults to `gpt-4` vs `gpt-3.5-turbo` as
the source-answer pair and `deepseek-v4-flash` as the only judge
(`--deepseek 1 --gemini 0 --xiaomi 0`). The lower-level debugging path remains
`09_prepare_position_bias_trials.py`, `07_run_length_bias_judge.py` with
position-bias input/output paths, and `10_analyze_position_bias_results.py`.
