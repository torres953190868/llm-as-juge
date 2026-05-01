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
python "screen_length_bias_eligibility.py" --dry-run
python "prepare_length_bias_trials.py" --dry-run
python "analyze_length_bias_results.py" --dry-run
```

The `FastChat/` directory is treated as an external data/source checkout. If it is
missing, restore it separately before running data preparation commands that read
MT-Bench files.
