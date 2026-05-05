"""Run LLM judges on pairwise position-bias trials.

This is a thin wrapper around the shared judge runner in
``07_run_length_bias_judge``, with default paths set for position-bias
trials and outputs.
"""

import argparse
import importlib
import sys

from length_bias_judge import DEEPSEEK_MODEL


DEFAULT_TRIALS = "position_bias_trials.jsonl"
DEFAULT_RAW_OUTPUT = "raw_position_bias_judgments.jsonl"
DEFAULT_PARSED_OUTPUT = "parsed_position_bias_judgments.jsonl"
DEFAULT_MODEL = DEEPSEEK_MODEL
DEFAULT_ENV_FILE = ".env"


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run LLM judges on pairwise position-bias trials."
    )
    parser.add_argument("--trials", default=DEFAULT_TRIALS)
    parser.add_argument("--raw-output", default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--parsed-output", default=DEFAULT_PARSED_OUTPUT)
    parser.add_argument("--judge-config", default=None)
    parser.add_argument("--judge-model", default=DEFAULT_MODEL)
    parser.add_argument("--deepseek", type=int, choices=(0, 1), default=1)
    parser.add_argument("--gemini", type=int, choices=(0, 1), default=1)
    parser.add_argument("--opencode-go", type=int, choices=(0, 1), default=0)
    parser.add_argument("--xiaomi", type=int, choices=(0, 1), default=1)
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--parallel", type=int, default=3)
    return parser


def run(args):
    """Delegate to the shared judge runner in ``07``."""
    runner = importlib.import_module("07_run_length_bias_judge")
    runner.run(args)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
