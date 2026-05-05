"""Shared helpers for bias-experiment orchestration scripts."""

import subprocess
import sys


# ---------------------------------------------------------------------------
# CLI argument groups
# ---------------------------------------------------------------------------

def add_judge_args(parser):
    """Add judge-related CLI arguments common to all experiment scripts."""
    parser.add_argument("--judge-config", default=None)
    parser.add_argument("--judge-model", default="deepseek-v4-pro")
    parser.add_argument("--deepseek", type=int, choices=(0, 1), default=1)
    parser.add_argument("--gemini", type=int, choices=(0, 1), default=1)
    parser.add_argument("--opencode-go", type=int, choices=(0, 1), default=0)
    parser.add_argument("--xiaomi", type=int, choices=(0, 1), default=1)
    return parser


def add_common_args(parser):
    """Add general CLI arguments common to all experiment scripts."""
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


# ---------------------------------------------------------------------------
# Command helpers
# ---------------------------------------------------------------------------

def command(script, args):
    cmd = [sys.executable, script]
    cmd.extend(args)
    return cmd


def run_command(cmd):
    print("Running: " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Judge flag builder
# ---------------------------------------------------------------------------

def judge_flags(args, extra_flags=None):
    """Build CLI flags for ``07_run_length_bias_judge.py``.

    ``extra_flags`` is an optional list of additional flags to prepend
    (e.g. ``["--trials", "position_bias_trials.jsonl"]``).
    """
    flags = list(extra_flags or [])
    flags.extend([
        "--judge-model", args.judge_model,
        "--deepseek", str(args.deepseek),
        "--gemini", str(args.gemini),
        "--opencode-go", str(args.opencode_go),
        "--xiaomi", str(args.xiaomi),
    ])
    if args.overwrite:
        flags.append("--overwrite")
    if args.judge_config:
        flags.extend(["--judge-config", args.judge_config])
    return flags
