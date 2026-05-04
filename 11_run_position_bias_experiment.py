import argparse
import subprocess
import sys


POSITION_TRIALS = "position_bias_trials.jsonl"
RAW_OUTPUT = "raw_position_bias_judgments.jsonl"
PARSED_OUTPUT = "parsed_position_bias_judgments.jsonl"
DEFAULT_SOURCE_MODEL_A = "gpt-4"
DEFAULT_SOURCE_MODEL_B = "gpt-3.5-turbo"
DEFAULT_JUDGE_MODEL = "deepseek-v4-flash"


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run the standalone position-bias experiment pipeline."
    )
    parser.add_argument(
        "--stage",
        choices=("all", "prepare", "judge", "analyze"),
        default="all",
    )
    parser.add_argument("--source-model-a", default=DEFAULT_SOURCE_MODEL_A)
    parser.add_argument("--source-model-b", default=DEFAULT_SOURCE_MODEL_B)
    parser.add_argument("--question-limit", type=int, default=None)
    parser.add_argument("--trial-limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--judge-config", default=None)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--deepseek", type=int, choices=(0, 1), default=1)
    parser.add_argument("--gemini", type=int, choices=(0, 1), default=0)
    parser.add_argument("--xiaomi", type=int, choices=(0, 1), default=0)
    return parser


def command(script, args):
    cmd = [sys.executable, script]
    cmd.extend(args)
    return cmd


def run_command(cmd):
    print("Running: " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def prepare_command(args):
    flags = [
        "--source-model-a",
        args.source_model_a,
        "--source-model-b",
        args.source_model_b,
        "--output-path",
        POSITION_TRIALS,
    ]
    if args.question_limit is not None:
        flags.extend(["--limit", str(args.question_limit)])
    return command("09_prepare_position_bias_trials.py", flags)


def judge_command(args):
    flags = [
        "--trials",
        POSITION_TRIALS,
        "--raw-output",
        RAW_OUTPUT,
        "--parsed-output",
        PARSED_OUTPUT,
        "--judge-model",
        args.judge_model,
        "--deepseek",
        str(args.deepseek),
        "--gemini",
        str(args.gemini),
        "--xiaomi",
        str(args.xiaomi),
    ]
    if args.trial_limit is not None:
        flags.extend(["--limit", str(args.trial_limit)])
    if args.overwrite:
        flags.append("--overwrite")
    if args.judge_config:
        flags.extend(["--judge-config", args.judge_config])
    return command("07_run_length_bias_judge.py", flags)


def analyze_command(_args):
    return command(
        "10_analyze_position_bias_results.py",
        [
            "--input",
            PARSED_OUTPUT,
            "--output-json",
            "position_bias_summary.json",
            "--output-txt",
            "position_bias_summary.txt",
        ],
    )


def selected_commands(args):
    cmds = []
    if args.stage in ("all", "prepare"):
        cmds.append(prepare_command(args))
    if args.stage in ("all", "judge"):
        cmds.append(judge_command(args))
    if args.stage in ("all", "analyze"):
        cmds.append(analyze_command(args))
    return cmds


def run(args):
    cmds = selected_commands(args)
    if args.dry_run:
        for cmd in cmds:
            print("Would run: " + " ".join(cmd), flush=True)
        return

    for cmd in cmds:
        run_command(cmd)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
