import argparse
import subprocess
import sys


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run the length-bias experiment pipeline."
    )
    parser.add_argument(
        "--stage",
        choices=("all", "prepare", "judge", "analyze"),
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--judge-config", default=None)
    parser.add_argument("--judge-model", default="deepseek-v4-flash")
    return parser


def command(script, args):
    cmd = [sys.executable, script]
    cmd.extend(args)
    return cmd


def run_command(cmd):
    print("Running: " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def common_flags(args, include_overwrite=False):
    flags = []
    if args.dry_run:
        flags.append("--dry-run")
    if args.limit is not None:
        flags.extend(["--limit", str(args.limit)])
    if include_overwrite and args.overwrite:
        flags.append("--overwrite")
    return flags


def prepare_command(args):
    return command("prepare_length_bias_trials.py", common_flags(args))


def judge_command(args):
    flags = common_flags(args, include_overwrite=True)
    flags.extend(["--judge-model", args.judge_model])
    if args.judge_config:
        flags.extend(["--judge-config", args.judge_config])
    return command("run_length_bias_judge.py", flags)


def analyze_command(args):
    flags = []
    if args.dry_run:
        flags.append("--dry-run")
    return command("analyze_length_bias_results.py", flags)


def run(args):
    if args.stage in ("all", "prepare"):
        run_command(prepare_command(args))
    if args.stage in ("all", "judge"):
        run_command(judge_command(args))
    if args.stage in ("all", "analyze"):
        run_command(analyze_command(args))


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
