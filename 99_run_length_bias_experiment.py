import argparse
import subprocess
import sys


CHECKED_PADDED = "mt_bench_questions_answers_padded_deepseek_checked.jsonl"


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run the length-bias experiment pipeline."
    )
    parser.add_argument(
        "--stage",
        choices=("all", "checked-all", "prepare", "judge", "analyze"),
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--judge-config", default=None)
    parser.add_argument("--judge-model", default="deepseek-v4-flash")
    parser.add_argument("--deepseek", type=int, choices=(0, 1), default=0)
    parser.add_argument("--gemini", type=int, choices=(0, 1), default=1)
    parser.add_argument("--xiaomi", type=int, choices=(0, 1), default=1)
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
    return command("06_prepare_length_bias_trials.py", common_flags(args))


def checked_prepare_command(args):
    flags = common_flags(args)
    flags.extend(["--input", CHECKED_PADDED])
    return command("06_prepare_length_bias_trials.py", flags)


def prepare_check_command(args):
    return command("03_prepare_manipulation_check_trials.py", common_flags(args))


def run_check_command(args):
    flags = common_flags(args, include_overwrite=True)
    flags.extend(["--judge-model", args.judge_model])
    flags.extend(["--deepseek", str(args.deepseek)])
    flags.extend(["--gemini", str(args.gemini)])
    flags.extend(["--xiaomi", str(args.xiaomi)])
    if args.judge_config:
        flags.extend(["--judge-config", args.judge_config])
    return command("04_run_manipulation_check_judge.py", flags)


def filter_check_command(args):
    flags = common_flags(args)
    return command("05_filter_manipulation_check_results.py", flags)


def judge_command(args):
    flags = common_flags(args, include_overwrite=True)
    flags.extend(["--judge-model", args.judge_model])
    flags.extend(["--deepseek", str(args.deepseek)])
    flags.extend(["--gemini", str(args.gemini)])
    flags.extend(["--xiaomi", str(args.xiaomi)])
    if args.judge_config:
        flags.extend(["--judge-config", args.judge_config])
    return command("07_run_length_bias_judge.py", flags)


def analyze_command(args):
    flags = []
    if args.dry_run:
        flags.append("--dry-run")
    flags.extend(["--deepseek", str(args.deepseek)])
    flags.extend(["--gemini", str(args.gemini)])
    flags.extend(["--xiaomi", str(args.xiaomi)])
    return command("08_analyze_length_bias_results.py", flags)


def run(args):
    if args.stage == "checked-all":
        cmds = [
            prepare_check_command(args),
            run_check_command(args),
            filter_check_command(args),
            checked_prepare_command(args),
            judge_command(args),
            analyze_command(args),
        ]
        if args.dry_run:
            for cmd in cmds:
                print("Would run: " + " ".join(cmd), flush=True)
            return
        for cmd in cmds:
            run_command(cmd)
        return

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
