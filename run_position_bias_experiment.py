import argparse

from run_bias_judge import (
    add_common_args,
    add_judge_args,
    command,
    judge_flags,
    run_command,
)


POSITION_TRIALS = "position_bias_trials.jsonl"
RAW_OUTPUT = "raw_position_bias_judgments.jsonl"
PARSED_OUTPUT = "parsed_position_bias_judgments.jsonl"
DEFAULT_SOURCE_MODEL_A = "gpt-4"
DEFAULT_SOURCE_MODEL_B = "gpt-3.5-turbo"


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
    parser.add_argument(
        "--exclude-question-id",
        dest="exclude_question_ids",
        action="append",
        type=int,
        default=[],
        help="Additional question_id to exclude from prepared position-bias trials.",
    )
    parser.add_argument(
        "--include-known-empty-response-questions",
        action="store_true",
        help=(
            "Pass through known empty-response MT-Bench questions instead of "
            "using the default exclusions in 09_prepare_position_bias_trials.py."
        ),
    )
    add_common_args(parser)
    add_judge_args(parser)
    return parser


def prepare_command(args):
    flags = [
        "--source-model-a", args.source_model_a,
        "--source-model-b", args.source_model_b,
        "--output-path", POSITION_TRIALS,
    ]
    if args.question_limit is not None:
        flags.extend(["--limit", str(args.question_limit)])
    for question_id in args.exclude_question_ids:
        flags.extend(["--exclude-question-id", str(question_id)])
    if args.include_known_empty_response_questions:
        flags.append("--include-known-empty-response-questions")
    return command("09_prepare_position_bias_trials.py", flags)


def judge_command(args):
    extra = [
        "--trials", POSITION_TRIALS,
        "--raw-output", RAW_OUTPUT,
        "--parsed-output", PARSED_OUTPUT,
    ]
    if args.trial_limit is not None:
        extra.extend(["--limit", str(args.trial_limit)])
    return command("10_run_position_bias_judge.py", judge_flags(args, extra))


def analyze_command(_args):
    return command(
        "11_analyze_position_bias_results.py",
        [
            "--input", PARSED_OUTPUT,
            "--output-json", "position_bias_summary.json",
            "--output-txt", "position_bias_summary.txt",
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
