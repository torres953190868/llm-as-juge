import argparse

from length_bias_common import read_jsonl, write_jsonl
from length_bias_manipulation import make_manipulation_trial
from length_bias_metadata import run_metadata


DEFAULT_INPUT = "mt_bench_questions_answers_padded_deepseek.jsonl"
DEFAULT_OUTPUT = "manipulation_check_trials.jsonl"


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Prepare dry-run friendly manipulation-check trials."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def limited_rows(rows, limit):
    if limit is None:
        return rows
    if limit < 0:
        raise ValueError("--limit must be non-negative")
    return rows[:limit]


def build_trials(rows, metadata):
    input_sha256 = metadata.get("input_sha256")
    return [
        make_manipulation_trial(row, input_sha256, metadata)
        for row in rows
    ]


def run(args):
    rows = limited_rows(read_jsonl(args.input), args.limit)
    metadata = run_metadata(
        input_path=args.input,
        extra={
            "script": "03_prepare_manipulation_check_trials.py",
            "output_path": args.output,
            "dry_run": args.dry_run,
            "limit": args.limit,
        },
    )
    trials = build_trials(rows, metadata)

    print(f"Read {len(rows)} padded rows")
    print(f"Built {len(trials)} manipulation-check trials")
    print(f"Input SHA256: {metadata.get('input_sha256')}")

    if trials:
        first = trials[0]
        print(
            "First trial: "
            f"q{first['question_id']} "
            f"category={first['category']} "
            f"ratio={first['length_ratio']:.3f}"
        )

    if args.dry_run:
        print("Dry run: no output file written")
        return

    write_jsonl(args.output, trials)
    print(f"Wrote {args.output}")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
