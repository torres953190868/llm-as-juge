import argparse

from length_bias_common import read_jsonl, utc_now, write_jsonl
from length_bias_metadata import run_metadata


DEFAULT_INPUT = "mt_bench_questions_answers_padded_deepseek.jsonl"
DEFAULT_OUTPUT = "length_bias_trials.jsonl"
DEFAULT_EXCLUDED = "excluded_length_bias_samples.jsonl"
DEFAULT_MIN_RATIO = 1.3
DEFAULT_MAX_RATIO = 2.0
PROMPT_CONDITIONS = ("standard_anti_length", "neutral_no_length")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Build pairwise length-bias judge trials from padded answers."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--excluded-output", default=DEFAULT_EXCLUDED)
    parser.add_argument("--min-ratio", type=float, default=DEFAULT_MIN_RATIO)
    parser.add_argument("--max-ratio", type=float, default=DEFAULT_MAX_RATIO)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--require-answer-turns", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def get_length_ratio(row):
    ratio = row.get("length_ratio")
    if ratio is not None:
        return float(ratio)

    original = row.get("original_word_count", 0)
    padded = row.get("padded_word_count", 0)
    if not original:
        return 0.0
    return padded / original


def answer_turn_status(row):
    original_turns = row.get("original_answer_turns")
    padded_turns = row.get("padded_answer_turns")

    if original_turns is None and padded_turns is None:
        return "not_available", None
    if not isinstance(original_turns, list) or not isinstance(padded_turns, list):
        return "invalid", "missing_answer_turns"
    if len(original_turns) != len(padded_turns):
        return "invalid", "answer_turn_count_mismatch"
    return "validated", None


def exclusion_reason(row, min_ratio, max_ratio, require_answer_turns=False):
    ratio = get_length_ratio(row)
    if ratio < min_ratio:
        return "below_min_length_ratio"
    if ratio > max_ratio:
        return "above_max_length_ratio"

    turn_status, turn_error = answer_turn_status(row)
    if require_answer_turns and turn_status != "validated":
        return turn_error or "missing_answer_turns"
    return turn_error


def question_turns(row):
    turns = row.get("question_turns")
    if isinstance(turns, list) and turns:
        return turns
    return [row["question"]]


def make_trial(row, prompt_condition, condition):
    is_long_a = condition == "long_A"
    long_answer = row["padded_answer"]
    short_answer = row["original_answer"]
    ratio = get_length_ratio(row)
    turn_status, _ = answer_turn_status(row)
    generated_at = utc_now()

    return {
        "trial_id": (
            f"q{row['question_id']}_{prompt_condition}_{condition}"
        ),
        "bias_type": "length",
        "question_id": row["question_id"],
        "category": row["category"],
        "condition": condition,
        "prompt_condition": prompt_condition,
        "question_turns": question_turns(row),
        "answer_a": long_answer if is_long_a else short_answer,
        "answer_b": short_answer if is_long_a else long_answer,
        "long_answer_position": "A" if is_long_a else "B",
        "original_word_count": row["original_word_count"],
        "padded_word_count": row["padded_word_count"],
        "length_ratio": ratio,
        "answer_turn_status": turn_status,
        "padding_prompt_version": row.get("padding_prompt_version"),
        "generated_at": generated_at,
        "created_at": generated_at,
    }


def build_trials(rows, min_ratio, max_ratio, require_answer_turns=False):
    trials = []
    excluded = []

    for row in rows:
        reason = exclusion_reason(row, min_ratio, max_ratio, require_answer_turns)
        if reason:
            generated_at = utc_now()
            excluded.append(
                {
                    "question_id": row.get("question_id"),
                    "category": row.get("category"),
                    "excluded_reason": reason,
                    "length_ratio": get_length_ratio(row),
                    "original_word_count": row.get("original_word_count"),
                    "padded_word_count": row.get("padded_word_count"),
                    "padding_prompt_version": row.get("padding_prompt_version"),
                    "generated_at": generated_at,
                    "created_at": generated_at,
                }
            )
            continue

        for prompt_condition in PROMPT_CONDITIONS:
            for condition in ("long_A", "long_B"):
                trials.append(make_trial(row, prompt_condition, condition))

    return trials, excluded


def run(args):
    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]

    trials, excluded = build_trials(
        rows, args.min_ratio, args.max_ratio, args.require_answer_turns
    )
    metadata = run_metadata(
        input_path=args.input,
        extra={
            "stage": "prepare_length_bias_trials",
            "output": args.output,
            "excluded_output": args.excluded_output,
            "min_ratio": args.min_ratio,
            "max_ratio": args.max_ratio,
            "require_answer_turns": args.require_answer_turns,
            "limit": args.limit,
        },
    )
    for trial in trials:
        trial["run_metadata"] = metadata
    for row in excluded:
        row["run_metadata"] = metadata
    print(f"Read {len(rows)} padded rows")
    print(f"Built {len(trials)} trials")
    print(f"Excluded {len(excluded)} rows")

    if args.dry_run:
        if trials:
            first = trials[0]
            print(
                "First trial: "
                f"{first['trial_id']} long={first['long_answer_position']}"
            )
        if excluded:
            first_excluded = excluded[0]
            print(
                "First excluded: "
                f"q{first_excluded['question_id']} "
                f"{first_excluded['excluded_reason']}"
            )
        return

    write_jsonl(args.output, trials)
    write_jsonl(args.excluded_output, excluded)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
