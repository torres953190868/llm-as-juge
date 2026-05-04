import argparse

from length_bias_common import read_jsonl, utc_now, write_jsonl
from length_bias_manipulation import sample_sha256
from length_bias_manipulation_judge import CHECK_FIELDS, strict_passed
from length_bias_metadata import run_metadata


DEFAULT_INPUT = "mt_bench_questions_answers_padded_deepseek.jsonl"
DEFAULT_CHECKS = "parsed_manipulation_check_judgments.jsonl"
DEFAULT_OUTPUT = "mt_bench_questions_answers_padded_deepseek_checked.jsonl"
DEFAULT_EXCLUDED = "manipulation_check_excluded_samples.jsonl"
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Filter padded rows using strict manipulation-check results."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--checks", default=DEFAULT_CHECKS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--excluded-output", default=DEFAULT_EXCLUDED)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def check_key(row):
    return (row.get("question_id"), row.get("sample_sha256"))


def padded_key(row):
    return (row.get("question_id"), sample_sha256(row))


def index_checks(check_rows):
    indexed = {}
    judges = set()
    duplicates = set()
    for row in check_rows:
        judge_model = row.get("judge_model")
        key = check_key(row)
        if not judge_model or not key[0] or not key[1]:
            continue
        judges.add(judge_model)
        judge_rows = indexed.setdefault(key, {})
        if judge_model in judge_rows:
            duplicates.add((key, judge_model))
        judge_rows[judge_model] = row
    return indexed, sorted(judges), duplicates


def exclusion_row(row, reason, metadata, details=None):
    generated_at = utc_now()
    result = {
        "question_id": row.get("question_id"),
        "category": row.get("category"),
        "sample_sha256": sample_sha256(row),
        "excluded_reason": reason,
        "length_ratio": row.get("length_ratio"),
        "original_word_count": row.get("original_word_count"),
        "padded_word_count": row.get("padded_word_count"),
        "padding_prompt_version": row.get("padding_prompt_version"),
        "run_metadata": metadata,
        "generated_at": generated_at,
        "created_at": generated_at,
    }
    if details:
        result["details"] = details
    return result


def accepted_row(row, metadata):
    result = dict(row)
    result["manipulation_check_status"] = "passed"
    result["manipulation_check_policy"] = "strict_v1"
    result["manipulation_check_metadata"] = metadata
    return result


def evaluate_row(row, indexed_checks, required_judges):
    key = padded_key(row)
    rows_by_judge = indexed_checks.get(key)
    if not rows_by_judge:
        return False, "missing_check_results", {"required_judges": required_judges}

    missing = [judge for judge in required_judges if judge not in rows_by_judge]
    if missing:
        return False, "missing_check_judges", {"missing_judges": missing}

    failing = []
    invalid = []
    for judge in required_judges:
        check = rows_by_judge[judge]
        if check.get("parse_status") != "parsed":
            invalid.append(judge)
            continue
        if not strict_passed(check):
            failing.append(
                {
                    "judge_model": judge,
                    **{field: check.get(field) for field in CHECK_FIELDS},
                    "manipulation_passed": check.get("manipulation_passed"),
                }
            )

    if invalid:
        return False, "invalid_check_result", {"invalid_judges": invalid}
    if failing:
        return False, "failed_strict_check", {"failing_checks": failing}
    return True, None, None


def filter_rows(padded_rows, check_rows, metadata):
    indexed_checks, required_judges, duplicates = index_checks(check_rows)
    accepted = []
    excluded = []

    if not required_judges:
        for row in padded_rows:
            excluded.append(
                exclusion_row(row, "no_check_judges", metadata)
            )
        return accepted, excluded, required_judges, duplicates

    for row in padded_rows:
        passed, reason, details = evaluate_row(row, indexed_checks, required_judges)
        if passed:
            accepted.append(accepted_row(row, metadata))
        else:
            excluded.append(exclusion_row(row, reason, metadata, details))

    return accepted, excluded, required_judges, duplicates


def run(args):
    padded_rows = read_jsonl(args.input)
    if args.limit is not None:
        padded_rows = padded_rows[: args.limit]
    check_rows = read_jsonl(args.checks)
    metadata = run_metadata(
        input_path=args.input,
        extra={
            "stage": "filter_manipulation_check_results",
            "script": "05_filter_manipulation_check_results.py",
            "checks": args.checks,
            "output": args.output,
            "excluded_output": args.excluded_output,
            "policy": "strict_v1",
            "limit": args.limit,
        },
    )
    accepted, excluded, required_judges, duplicates = filter_rows(
        padded_rows, check_rows, metadata
    )

    print(f"Read {len(padded_rows)} padded row(s)")
    print(f"Read {len(check_rows)} parsed manipulation-check row(s)")
    print(f"Required judges: {', '.join(required_judges) if required_judges else 'none'}")
    print(f"Accepted {len(accepted)} row(s)")
    print(f"Excluded {len(excluded)} row(s)")
    if duplicates:
        print(f"Warning: {len(duplicates)} duplicate check key(s); last row wins")

    if args.dry_run:
        if accepted:
            print(f"First accepted: q{accepted[0].get('question_id')}")
        if excluded:
            first = excluded[0]
            print(
                "First excluded: "
                f"q{first.get('question_id')} {first.get('excluded_reason')}"
            )
        return

    write_jsonl(args.output, accepted)
    write_jsonl(args.excluded_output, excluded)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.excluded_output}")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
