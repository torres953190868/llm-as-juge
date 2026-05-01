import argparse
import json
import re
from collections import Counter, defaultdict

from length_bias_common import utc_now, word_count, write_jsonl
from length_bias_samples import join_turns


DEFAULT_QUESTION_FILE = "FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl"
DEFAULT_ANSWER_FILE = "FastChat/data/mt_bench/model_answer/gpt-4.jsonl"
DEFAULT_SCREENED_OUTPUT = "length_bias_screened_samples.jsonl"
DEFAULT_ELIGIBLE_OUTPUT = "length_bias_eligible_samples.jsonl"
DEFAULT_EXCLUDED_OUTPUT = "length_bias_screening_excluded_samples.jsonl"
DEFAULT_SUMMARY_OUTPUT = "length_bias_screening_summary.json"
SCREENING_VERSION = "length_bias_screening_v1_conservative"
DEFAULT_MIN_ANSWER_WORDS = 100
DEFAULT_MIN_TURN_WORDS = 60

AUTO_EXCLUDED_CATEGORIES = {
    "coding": "coding_task",
    "extraction": "extraction_or_structured_output_task",
    "math": "math_precision_task",
    "reasoning": "reasoning_precision_task",
}

STRICT_OUTPUT_RE = re.compile(
    r"\b(json|csv|xml|yaml)\b|"
    r"output\s+(in|as)|"
    r"return\s+the\s+results|"
    r"present\s+the\s+results\s+in|"
    r"following\s+format|"
    r"specified\s+format",
    re.IGNORECASE,
)

STRICT_LENGTH_OR_EDITING_RE = re.compile(
    r"fewer\s+than|"
    r"less\s+than|"
    r"no\s+more\s+than|"
    r"under\s+\d+|"
    r"\bheadline\b|"
    r"\btitle\b|"
    r"edit\s+the\s+following|"
    r"correct\s+any\s+grammatical|"
    r"\btranslator\b|"
    r"\btranslate\b|"
    r"proof.*less\s+than",
    re.IGNORECASE,
)

PRECISION_OR_FACT_RE = re.compile(
    r"balanced\s+chemical\s+equation|"
    r"\bequations?\b|"
    r"\bformula\b|"
    r"\bcalculate\b|"
    r"identify\s+and\s+fix\s+one\s+incorrect\s+fact",
    re.IGNORECASE,
)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Screen MT-Bench samples for length-bias padding eligibility."
    )
    parser.add_argument("--questions", default=DEFAULT_QUESTION_FILE)
    parser.add_argument("--answers", default=DEFAULT_ANSWER_FILE)
    parser.add_argument("--screened-output", default=DEFAULT_SCREENED_OUTPUT)
    parser.add_argument("--eligible-output", default=DEFAULT_ELIGIBLE_OUTPUT)
    parser.add_argument("--excluded-output", default=DEFAULT_EXCLUDED_OUTPUT)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--min-answer-words", type=int, default=DEFAULT_MIN_ANSWER_WORDS)
    parser.add_argument("--min-turn-words", type=int, default=DEFAULT_MIN_TURN_WORDS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def read_jsonl_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_answer_map(path):
    answers = {}
    for row in read_jsonl_rows(path):
        answers[row["question_id"]] = row
    return answers


def answer_turns(answer_row):
    choices = answer_row.get("choices", [])
    if not choices:
        return []
    turns = choices[0].get("turns", [])
    if not isinstance(turns, list):
        return []
    return turns


def classify_sample(question_row, turns, args):
    question_text = join_turns(question_row.get("turns", []))
    answer_text = join_turns(turns)
    category = question_row.get("category", "unknown")
    turn_word_counts = [word_count(turn) for turn in turns]

    exclusion_reasons = []
    review_reasons = []

    category_reason = AUTO_EXCLUDED_CATEGORIES.get(category)
    if category_reason:
        exclusion_reasons.append(category_reason)

    if not turns:
        exclusion_reasons.append("missing_answer_turns")
    elif word_count(answer_text) < args.min_answer_words:
        exclusion_reasons.append("too_short_answer")
    elif min(turn_word_counts) < args.min_turn_words:
        exclusion_reasons.append("too_short_answer_turn")

    if "```" in answer_text:
        exclusion_reasons.append("contains_code_block")
    if STRICT_OUTPUT_RE.search(question_text):
        exclusion_reasons.append("strict_output_format_task")

    if not exclusion_reasons:
        if STRICT_LENGTH_OR_EDITING_RE.search(question_text):
            review_reasons.append("strict_length_or_editing_constraint")
        if PRECISION_OR_FACT_RE.search(question_text):
            review_reasons.append("precision_or_fact_correction_risk")

    if exclusion_reasons:
        return "excluded", exclusion_reasons, []
    if review_reasons:
        return "manual_review", review_reasons, review_reasons
    return "eligible", [], []


def make_screened_row(question_row, answer_row, args):
    question_turns = question_row.get("turns", [])
    turns = answer_turns(answer_row) if answer_row else []
    question_text = join_turns(question_turns)
    answer_text = join_turns(turns)
    eligibility, exclusion_reasons, review_reasons = classify_sample(
        question_row, turns, args
    )

    row = {
        "question_id": question_row["question_id"],
        "category": question_row["category"],
        "question_turns": question_turns,
        "question": question_text,
        "original_answer_turns": turns,
        "original_answer": answer_text,
        "model_id": answer_row.get("model_id") if answer_row else None,
        "answer_id": answer_row.get("answer_id") if answer_row else None,
        "tstamp": answer_row.get("tstamp") if answer_row else None,
        "eligibility": eligibility,
        "eligibility_reasons": exclusion_reasons or review_reasons,
        "exclusion_reasons": exclusion_reasons,
        "review_reasons": review_reasons,
        "original_word_count": word_count(answer_text),
        "original_char_count": len(answer_text),
        "original_turn_word_counts": [word_count(turn) for turn in turns],
        "screening_version": SCREENING_VERSION,
        "created_at": utc_now(),
    }
    return row


def summarize(rows, args):
    status_counts = Counter(row["eligibility"] for row in rows)
    category_counts = defaultdict(Counter)
    reason_counts = Counter()

    for row in rows:
        category_counts[row["category"]][row["eligibility"]] += 1
        for reason in row["eligibility_reasons"]:
            reason_counts[reason] += 1

    return {
        "screening_version": SCREENING_VERSION,
        "generated_at": utc_now(),
        "sources": {
            "questions": args.questions,
            "answers": args.answers,
        },
        "thresholds": {
            "min_answer_words": args.min_answer_words,
            "min_turn_words": args.min_turn_words,
        },
        "total_rows": len(rows),
        "counts_by_status": dict(sorted(status_counts.items())),
        "counts_by_category": {
            category: dict(sorted(counts.items()))
            for category, counts in sorted(category_counts.items())
        },
        "counts_by_reason": dict(sorted(reason_counts.items())),
    }


def print_summary(summary):
    print(f"Screened {summary['total_rows']} sample(s)")
    counts = summary["counts_by_status"]
    print(
        "Status: "
        f"eligible={counts.get('eligible', 0)}, "
        f"manual_review={counts.get('manual_review', 0)}, "
        f"excluded={counts.get('excluded', 0)}"
    )
    print("By category:")
    for category, counts_by_status in summary["counts_by_category"].items():
        details = ", ".join(
            f"{status}={count}" for status, count in counts_by_status.items()
        )
        print(f"  {category}: {details}")
    print("By reason:")
    if not summary["counts_by_reason"]:
        print("  n/a")
    for reason, count in summary["counts_by_reason"].items():
        print(f"  {reason}: {count}")


def run(args):
    questions = read_jsonl_rows(args.questions)
    if args.limit is not None:
        questions = questions[: args.limit]
    answers = load_answer_map(args.answers)

    rows = [
        make_screened_row(question, answers.get(question["question_id"]), args)
        for question in questions
    ]
    eligible = [row for row in rows if row["eligibility"] == "eligible"]
    excluded = [row for row in rows if row["eligibility"] != "eligible"]
    summary = summarize(rows, args)

    print_summary(summary)
    if args.dry_run:
        return

    write_jsonl(args.screened_output, rows)
    write_jsonl(args.eligible_output, eligible)
    write_jsonl(args.excluded_output, excluded)
    with open(args.summary_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(rows)} screened rows to {args.screened_output}")
    print(f"Wrote {len(eligible)} eligible rows to {args.eligible_output}")
    print(f"Wrote {len(excluded)} excluded/review rows to {args.excluded_output}")
    print(f"Wrote summary to {args.summary_output}")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
