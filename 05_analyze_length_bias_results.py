import argparse
import json
from collections import defaultdict

from length_bias_common import read_jsonl
from length_bias_judge import DEEPSEEK_MODEL, GEMINI_MODEL, XIAOMI_MODEL
from length_bias_metadata import run_metadata
from length_bias_pairing import PAIR_PATTERNS, summarize_swapped_pairs
from length_bias_plotting import render_academic_svg
from length_bias_statistics import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_BOOTSTRAP_SEED,
    build_sample_coverage,
    build_statistical_summary,
    compare_judge_results,
    render_statistics_text,
)


DEFAULT_INPUT = "parsed_length_bias_judgments.jsonl"
DEFAULT_OUTPUT_JSON = "length_bias_summary.json"
DEFAULT_OUTPUT_TXT = "length_bias_summary.txt"
DEFAULT_OUTPUT_IMAGE = "length_bias_summary.svg"
DEFAULT_SCREENING_SUMMARY = "length_bias_screening_summary.json"
DEFAULT_SCREENED = "length_bias_screened_samples.jsonl"
DEFAULT_PADDED = "mt_bench_questions_answers_padded_deepseek.jsonl"
DEFAULT_TRIALS = "length_bias_trials.jsonl"


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Summarize parsed length-bias judge results."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-txt", default=DEFAULT_OUTPUT_TXT)
    parser.add_argument("--output-image", default=DEFAULT_OUTPUT_IMAGE)
    parser.add_argument("--deepseek", type=int, choices=(0, 1), default=1)
    parser.add_argument("--gemini", type=int, choices=(0, 1), default=1)
    parser.add_argument("--xiaomi", type=int, choices=(0, 1), default=1)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=DEFAULT_BOOTSTRAP_ITERATIONS,
    )
    parser.add_argument("--screening-summary", default=DEFAULT_SCREENING_SUMMARY)
    parser.add_argument("--screened", default=DEFAULT_SCREENED)
    parser.add_argument("--padded", default=DEFAULT_PADDED)
    parser.add_argument("--trials", default=DEFAULT_TRIALS)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def empty_counts():
    return {
        "total": 0,
        "long_wins": 0,
        "short_wins": 0,
        "ties": 0,
        "invalid": 0,
    }


def add_row(counts, row):
    counts["total"] += 1
    winner = row.get("winner")
    long_won = row.get("long_answer_won")

    if winner == "invalid":
        counts["invalid"] += 1
    elif winner == "tie":
        counts["ties"] += 1
    elif long_won is True:
        counts["long_wins"] += 1
    elif long_won is False:
        counts["short_wins"] += 1
    else:
        counts["invalid"] += 1


def finalize_counts(counts):
    total = counts["total"]
    decisive = counts["long_wins"] + counts["short_wins"]

    summary = dict(counts)
    summary["decisive_total"] = decisive
    summary["long_win_rate"] = safe_div(counts["long_wins"], decisive)
    summary["short_win_rate"] = safe_div(counts["short_wins"], decisive)
    summary["tie_rate"] = safe_div(counts["ties"], total)
    summary["invalid_rate"] = safe_div(counts["invalid"], total)
    summary["net_length_preference"] = safe_div(
        counts["long_wins"] - counts["short_wins"], total
    )
    return summary


def safe_div(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


def selected_judge_models(args):
    models = []
    if args.deepseek:
        models.append(DEEPSEEK_MODEL)
    if args.gemini:
        models.append(GEMINI_MODEL)
    if args.xiaomi:
        models.append(XIAOMI_MODEL)
    return models


def filter_rows_by_judge(rows, judge_models):
    selected = set(judge_models)
    return [
        row for row in rows
        if row.get("judge_model") in selected
        and row.get("bias_type", "length") == "length"
    ]


def summarize_by_judge(rows):
    by_judge = defaultdict(empty_counts)
    for row in rows:
        add_row(by_judge[row.get("judge_model", "unknown")], row)
    return {
        judge: finalize_counts(counts)
        for judge, counts in sorted(by_judge.items())
    }


def summarize(rows, selected_models=None, source_row_count=None):
    by_judge_prompt = defaultdict(empty_counts)
    by_position = defaultdict(empty_counts)
    by_prompt_position = defaultdict(empty_counts)
    by_category = defaultdict(empty_counts)
    by_question = defaultdict(empty_counts)
    overall = empty_counts()

    for row in rows:
        add_row(overall, row)
        judge_prompt_key = (
            row.get("judge_model", "unknown"),
            row.get("prompt_condition", "unknown"),
        )
        category_key = (
            row.get("judge_model", "unknown"),
            row.get("prompt_condition", "unknown"),
            row.get("category", "unknown"),
        )
        position_key = row.get("condition", "unknown")
        prompt_position_key = (
            row.get("prompt_condition", "unknown"),
            row.get("condition", "unknown"),
        )
        question_key = (
            row.get("question_id", "unknown"),
            row.get("category", "unknown"),
        )
        add_row(by_judge_prompt[judge_prompt_key], row)
        add_row(by_position[position_key], row)
        add_row(by_prompt_position[prompt_position_key], row)
        add_row(by_category[category_key], row)
        add_row(by_question[question_key], row)

    comparison = compare_judge_results(rows)
    swapped_pairs = summarize_swapped_pairs(rows)
    return {
        "selected_judge_models": selected_models or [],
        "source_row_count": len(rows) if source_row_count is None else source_row_count,
        "filtered_row_count": len(rows),
        "overall": finalize_counts(overall),
        "by_judge": summarize_by_judge(rows),
        "cross_judge_agreement": comparison["pairwise_agreement"],
        "judge_disagreements": comparison["disagreements"],
        "swapped_pair_analysis": swapped_pairs,
        "by_judge_prompt": {
            f"{judge}::{prompt}": finalize_counts(counts)
            for (judge, prompt), counts in sorted(by_judge_prompt.items())
        },
        "by_position": {
            position: finalize_counts(counts)
            for position, counts in sorted(by_position.items())
        },
        "by_prompt_position": {
            f"{prompt}::{position}": finalize_counts(counts)
            for (prompt, position), counts in sorted(by_prompt_position.items())
        },
        "by_category": {
            f"{judge}::{prompt}::{category}": finalize_counts(counts)
            for (judge, prompt, category), counts in sorted(by_category.items())
        },
        "by_question": {
            f"q{question_id}::{category}": finalize_counts(counts)
            for (question_id, category), counts in sorted(by_question.items())
        },
    }


def percent(value):
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def render_text(summary):
    lines = []
    overall = summary["overall"]
    lines.append("Length Bias Summary")
    lines.append("===================")
    lines.append("")
    lines.append(
        "Selected judges: "
        + ", ".join(summary.get("selected_judge_models", []))
    )
    lines.append(
        f"Rows: filtered={summary.get('filtered_row_count', 0)}, "
        f"source={summary.get('source_row_count', 0)}"
    )
    lines.append("")
    lines.append(render_section_line("overall", overall))
    lines.extend(render_statistics_text(summary))
    lines.append("")
    lines.append("By judge")
    lines.append("--------")
    for key, counts in summary["by_judge"].items():
        lines.append(render_section_line(key, counts))
    lines.append("")
    lines.append("Cross-judge agreement")
    lines.append("---------------------")
    if summary["cross_judge_agreement"]:
        for key, counts in summary["cross_judge_agreement"].items():
            lines.append(
                f"{key}: common={counts['common_trials']}, "
                f"winner_agreement={percent(counts['winner_agreement_rate'])}, "
                f"length_preference_agreement="
                f"{percent(counts['length_preference_agreement_rate'])}"
            )
    else:
        lines.append("n/a")
    lines.append(f"Disagreement examples: {len(summary['judge_disagreements'])}")
    lines.append("")
    lines.append("Swapped long_A / long_B pairs")
    lines.append("-----------------------------")
    swapped_pairs = summary.get("swapped_pair_analysis", {})
    if swapped_pairs:
        lines.append(render_pair_line("overall", swapped_pairs["overall"]))
        lines.append("")
        lines.append("By judge")
        for key, counts in swapped_pairs["by_judge"].items():
            lines.append(render_pair_line(key, counts))
        lines.append("")
        lines.append("By judge and prompt")
        for key, counts in swapped_pairs["by_judge_prompt"].items():
            lines.append(render_pair_line(key, counts))
    else:
        lines.append("n/a")
    lines.append("")
    lines.append("By judge and prompt")
    lines.append("-------------------")
    for key, counts in summary["by_judge_prompt"].items():
        lines.append(render_section_line(key, counts))
    lines.append("")
    lines.append("By long-answer position")
    lines.append("-----------------------")
    for key, counts in summary["by_position"].items():
        lines.append(render_section_line(key, counts))
    lines.append("")
    lines.append("By prompt and position")
    lines.append("----------------------")
    for key, counts in summary["by_prompt_position"].items():
        lines.append(render_section_line(key, counts))
    lines.append("")
    lines.append("By category")
    lines.append("-----------")
    for key, counts in summary["by_category"].items():
        lines.append(render_section_line(key, counts))
    lines.append("")
    return "\n".join(lines)


def render_section_line(label, counts):
    return (
        f"{label}: total={counts['total']}, "
        f"long={counts['long_wins']}, short={counts['short_wins']}, "
        f"tie={counts['ties']}, invalid={counts['invalid']}, "
        f"long_win_rate={percent(counts['long_win_rate'])}, "
        f"tie_rate={percent(counts['tie_rate'])}, "
        f"net={percent(counts['net_length_preference'])}"
    )


def render_pair_line(label, counts):
    pattern_text = ", ".join(
        f"{pattern}={counts.get(pattern, 0)}" for pattern in PAIR_PATTERNS
    )
    return (
        f"{label}: total_pairs={counts['total_pairs']}, "
        f"{pattern_text}, "
        f"long_consistent_rate={percent(counts['long_consistent_rate'])}, "
        f"position_A_rate={percent(counts['position_A_rate'])}, "
        f"tie_both_rate={percent(counts['tie_both_rate'])}"
    )


def run(args):
    judge_models = selected_judge_models(args)
    if not judge_models:
        raise SystemExit("No judge models selected. Set at least one of --deepseek, --gemini, or --xiaomi to 1.")

    source_rows = read_jsonl(args.input)
    rows = filter_rows_by_judge(source_rows, judge_models)
    summary = summarize(
        rows,
        selected_models=judge_models,
        source_row_count=len(source_rows),
    )
    summary["statistical_analysis"] = build_statistical_summary(
        rows,
        seed=args.bootstrap_seed,
        iterations=args.bootstrap_iterations,
    )
    summary["sample_coverage"] = build_sample_coverage(
        rows,
        screening_summary_path=args.screening_summary,
        screened_path=args.screened,
        padded_path=args.padded,
        trials_path=args.trials,
    )
    summary["run_metadata"] = run_metadata(
        args.input,
        extra={
            "stage": "analyze_length_bias_results",
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_iterations": args.bootstrap_iterations,
            "coverage_paths": {
                "screening_summary": args.screening_summary,
                "screened": args.screened,
                "padded": args.padded,
                "trials": args.trials,
            },
        },
    )
    text = render_text(summary)
    print(f"Read {len(source_rows)} parsed judgment rows")
    print(f"Selected {len(rows)} rows for analysis")
    print(text)

    if args.dry_run:
        return

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(args.output_txt, "w", encoding="utf-8") as f:
        f.write(text)
    with open(args.output_image, "w", encoding="utf-8") as f:
        f.write(render_academic_svg(summary))


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
