import argparse
import json
from collections import defaultdict

from length_bias_common import read_jsonl
from length_bias_metadata import run_metadata


DEFAULT_INPUT = "parsed_position_bias_judgments.jsonl"
DEFAULT_OUTPUT_JSON = "position_bias_summary.json"
DEFAULT_OUTPUT_TXT = "position_bias_summary.txt"


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Summarize parsed position-bias judge results."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-txt", default=DEFAULT_OUTPUT_TXT)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def safe_div(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


def percent(value):
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def position_rows(rows):
    return [
        row for row in rows
        if row.get("bias_type") == "position" or row.get("model_a_position")
    ]


def empty_counts():
    return {
        "total": 0,
        "model_a_wins": 0,
        "model_b_wins": 0,
        "position_a_wins": 0,
        "position_b_wins": 0,
        "ties": 0,
        "invalid": 0,
    }


def add_row(counts, row):
    counts["total"] += 1
    winner = row.get("winner")
    if winner == "invalid":
        counts["invalid"] += 1
    elif winner == "tie":
        counts["ties"] += 1
    elif winner == "A":
        counts["position_a_wins"] += 1
    elif winner == "B":
        counts["position_b_wins"] += 1
    else:
        counts["invalid"] += 1

    model_a_won = row.get("model_a_won")
    if model_a_won is True:
        counts["model_a_wins"] += 1
    elif model_a_won is False:
        counts["model_b_wins"] += 1


def finalize_counts(counts):
    decisive = counts["model_a_wins"] + counts["model_b_wins"]
    position_decisive = counts["position_a_wins"] + counts["position_b_wins"]
    summary = dict(counts)
    summary["decisive_total"] = decisive
    summary["model_a_win_rate"] = safe_div(counts["model_a_wins"], decisive)
    summary["position_a_win_rate"] = safe_div(
        counts["position_a_wins"], position_decisive
    )
    summary["tie_rate"] = safe_div(counts["ties"], counts["total"])
    summary["invalid_rate"] = safe_div(counts["invalid"], counts["total"])
    return summary


def swapped_pair_key(row):
    return (
        row.get("judge_model", "unknown"),
        row.get("prompt_condition", "unknown"),
        row.get("question_id", "unknown"),
        row.get("source_model_a", "unknown"),
        row.get("source_model_b", "unknown"),
    )


def classify_swapped_pair(pair_rows):
    model_a_a = pair_rows.get("model_a_A")
    model_a_b = pair_rows.get("model_a_B")
    if not model_a_a or not model_a_b:
        return "missing_pair"
    if "invalid" in {model_a_a.get("winner"), model_a_b.get("winner")}:
        return "invalid_pair"
    if model_a_a.get("winner") == "tie" and model_a_b.get("winner") == "tie":
        return "tie_both"
    if model_a_a.get("model_a_won") is True and model_a_b.get("model_a_won") is True:
        return "source_model_a_both"
    if model_a_a.get("model_a_won") is False and model_a_b.get("model_a_won") is False:
        return "source_model_b_both"
    if model_a_a.get("winner") == "A" and model_a_b.get("winner") == "A":
        return "position_A_both"
    if model_a_a.get("winner") == "B" and model_a_b.get("winner") == "B":
        return "position_B_both"
    return "mixed_or_partial_tie"


def summarize_swapped_pairs(rows):
    pairs = defaultdict(dict)
    for row in rows:
        condition = row.get("condition")
        if condition in {"model_a_A", "model_a_B"}:
            pairs[swapped_pair_key(row)][condition] = row

    counts = defaultdict(int)
    counts["total_pairs"] = 0
    for pair_rows in pairs.values():
        counts["total_pairs"] += 1
        counts[classify_swapped_pair(pair_rows)] += 1

    total = counts["total_pairs"]
    return {
        **dict(counts),
        "source_model_a_consistent_rate": safe_div(
            counts["source_model_a_both"], total
        ),
        "position_A_consistent_rate": safe_div(counts["position_A_both"], total),
    }


def summarize(rows):
    overall = empty_counts()
    by_judge = defaultdict(empty_counts)
    for row in rows:
        add_row(overall, row)
        add_row(by_judge[row.get("judge_model", "unknown")], row)
    return {
        "overall": finalize_counts(overall),
        "by_judge": {
            judge: finalize_counts(counts)
            for judge, counts in sorted(by_judge.items())
        },
        "swapped_pair_analysis": summarize_swapped_pairs(rows),
    }


def render_count_line(label, counts):
    return (
        f"{label}: total={counts['total']}, "
        f"model_a={counts['model_a_wins']}, model_b={counts['model_b_wins']}, "
        f"position_A={counts['position_a_wins']}, "
        f"position_B={counts['position_b_wins']}, ties={counts['ties']}, "
        f"invalid={counts['invalid']}, "
        f"model_a_win_rate={percent(counts['model_a_win_rate'])}, "
        f"position_A_win_rate={percent(counts['position_a_win_rate'])}"
    )


def render_text(summary):
    lines = [
        "Position Bias Summary",
        "=====================",
        "",
        render_count_line("overall", summary["overall"]),
        "",
        "By judge",
        "--------",
    ]
    for judge, counts in summary["by_judge"].items():
        lines.append(render_count_line(judge, counts))
    paired = summary["swapped_pair_analysis"]
    lines.extend(
        [
            "",
            "Swapped pairs",
            "-------------",
            (
                f"total_pairs={paired.get('total_pairs', 0)}, "
                f"source_model_a_both={paired.get('source_model_a_both', 0)}, "
                f"source_model_b_both={paired.get('source_model_b_both', 0)}, "
                f"position_A_both={paired.get('position_A_both', 0)}, "
                f"position_B_both={paired.get('position_B_both', 0)}, "
                f"tie_both={paired.get('tie_both', 0)}, "
                f"mixed={paired.get('mixed_or_partial_tie', 0)}, "
                f"position_A_consistent_rate="
                f"{percent(paired.get('position_A_consistent_rate'))}"
            ),
        ]
    )
    return "\n".join(lines)


def run(args):
    source_rows = read_jsonl(args.input)
    rows = position_rows(source_rows)
    summary = summarize(rows)
    summary["source_row_count"] = len(source_rows)
    summary["filtered_row_count"] = len(rows)
    summary["run_metadata"] = run_metadata(
        args.input,
        extra={"stage": "analyze_position_bias_results"},
    )
    text = render_text(summary)
    print(f"Read {len(source_rows)} parsed judgment rows")
    print(f"Selected {len(rows)} position-bias rows")
    print(text)

    if args.dry_run:
        return

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(args.output_txt, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
