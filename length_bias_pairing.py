from collections import defaultdict


PAIR_PATTERNS = (
    "long_both_positions",
    "short_both_positions",
    "position_A_both",
    "position_B_both",
    "tie_both",
    "mixed_or_partial_tie",
    "invalid_pair",
    "missing_pair",
)


def empty_pair_counts():
    counts = {"total_pairs": 0}
    for pattern in PAIR_PATTERNS:
        counts[pattern] = 0
    return counts


def safe_div(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


def pair_key(row):
    return (
        row.get("judge_model", "unknown"),
        row.get("prompt_condition", "unknown"),
        row.get("question_id", "unknown"),
    )


def collect_pairs(rows):
    pairs = defaultdict(dict)
    for row in rows:
        condition = row.get("condition")
        if condition in ("long_A", "long_B"):
            pairs[pair_key(row)][condition] = row
    return pairs


def classify_pair(pair_rows):
    long_a = pair_rows.get("long_A")
    long_b = pair_rows.get("long_B")
    if not long_a or not long_b:
        return "missing_pair"

    winner_a = long_a.get("winner")
    winner_b = long_b.get("winner")
    long_won_a = long_a.get("long_answer_won")
    long_won_b = long_b.get("long_answer_won")

    if winner_a == "invalid" or winner_b == "invalid":
        return "invalid_pair"
    if winner_a == "tie" and winner_b == "tie":
        return "tie_both"
    if long_won_a is True and long_won_b is True:
        return "long_both_positions"
    if long_won_a is False and long_won_b is False:
        return "short_both_positions"
    if winner_a == "A" and winner_b == "A":
        return "position_A_both"
    if winner_a == "B" and winner_b == "B":
        return "position_B_both"
    return "mixed_or_partial_tie"


def add_pair(counts, pattern):
    counts["total_pairs"] += 1
    counts[pattern] += 1


def finalize_pair_counts(counts):
    total = counts["total_pairs"]
    summary = dict(counts)
    summary["long_consistent_rate"] = safe_div(
        counts["long_both_positions"], total
    )
    summary["position_A_rate"] = safe_div(counts["position_A_both"], total)
    summary["tie_both_rate"] = safe_div(counts["tie_both"], total)
    return summary


def summarize_swapped_pairs(rows):
    overall = empty_pair_counts()
    by_judge = defaultdict(empty_pair_counts)
    by_judge_prompt = defaultdict(empty_pair_counts)

    for (judge, prompt, _question_id), pair_rows in sorted(collect_pairs(rows).items()):
        pattern = classify_pair(pair_rows)
        add_pair(overall, pattern)
        add_pair(by_judge[judge], pattern)
        add_pair(by_judge_prompt[(judge, prompt)], pattern)

    return {
        "overall": finalize_pair_counts(overall),
        "by_judge": {
            judge: finalize_pair_counts(counts)
            for judge, counts in sorted(by_judge.items())
        },
        "by_judge_prompt": {
            f"{judge}::{prompt}": finalize_pair_counts(counts)
            for (judge, prompt), counts in sorted(by_judge_prompt.items())
        },
    }
