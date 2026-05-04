import json
import os
import random
from collections import defaultdict
from itertools import combinations

from length_bias_metadata import maybe_file_sha256


DEFAULT_BOOTSTRAP_SEED = 20260504
DEFAULT_BOOTSTRAP_ITERATIONS = 5000


def safe_div(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


def length_preference_score(row):
    if row.get("winner") == "invalid":
        return None
    long_won = row.get("long_answer_won")
    if long_won is True:
        return 1.0
    if long_won is False:
        return -1.0
    if row.get("winner") == "tie":
        return 0.0
    return None


def mean(values):
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def percentile(sorted_values, q):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bootstrap_ci_by_cluster(cluster_values, seed, iterations):
    clusters = sorted(cluster_values)
    if not clusters:
        return {"seed": seed, "iterations": iterations, "lower": None, "upper": None}

    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        sampled_values = []
        for _ in clusters:
            cluster = rng.choice(clusters)
            sampled_values.extend(cluster_values[cluster])
        estimates.append(mean(sampled_values))

    estimates.sort()
    return {
        "seed": seed,
        "iterations": iterations,
        "lower": percentile(estimates, 0.025),
        "upper": percentile(estimates, 0.975),
    }


def summarize_cluster_scores(rows, cluster_key, seed, iterations):
    cluster_values = defaultdict(list)
    invalid_rows = 0
    for row in rows:
        score = length_preference_score(row)
        if score is None:
            invalid_rows += 1
            continue
        cluster_values[cluster_key(row)].append(score)

    valid_scores = [
        score
        for scores in cluster_values.values()
        for score in scores
    ]
    return {
        "clusters": len(cluster_values),
        "valid_observations": len(valid_scores),
        "invalid_or_unscored_rows": invalid_rows,
        "mean_net_length_preference": mean(valid_scores),
        "cluster_means": {
            str(key): mean(values)
            for key, values in sorted(cluster_values.items(), key=lambda item: str(item[0]))
        },
        "bootstrap_95_ci": bootstrap_ci_by_cluster(
            cluster_values, seed=seed, iterations=iterations
        ),
        "score_definition": "long win=1, short win=-1, tie=0, invalid/unparsed omitted",
    }


def summarize_swapped_paired_scores(rows, seed, iterations):
    by_pair = defaultdict(dict)
    for row in rows:
        key = (
            row.get("question_id"),
            row.get("judge_model", "unknown"),
            row.get("prompt_condition", "unknown"),
        )
        condition = row.get("condition")
        if condition in {"long_A", "long_B"}:
            by_pair[key][condition] = row

    pair_means_by_question = defaultdict(list)
    deltas_by_question = defaultdict(list)
    complete_pairs = 0
    skipped_pairs = 0

    for key, pair_rows in sorted(by_pair.items(), key=lambda item: str(item[0])):
        long_a_score = length_preference_score(pair_rows.get("long_A", {}))
        long_b_score = length_preference_score(pair_rows.get("long_B", {}))
        if long_a_score is None or long_b_score is None:
            skipped_pairs += 1
            continue
        question_id = key[0]
        complete_pairs += 1
        pair_means_by_question[question_id].append((long_a_score + long_b_score) / 2)
        deltas_by_question[question_id].append(long_a_score - long_b_score)

    pair_means = [value for values in pair_means_by_question.values() for value in values]
    deltas = [value for values in deltas_by_question.values() for value in values]
    return {
        "complete_pairs": complete_pairs,
        "skipped_pairs": skipped_pairs,
        "question_clusters": len(pair_means_by_question),
        "mean_paired_length_preference": mean(pair_means),
        "mean_position_delta_long_A_minus_long_B": mean(deltas),
        "length_preference_bootstrap_95_ci": bootstrap_ci_by_cluster(
            pair_means_by_question, seed=seed, iterations=iterations
        ),
        "position_delta_bootstrap_95_ci": bootstrap_ci_by_cluster(
            deltas_by_question, seed=seed, iterations=iterations
        ),
        "interpretation": (
            "Positive paired length preference means the long answer is favored after "
            "averaging each long_A/long_B swap. Positive position delta means the "
            "length preference is stronger when the long answer is in position A."
        ),
    }


def summarize_data_shape(rows):
    questions = {row.get("question_id") for row in rows if row.get("question_id") is not None}
    prompts = {row.get("prompt_condition") for row in rows if row.get("prompt_condition")}
    positions = {row.get("condition") for row in rows if row.get("condition")}
    judges = {row.get("judge_model") for row in rows if row.get("judge_model")}
    expected = len(questions) * len(prompts) * len(positions) * len(judges)
    return {
        "rows": len(rows),
        "questions": len(questions),
        "prompt_conditions": len(prompts),
        "positions": len(positions),
        "judges": len(judges),
        "full_factorial_expected_rows": expected,
        "is_full_factorial": len(rows) == expected,
        "interpretation": (
            f"{len(rows)} rows = {len(questions)} questions x {len(prompts)} prompts "
            f"x {len(positions)} positions x {len(judges)} judges for the current data."
        ),
    }


def build_statistical_summary(rows, seed=None, iterations=None):
    seed = DEFAULT_BOOTSTRAP_SEED if seed is None else seed
    iterations = DEFAULT_BOOTSTRAP_ITERATIONS if iterations is None else iterations
    return {
        "data_shape": summarize_data_shape(rows),
        "question_cluster": summarize_cluster_scores(
            rows,
            cluster_key=lambda row: row.get("question_id", "unknown"),
            seed=seed,
            iterations=iterations,
        ),
        "swapped_paired": summarize_swapped_paired_scores(
            rows, seed=seed, iterations=iterations
        ),
    }


def percent(value):
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def render_statistics_text(summary):
    lines = []
    stats = summary.get("statistical_analysis", {})
    shape = stats.get("data_shape", {})
    if shape.get("interpretation"):
        lines.append(shape["interpretation"])
    lines.append("")
    lines.append("Question-cluster statistics")
    lines.append("---------------------------")
    question_stats = stats.get("question_cluster", {})
    if question_stats:
        ci = question_stats.get("bootstrap_95_ci", {})
        lines.append(
            f"clusters={question_stats.get('clusters')}, "
            f"valid_observations={question_stats.get('valid_observations')}, "
            f"mean_net={percent(question_stats.get('mean_net_length_preference'))}, "
            f"bootstrap_95_ci=[{percent(ci.get('lower'))}, {percent(ci.get('upper'))}], "
            f"seed={ci.get('seed')}"
        )
        lines.append(question_stats.get("score_definition", ""))
    else:
        lines.append("n/a")
    lines.append("")
    lines.append("Paired swapped-position statistics")
    lines.append("----------------------------------")
    paired = stats.get("swapped_paired", {})
    if paired:
        pref_ci = paired.get("length_preference_bootstrap_95_ci", {})
        delta_ci = paired.get("position_delta_bootstrap_95_ci", {})
        lines.append(
            f"complete_pairs={paired.get('complete_pairs')}, "
            f"question_clusters={paired.get('question_clusters')}, "
            f"paired_mean={percent(paired.get('mean_paired_length_preference'))}, "
            f"paired_ci=[{percent(pref_ci.get('lower'))}, {percent(pref_ci.get('upper'))}]"
        )
        lines.append(
            f"position_delta_A_minus_B="
            f"{percent(paired.get('mean_position_delta_long_A_minus_long_B'))}, "
            f"delta_ci=[{percent(delta_ci.get('lower'))}, {percent(delta_ci.get('upper'))}]"
        )
        lines.append(paired.get("interpretation", ""))
    else:
        lines.append("n/a")
    coverage = summary.get("sample_coverage", {})
    if coverage:
        lines.append("")
        lines.append("Sample coverage")
        lines.append("---------------")
        lines.append(coverage.get("attrition_interpretation", "n/a"))
    return lines


def compare_judge_results(rows, max_disagreements=20):
    by_trial = defaultdict(dict)
    for row in rows:
        trial_id = row.get("trial_id")
        judge = row.get("judge_model", "unknown")
        if trial_id:
            by_trial[trial_id][judge] = row

    pair_counts = defaultdict(
        lambda: {
            "common_trials": 0,
            "winner_agreements": 0,
            "length_preference_agreements": 0,
        }
    )
    disagreements = []
    for trial_id, judge_rows in sorted(by_trial.items()):
        if len(judge_rows) < 2:
            continue
        judges = sorted(judge_rows)
        for left, right in combinations(judges, 2):
            left_row = judge_rows[left]
            right_row = judge_rows[right]
            counts = pair_counts[(left, right)]
            counts["common_trials"] += 1
            if left_row.get("winner") == right_row.get("winner"):
                counts["winner_agreements"] += 1
            if left_row.get("long_answer_won") == right_row.get("long_answer_won"):
                counts["length_preference_agreements"] += 1
        winners = {row.get("winner") for row in judge_rows.values()}
        length_prefs = {row.get("long_answer_won") for row in judge_rows.values()}
        if (len(winners) > 1 or len(length_prefs) > 1) and len(disagreements) < max_disagreements:
            first = next(iter(judge_rows.values()))
            disagreements.append(
                {
                    "trial_id": trial_id,
                    "question_id": first.get("question_id"),
                    "category": first.get("category"),
                    "condition": first.get("condition"),
                    "prompt_condition": first.get("prompt_condition"),
                    "judges": {
                        judge: {
                            "winner": row.get("winner"),
                            "long_answer_won": row.get("long_answer_won"),
                        }
                        for judge, row in sorted(judge_rows.items())
                    },
                }
            )

    pairwise = {}
    for (left, right), counts in sorted(pair_counts.items()):
        common = counts["common_trials"]
        pairwise[f"{left}::{right}"] = {
            **counts,
            "winner_agreement_rate": safe_div(counts["winner_agreements"], common),
            "length_preference_agreement_rate": safe_div(
                counts["length_preference_agreements"], common
            ),
        }
    return {
        "pairwise_agreement": pairwise,
        "disagreements": disagreements,
    }


def read_json_file(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_jsonl(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def file_record(path):
    return {
        "path": path,
        "exists": os.path.exists(path),
        "row_count": count_jsonl(path) if path.endswith(".jsonl") else None,
        "sha256": maybe_file_sha256(path),
    }


def build_sample_coverage(parsed_rows, screening_summary_path, screened_path, padded_path, trials_path):
    screening_summary = read_json_file(screening_summary_path)
    parsed_questions = {
        row.get("question_id")
        for row in parsed_rows
        if row.get("question_id") is not None
    }
    trial_rows = count_jsonl(trials_path)
    padded_rows = count_jsonl(padded_path)
    screened_rows = count_jsonl(screened_path)

    coverage = {
        "files": {
            "screening_summary": file_record(screening_summary_path),
            "screened_samples": file_record(screened_path),
            "padded_samples": file_record(padded_path),
            "trials": file_record(trials_path),
        },
        "parsed_unique_questions": len(parsed_questions),
        "parsed_rows": len(parsed_rows),
        "trial_rows": trial_rows,
        "padded_rows": padded_rows,
        "screened_rows": screened_rows,
    }
    if screening_summary:
        counts = screening_summary.get("counts_by_status", {})
        coverage["screening_total_rows"] = screening_summary.get("total_rows")
        coverage["screening_eligible_rows"] = counts.get("eligible")
        coverage["screening_status_counts"] = counts
        coverage["screening_category_counts"] = screening_summary.get("counts_by_category", {})
    coverage["attrition_interpretation"] = (
        f"{coverage.get('screening_total_rows', screened_rows)} screened rows -> "
        f"{coverage.get('screening_eligible_rows', padded_rows)} eligible rows -> "
        f"{len(parsed_questions)} analyzed questions."
    )
    return coverage
