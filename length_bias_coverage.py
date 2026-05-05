import json
import os

from length_bias_metadata import maybe_file_sha256


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


def build_sample_coverage(
    parsed_rows,
    screening_summary_path,
    screened_path,
    padded_path,
    trials_path,
):
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
        coverage["screening_category_counts"] = screening_summary.get(
            "counts_by_category", {}
        )
    coverage["attrition_interpretation"] = (
        f"{coverage.get('screening_total_rows', screened_rows)} screened rows -> "
        f"{coverage.get('screening_eligible_rows', padded_rows)} eligible rows -> "
        f"{len(parsed_questions)} analyzed questions."
    )
    return coverage
