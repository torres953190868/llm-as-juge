import json

from length_bias_common import truncate_file


PADDING_COMPATIBILITY_KEYS = ("model", "min_ratio", "max_ratio", "prompt_version")


def load_completed_ids(path):
    try:
        f = open(path, "r", encoding="utf-8")
    except FileNotFoundError:
        return set()

    completed = set()
    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "question_id" in row and row.get("padded_answer"):
                completed.add(int(row["question_id"]))
    return completed


def load_successful_padding_rows(path):
    try:
        f = open(path, "r", encoding="utf-8")
    except FileNotFoundError:
        return []

    rows = []
    with f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path} line {line_number} is not valid JSON."
                ) from exc
            if row.get("question_id") is not None and row.get("padded_answer"):
                rows.append(row)
    return rows


def padding_config_signature(metadata):
    config = metadata.get("padding_config") or {}
    return {key: config.get(key) for key in PADDING_COMPATIBILITY_KEYS}


def output_row_signature(row):
    metadata = row.get("run_metadata") or {}
    signature = {
        "input_path": metadata.get("input_path"),
        "input_sha256": metadata.get("input_sha256"),
    }
    signature.update(padding_config_signature(metadata))
    if signature["model"] is None:
        signature["model"] = metadata.get("model", row.get("model"))
    return signature


def format_signature(signature):
    fields = (
        "input_path",
        "input_sha256",
        "model",
        "min_ratio",
        "max_ratio",
        "prompt_version",
    )
    return ", ".join(
        f"{field}={signature.get(field) if signature.get(field) is not None else '<missing>'}"
        for field in fields
    )


def unique_signatures(rows):
    signatures = []
    seen = set()
    for row in rows:
        signature = output_row_signature(row)
        key = tuple(sorted(signature.items()))
        if key not in seen:
            signatures.append(signature)
            seen.add(key)
    return signatures


def load_compatible_completed_ids(path, expected_metadata):
    rows = load_successful_padding_rows(path)
    if not rows:
        return set()

    expected = {
        "input_path": expected_metadata.get("input_path"),
        "input_sha256": expected_metadata.get("input_sha256"),
    }
    expected.update(padding_config_signature(expected_metadata))

    mismatched_rows = [
        row for row in rows
        if output_row_signature(row) != expected
    ]
    if mismatched_rows:
        existing = "\n".join(
            f"  - {format_signature(signature)}"
            for signature in unique_signatures(rows)
        )
        raise ValueError(
            f"Refusing to append to {path} because existing padded rows were "
            "created from a different input or padding configuration.\n"
            f"Current run: {format_signature(expected)}\n"
            f"Existing output contains:\n{existing}\n"
            "Use --overwrite to replace the output, or pass a different "
            "--output-jsonl path."
        )

    return {int(row["question_id"]) for row in rows}


def truncate_outputs(paths):
    for path in paths:
        truncate_file(path)


def append_padding_txt(path, row):
    text = (
        "##\n"
        f"[question_id]: {row['question_id']}\n"
        f"[category]: {row['category']}\n"
        f"[question]:\n{row['question']}\n"
        f"[original_answer]:\n{row['original_answer']}\n"
        f"[padded_answer]:\n{row['padded_answer']}\n"
        f"[original_word_count]: {row['original_word_count']}\n"
        f"[padded_word_count]: {row['padded_word_count']}\n"
        "##\n\n"
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)
