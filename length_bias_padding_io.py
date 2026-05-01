import json

from length_bias_common import truncate_file


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
