import json
from pathlib import Path


def join_turns(turns):
    return "\n".join(turns).strip()


def resolve_input_format(path, input_format):
    if input_format != "auto":
        return input_format
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    return "txt"


def load_padding_samples(path, input_format="auto"):
    resolved_format = resolve_input_format(path, input_format)
    if resolved_format == "jsonl":
        return parse_jsonl_samples(path)
    if resolved_format == "txt":
        return parse_txt_samples(path)
    raise ValueError(f"Unsupported input format: {input_format}")


def parse_jsonl_samples(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("eligibility") and row.get("eligibility") != "eligible":
                continue
            samples.append(normalize_jsonl_sample(row, line_number))
    return samples


def normalize_jsonl_sample(row, line_number):
    question_turns = row.get("question_turns")
    if not isinstance(question_turns, list) or not question_turns:
        question = row.get("question")
        if not question:
            raise ValueError(f"JSONL row {line_number} is missing question_turns")
        question_turns = [question]

    answer_turns = row.get("original_answer_turns") or row.get("answer_turns")
    if not isinstance(answer_turns, list) or not answer_turns:
        answer = row.get("original_answer")
        if not answer:
            raise ValueError(
                f"JSONL row {line_number} is missing original_answer_turns"
            )
        answer_turns = [answer]

    sample = dict(row)
    sample["question_turns"] = question_turns
    sample["original_answer_turns"] = answer_turns
    sample["question"] = join_turns(question_turns)
    sample["original_answer"] = join_turns(answer_turns)
    return sample


def parse_txt_samples(path):
    text = read_text(path)
    blocks = []
    current = []

    for line in text.splitlines():
        if line.strip() == "##":
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)

    if current:
        blocks.append(current)

    samples = []
    for block in blocks:
        sample = parse_txt_block(block)
        if sample:
            samples.append(sample)
    return samples


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_txt_block(lines):
    question_id = None
    category = None
    question_start = None
    answer_start = None

    for idx, line in enumerate(lines):
        if line.startswith("[question_id]:"):
            question_id = int(line.split(":", 1)[1].strip())
        elif line.startswith("[category]:"):
            category = line.split(":", 1)[1].strip()
        elif line.strip() == "[question]:":
            question_start = idx + 1
        elif line.strip() == "[answer]:":
            answer_start = idx + 1

    if question_id is None or category is None:
        return None
    if question_start is None or answer_start is None or question_start > answer_start:
        raise ValueError(f"Malformed block for question_id {question_id}")

    question = "\n".join(lines[question_start:answer_start - 1]).strip()
    answer = "\n".join(lines[answer_start:]).strip()

    return {
        "question_id": question_id,
        "category": category,
        "question": question,
        "original_answer": answer,
    }
