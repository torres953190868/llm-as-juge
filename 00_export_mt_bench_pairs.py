import json
from pathlib import Path


QUESTION_FILE = Path("FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl")
ANSWER_FILE = Path("FastChat/data/mt_bench/model_answer/gpt-4.jsonl")
OUTPUT_FILE = Path("mt_bench_questions_and_answers.txt")
TARGET_QUESTION_IDS = {
    96, 97, 100,
    141, 143, 146, 149, 150,
    151, 153, 154, 155, 156, 157, 158, 159, 160,
}


with QUESTION_FILE.open(encoding="utf-8") as f:
    questions = [json.loads(line) for line in f if line.strip()]

with ANSWER_FILE.open(encoding="utf-8") as f:
    answers = [json.loads(line) for line in f if line.strip()]

question_map = {item["question_id"]: item for item in questions}
answer_map = {item["question_id"]: item for item in answers}

lines = []
for question_id in sorted(TARGET_QUESTION_IDS):
    question = question_map.get(question_id)
    if not question:
        print(f"Skipping missing question_id in questions: {question_id}")
        continue

    answer = answer_map.get(question_id)
    if not answer:
        print(f"Skipping missing question_id in answers: {question_id}")
        continue

    question_text = "\n".join(question["turns"])
    answer_text = "\n".join(answer["choices"][0]["turns"])

    lines.append(
        "##\n"
        f"[question_id]: {question_id}\n"
        f"[category]: {question['category']}\n"
        f"[question]:\n{question_text}\n"
        f"[answer]:\n{answer_text}\n"
        "##\n"
    )

OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {len(lines)} question-answer pairs to {OUTPUT_FILE}")
