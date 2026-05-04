from pathlib import Path

from length_bias_common import read_jsonl, utc_now, word_count
from length_bias_metadata import file_sha256, run_metadata, text_sha256


BIAS_TYPE = "position"
DEFAULT_QUESTIONS_PATH = (
    "FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl"
)
DEFAULT_ANSWERS_DIR = "FastChat/data/mt_bench/model_answer"
DEFAULT_SOURCE_MODEL_A = "gpt-4"
DEFAULT_SOURCE_MODEL_B = "gpt-3.5-turbo"
DEFAULT_OUTPUT_PATH = "position_bias_trials.jsonl"


def answer_file_path(answers_dir, model_id):
    return Path(answers_dir) / f"{model_id}.jsonl"


def load_questions(path):
    questions = {}
    for row in read_jsonl(path):
        question_id = row["question_id"]
        questions[question_id] = {
            "question_id": question_id,
            "category": row.get("category"),
            "question_turns": row.get("turns", []),
        }
    return questions


def answer_turns(row):
    choices = row.get("choices") or []
    if not choices:
        return []
    turns = choices[0].get("turns") or []
    return [str(turn) for turn in turns]


def answer_text(row):
    return "\n\n".join(answer_turns(row))


def load_model_answers(answers_dir, model_id):
    path = answer_file_path(answers_dir, model_id)
    answers = {}
    for row in read_jsonl(path):
        text = answer_text(row)
        if text:
            answers[row["question_id"]] = {
                "answer_id": row.get("answer_id"),
                "model_id": row.get("model_id", model_id),
                "answer_turns": answer_turns(row),
                "answer": text,
                "tstamp": row.get("tstamp"),
            }
    return answers


def base_trial(
    question,
    source_model_a,
    source_model_b,
    source_answer_a,
    source_answer_b,
    condition,
):
    model_a_first = condition == "model_a_A"
    generated_at = utc_now()
    answer_a = source_answer_a["answer"] if model_a_first else source_answer_b["answer"]
    answer_b = source_answer_b["answer"] if model_a_first else source_answer_a["answer"]
    answer_a_turns = (
        source_answer_a["answer_turns"] if model_a_first else source_answer_b["answer_turns"]
    )
    answer_b_turns = (
        source_answer_b["answer_turns"] if model_a_first else source_answer_a["answer_turns"]
    )

    return {
        "bias_type": BIAS_TYPE,
        "source_model_a": source_model_a,
        "source_model_b": source_model_b,
        "model_a_position": "A" if model_a_first else "B",
        "condition": condition,
        "prompt_condition": "neutral_no_length",
        "question_id": question["question_id"],
        "category": question.get("category"),
        "question_turns": question.get("question_turns", []),
        "answer_a": answer_a,
        "answer_b": answer_b,
        "answer_a_turns": answer_a_turns,
        "answer_b_turns": answer_b_turns,
        "source_model_for_answer_a": source_model_a if model_a_first else source_model_b,
        "source_model_for_answer_b": source_model_b if model_a_first else source_model_a,
        "source_answer_id_a": (
            source_answer_a.get("answer_id")
            if model_a_first
            else source_answer_b.get("answer_id")
        ),
        "source_answer_id_b": (
            source_answer_b.get("answer_id")
            if model_a_first
            else source_answer_a.get("answer_id")
        ),
        "answer_a_word_count": word_count(answer_a),
        "answer_b_word_count": word_count(answer_b),
        "source_model_a_word_count": word_count(source_answer_a["answer"]),
        "source_model_b_word_count": word_count(source_answer_b["answer"]),
        "question_turn_count": len(question.get("question_turns", [])),
        "answer_turn_count_a": len(answer_a_turns),
        "answer_turn_count_b": len(answer_b_turns),
        "question_hash": text_sha256("\n\n".join(question.get("question_turns", []))),
        "answer_a_hash": text_sha256(answer_a),
        "answer_b_hash": text_sha256(answer_b),
        "generated_at": generated_at,
        "created_at": generated_at,
    }


def build_position_trials(questions, answers_a, answers_b, source_model_a, source_model_b):
    trials = []
    skipped = []

    for question_id in sorted(questions):
        question = questions[question_id]
        source_answer_a = answers_a.get(question_id)
        source_answer_b = answers_b.get(question_id)
        if not source_answer_a or not source_answer_b:
            skipped.append(
                {
                    "question_id": question_id,
                    "category": question.get("category"),
                    "reason": "missing_answer",
                    "has_source_model_a": source_answer_a is not None,
                    "has_source_model_b": source_answer_b is not None,
                }
            )
            continue

        for condition in ("model_a_A", "model_a_B"):
            trial = base_trial(
                question,
                source_model_a,
                source_model_b,
                source_answer_a,
                source_answer_b,
                condition,
            )
            trial["trial_id"] = (
                f"q{question_id}_{BIAS_TYPE}_{source_model_a}_vs_"
                f"{source_model_b}_{condition}"
            )
            trials.append(trial)

    return trials, skipped


def preparation_metadata(
    questions_path,
    answers_dir,
    source_model_a,
    source_model_b,
    limit=None,
):
    answer_path_a = answer_file_path(answers_dir, source_model_a)
    answer_path_b = answer_file_path(answers_dir, source_model_b)
    return run_metadata(
        input_path=str(questions_path),
        extra={
            "bias_type": BIAS_TYPE,
            "source_model_a": source_model_a,
            "source_model_b": source_model_b,
            "questions_path": str(questions_path),
            "questions_sha256": file_sha256(questions_path),
            "answers_dir": str(answers_dir),
            "source_model_a_answers_path": str(answer_path_a),
            "source_model_a_answers_sha256": file_sha256(answer_path_a),
            "source_model_b_answers_path": str(answer_path_b),
            "source_model_b_answers_sha256": file_sha256(answer_path_b),
            "limit": limit,
        },
    )


def attach_metadata(trials, metadata):
    for trial in trials:
        trial["run_metadata"] = metadata
    return trials
