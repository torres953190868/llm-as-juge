import argparse
from pathlib import Path

from length_bias_common import write_jsonl
from length_bias_position import (
    DEFAULT_ANSWERS_DIR,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_QUESTIONS_PATH,
    DEFAULT_SOURCE_MODEL_A,
    DEFAULT_SOURCE_MODEL_B,
    attach_metadata,
    build_position_trials,
    load_model_answers,
    load_questions,
    preparation_metadata,
)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Build swapped A/B position-bias trials from MT-Bench answers."
    )
    parser.add_argument("--source-model-a", default=DEFAULT_SOURCE_MODEL_A)
    parser.add_argument("--source-model-b", default=DEFAULT_SOURCE_MODEL_B)
    parser.add_argument("--questions-path", default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--answers-dir", default=DEFAULT_ANSWERS_DIR)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def limited_questions(questions, limit):
    if limit is None:
        return questions
    limited_ids = sorted(questions)[:limit]
    return {question_id: questions[question_id] for question_id in limited_ids}


def validate_inputs(questions_path, answers_dir, source_model_a, source_model_b):
    missing = []
    paths = [
        Path(questions_path),
        Path(answers_dir) / f"{source_model_a}.jsonl",
        Path(answers_dir) / f"{source_model_b}.jsonl",
    ]
    for path in paths:
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing required input file(s): " + ", ".join(missing))


def run(args):
    validate_inputs(
        args.questions_path,
        args.answers_dir,
        args.source_model_a,
        args.source_model_b,
    )

    questions = load_questions(args.questions_path)
    questions = limited_questions(questions, args.limit)
    answers_a = load_model_answers(args.answers_dir, args.source_model_a)
    answers_b = load_model_answers(args.answers_dir, args.source_model_b)
    trials, skipped = build_position_trials(
        questions,
        answers_a,
        answers_b,
        args.source_model_a,
        args.source_model_b,
    )
    metadata = preparation_metadata(
        args.questions_path,
        args.answers_dir,
        args.source_model_a,
        args.source_model_b,
        args.limit,
    )
    attach_metadata(trials, metadata)

    print(f"Loaded {len(questions)} questions")
    print(f"Loaded {len(answers_a)} answers for {args.source_model_a}")
    print(f"Loaded {len(answers_b)} answers for {args.source_model_b}")
    print(f"Built {len(trials)} position-bias trials")
    print(f"Skipped {len(skipped)} questions")

    if trials:
        first = trials[0]
        print(
            "First trial: "
            f"{first['trial_id']} model_a_position={first['model_a_position']}"
        )
    if skipped:
        first_skipped = skipped[0]
        print(
            "First skipped: "
            f"q{first_skipped['question_id']} {first_skipped['reason']}"
        )

    if args.dry_run:
        print("Dry run: no output written")
        return trials, skipped

    write_jsonl(args.output_path, trials)
    print(f"Wrote {len(trials)} trials to {args.output_path}")
    return trials, skipped


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
