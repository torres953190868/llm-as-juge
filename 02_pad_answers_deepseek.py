import argparse
import concurrent.futures
import threading

from length_bias_common import (
    DEEPSEEK_API_KEY_ENV,
    DEEPSEEK_ENDPOINT,
    append_jsonl,
    call_with_retries,
    error_details,
    extract_chat_content,
    get_api_key,
    length_ratio,
    utc_now,
    word_count,
)
from length_bias_metadata import maybe_file_sha256
from length_bias_padding_metadata import (
    make_accepted_run_metadata,
    make_attempt_metadata,
    make_base_run_metadata,
)
from length_bias_padding_prompt import build_payload
from length_bias_padding_io import (
    append_padding_txt,
    load_compatible_completed_ids,
    truncate_outputs,
)
from length_bias_samples import join_turns, load_padding_samples


DEFAULT_INPUT = "length_bias_eligible_samples.jsonl"
DEFAULT_OUTPUT_JSONL = "mt_bench_questions_answers_padded_deepseek.jsonl"
DEFAULT_OUTPUT_TXT = "mt_bench_questions_answers_padded_deepseek.txt"
DEFAULT_RAW_OUTPUT = "raw_deepseek_padding_responses.jsonl"
DEFAULT_FAILED_OUTPUT = "failed_deepseek_padding.jsonl"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_ENV_FILE = ".env"
API_KEY_ENV = DEEPSEEK_API_KEY_ENV
PROMPT_VERSION = "deepseek_padding_v5_retry_direction"
DEFAULT_MIN_RATIO = 1.3
DEFAULT_MAX_RATIO = 2.0
DEFAULT_PADDING_ATTEMPTS = 3
DEFAULT_MAX_TOKENS = 4096
DEFAULT_PARALLEL = 3


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Generate length-padded MT-Bench answers with the DeepSeek API."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument(
        "--input-format",
        choices=("auto", "txt", "jsonl"),
        default="auto",
    )
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--output-txt", default=DEFAULT_OUTPUT_TXT)
    parser.add_argument("--raw-output", default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--failed-output", default=DEFAULT_FAILED_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-ratio", type=float, default=DEFAULT_MIN_RATIO)
    parser.add_argument("--max-ratio", type=float, default=DEFAULT_MAX_RATIO)
    parser.add_argument("--padding-attempts", type=int, default=DEFAULT_PADDING_ATTEMPTS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def make_result_row(
    sample,
    padded_answer,
    model,
    padded_answer_turns=None,
    run_metadata_value=None,
):
    original_answer = sample["original_answer"]
    ratio = length_ratio(original_answer, padded_answer)
    generated_at = utc_now()
    row = {
        "question_id": sample["question_id"],
        "category": sample["category"],
        "question": sample["question"],
        "original_answer": original_answer,
        "padded_answer": padded_answer,
        "model": model,
        "generated_at": generated_at,
        "created_at": generated_at,
        "original_word_count": word_count(original_answer),
        "padded_word_count": word_count(padded_answer),
        "original_char_count": len(original_answer),
        "padded_char_count": len(padded_answer),
        "length_ratio": ratio,
        "padding_prompt_version": PROMPT_VERSION,
    }
    if run_metadata_value is not None:
        row["run_metadata"] = run_metadata_value
    if "question_turns" in sample:
        row["question_turns"] = sample["question_turns"]
    if "original_answer_turns" in sample:
        original_turns = sample["original_answer_turns"]
        row["original_answer_turns"] = original_turns
        row["original_turn_word_counts"] = [
            word_count(turn) for turn in original_turns
        ]
    if padded_answer_turns is not None:
        row["padded_answer_turns"] = padded_answer_turns
        row["padded_turn_word_counts"] = [
            word_count(turn) for turn in padded_answer_turns
        ]
    return row


def completed_ids_for_run(args, base_metadata):
    if args.overwrite:
        return set()
    try:
        return load_compatible_completed_ids(args.output_jsonl, base_metadata)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def length_failure_reason(row, min_ratio, max_ratio):
    ratio = row["length_ratio"]
    if ratio < min_ratio:
        return "below_min_length_ratio"
    if ratio > max_ratio:
        return "above_max_length_ratio"
    return None


def has_structured_answer_turns(sample):
    turns = sample.get("original_answer_turns")
    return isinstance(turns, list) and bool(turns)


def sample_answer_turns(sample):
    if has_structured_answer_turns(sample):
        return sample["original_answer_turns"]
    return [sample["original_answer"]]


def add_turn_metadata(row, turn_index, turn_count):
    if turn_index is None:
        return
    row["turn_index"] = turn_index
    row["turn_count"] = turn_count


def make_turn_sample(sample, answer_text):
    turn_sample = dict(sample)
    turn_sample["original_answer"] = answer_text
    return turn_sample


def pad_answer_text(
    sample,
    answer_text,
    args,
    api_key,
    output_lock,
    base_metadata,
    turn_index=None,
    turn_count=None,
):
    last_failure = None
    last_padded_answer = None
    question_id = sample["question_id"]
    turn_sample = make_turn_sample(sample, answer_text)

    for attempt in range(1, args.padding_attempts + 1):
        payload = build_payload(
            turn_sample, args.model, attempt, args.min_ratio, args.max_ratio,
            args.max_tokens, last_failure, last_padded_answer,
        )
        attempt_metadata = make_attempt_metadata(
            args,
            payload,
            base_metadata,
            PROMPT_VERSION,
            attempt,
            turn_index,
            turn_count,
        )
        raw_row = {
            "question_id": question_id,
            "model": args.model,
            "attempt": attempt,
            "generated_at": attempt_metadata["generated_at"],
            "created_at": attempt_metadata["created_at"],
            "padding_prompt_version": PROMPT_VERSION,
        }
        add_turn_metadata(raw_row, turn_index, turn_count)
        raw_row["run_metadata"] = attempt_metadata

        try:
            response = call_with_retries(DEEPSEEK_ENDPOINT, payload, api_key)
            padded_answer = extract_chat_content(response)
            result_row = make_result_row(turn_sample, padded_answer, args.model)
            failure = length_failure_reason(result_row, args.min_ratio, args.max_ratio)
            raw_row["response"] = response
            raw_row["length_ratio"] = result_row["length_ratio"]
            raw_row["original_word_count"] = result_row["original_word_count"]
            raw_row["padded_word_count"] = result_row["padded_word_count"]
            raw_row["accepted"] = failure is None
            with output_lock:
                append_jsonl(args.raw_output, raw_row)

            if failure is None:
                return padded_answer, None, attempt_metadata

            last_failure = {
                "question_id": question_id,
                "category": turn_sample["category"],
                "attempt": attempt,
                "failed_reason": failure,
                "length_ratio": result_row["length_ratio"],
                "original_word_count": result_row["original_word_count"],
                "padded_word_count": result_row["padded_word_count"],
                "generated_at": attempt_metadata["generated_at"],
                "created_at": attempt_metadata["created_at"],
                "padding_prompt_version": PROMPT_VERSION,
                "run_metadata": attempt_metadata,
            }
            add_turn_metadata(last_failure, turn_index, turn_count)
            last_padded_answer = padded_answer
            print(
                f"Attempt {attempt} rejected for question_id {question_id}: "
                f"{failure} ({result_row['length_ratio']:.2f})"
            )
        except Exception as exc:
            details = error_details(exc)
            raw_row["error"] = details
            with output_lock:
                append_jsonl(args.raw_output, raw_row)
            last_failure = {
                "question_id": question_id,
                "category": turn_sample["category"],
                "attempt": attempt,
                "failed_reason": "api_error",
                "error": details,
                "generated_at": attempt_metadata["generated_at"],
                "created_at": attempt_metadata["created_at"],
                "padding_prompt_version": PROMPT_VERSION,
                "run_metadata": attempt_metadata,
            }
            add_turn_metadata(last_failure, turn_index, turn_count)
            print(f"Attempt {attempt} failed for question_id {question_id}: {exc}")

    return None, last_failure, None


def process_sample(sample, index, total, args, api_key, output_lock, base_metadata):
    question_id = sample["question_id"]
    turns = sample_answer_turns(sample)
    structured_turns = has_structured_answer_turns(sample)
    print(f"[{index}/{total}] Padding question_id {question_id}")

    padded_turns = []
    turn_metadatas = []
    for turn_index, answer_text in enumerate(turns, start=1):
        if structured_turns:
            print(f"Padding question_id {question_id} turn {turn_index}/{len(turns)}")
        padded_answer, failure, turn_metadata = pad_answer_text(
            sample,
            answer_text,
            args,
            api_key,
            output_lock,
            base_metadata,
            turn_index if structured_turns else None,
            len(turns) if structured_turns else None,
        )
        if failure:
            with output_lock:
                append_jsonl(args.failed_output, failure)
            return f"Failed question_id {question_id}: no acceptable padded answer"
        padded_turns.append(padded_answer)
        turn_metadatas.append(turn_metadata)

    padded_answer = join_turns(padded_turns) if structured_turns else padded_turns[0]
    accepted_metadata = make_accepted_run_metadata(
        args, base_metadata, turn_metadatas, PROMPT_VERSION
    )
    accepted_row = make_result_row(
        sample,
        padded_answer,
        args.model,
        padded_turns if structured_turns else None,
        accepted_metadata,
    )
    with output_lock:
        append_jsonl(args.output_jsonl, accepted_row)
        append_padding_txt(args.output_txt, accepted_row)
    return f"Accepted question_id {question_id}"


def run(args):
    samples = load_padding_samples(args.input, args.input_format)
    if args.limit is not None:
        samples = samples[:args.limit]
    base_metadata = make_base_run_metadata(args, PROMPT_VERSION)

    if args.dry_run:
        ids = [sample["question_id"] for sample in samples]
        print(f"Parsed {len(samples)} samples from {args.input}")
        print(f"Input SHA256: {maybe_file_sha256(args.input)}")
        print("Question IDs: " + ", ".join(str(item) for item in ids))
        if samples:
            first = samples[0]
            q_turns = len(first.get("question_turns", [first["question"]]))
            a_turns = len(sample_answer_turns(first))
            print(f"First sample turns: question={q_turns}, answer={a_turns}")
        return

    completed_ids = completed_ids_for_run(args, base_metadata)

    api_key = get_api_key(API_KEY_ENV, args.env_file)
    if not api_key:
        raise SystemExit(
            f"Missing {API_KEY_ENV}. Set it in the environment or in {args.env_file}."
        )

    if args.overwrite:
        truncate_outputs(
            [args.output_jsonl, args.output_txt, args.raw_output, args.failed_output]
        )

    total = len(samples)

    pending = [
        (index, sample)
        for index, sample in enumerate(samples, start=1)
        if sample["question_id"] not in completed_ids
    ]
    skipped = total - len(pending)
    if skipped:
        print(f"Skipping {skipped} completed sample(s)")

    output_lock = threading.Lock()
    max_workers = max(1, args.parallel)
    print(f"Running with parallel={max_workers}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_sample,
                sample,
                index,
                total,
                args,
                api_key,
                output_lock,
                base_metadata,
            )
            for index, sample in pending
        ]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
