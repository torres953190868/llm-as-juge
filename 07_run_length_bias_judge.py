import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from length_bias_common import DEEPSEEK_API_KEY_ENV, DEEPSEEK_ENDPOINT
from length_bias_common import append_jsonl, call_with_retries, error_details
from length_bias_common import extract_chat_content, get_api_key, read_jsonl
from length_bias_common import truncate_file, utc_now, write_jsonl
from length_bias_judge import DEEPSEEK_MODEL, build_payload, build_user_prompt
from length_bias_judge import load_judge_configs
from length_bias_judge import long_answer_won, parse_winner
from length_bias_metadata import maybe_file_sha256, prompt_hash_metadata
from length_bias_metadata import sanitize_judge_config


DEFAULT_TRIALS = "length_bias_trials.jsonl"
DEFAULT_RAW_OUTPUT = "raw_length_bias_judgments.jsonl"
DEFAULT_PARSED_OUTPUT = "parsed_length_bias_judgments.jsonl"
DEFAULT_MODEL = DEEPSEEK_MODEL
DEFAULT_ENV_FILE = ".env"


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run LLM judges on pairwise length-bias trials."
    )
    parser.add_argument("--trials", default=DEFAULT_TRIALS)
    parser.add_argument("--raw-output", default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--parsed-output", default=DEFAULT_PARSED_OUTPUT)
    parser.add_argument("--judge-config", default=None)
    parser.add_argument("--judge-model", default=DEFAULT_MODEL)
    parser.add_argument("--deepseek", type=int, choices=(0, 1), default=0)
    parser.add_argument("--gemini", type=int, choices=(0, 1), default=1)
    parser.add_argument("--xiaomi", type=int, choices=(0, 1), default=1)
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--parallel", type=int, default=3)
    return parser


def completed_keys(path):
    keys = set()
    for row in read_jsonl(path):
        if row.get("winner") != "invalid":
            keys.add((row.get("trial_id"), row.get("judge_model")))
    return keys


def remove_invalid_parsed_rows(path, retry_keys):
    if not retry_keys:
        return 0

    kept = []
    removed = 0
    for row in read_jsonl(path):
        key = (row.get("trial_id"), row.get("judge_model"))
        if key in retry_keys and row.get("winner") == "invalid":
            removed += 1
            continue
        kept.append(row)
    if removed:
        write_jsonl(path, kept)
    return removed


def prompt_metadata(payload):
    messages = payload.get("messages", [])
    system_text = "\n\n".join(
        message.get("content", "")
        for message in messages
        if message.get("role") == "system"
    )
    user_text = "\n\n".join(
        message.get("content", "")
        for message in messages
        if message.get("role") == "user"
    )
    return prompt_hash_metadata(system_text, user_text)


def judgment_metadata(config, payload, trials_path=None, trials_sha256=None):
    generated_at = utc_now()
    return {
        "generated_at": generated_at,
        "created_at": generated_at,
        "judge_config": sanitize_judge_config(config),
        "model": config.get("model", config["judge_model"]),
        "judge_model": config["judge_model"],
        "temperature": config.get("temperature", 0.0),
        "endpoint": config.get("endpoint", DEEPSEEK_ENDPOINT),
        "input_path": trials_path,
        "input_sha256": trials_sha256,
        "trials_path": trials_path,
        "trials_sha256": trials_sha256,
        **prompt_metadata(payload),
    }


def position_source_winner(trial, winner):
    if winner in ("invalid", "tie"):
        return None
    if winner == trial.get("model_a_position"):
        return trial.get("source_model_a")
    return trial.get("source_model_b")


def make_outcome_fields(trial, winner):
    bias_type = trial.get("bias_type", "length")
    if bias_type == "position":
        return {
            "model_a_position": trial.get("model_a_position"),
            "source_model_a": trial.get("source_model_a"),
            "source_model_b": trial.get("source_model_b"),
            "model_a_won": (
                None if winner in ("invalid", "tie")
                else winner == trial.get("model_a_position")
            ),
            "position_winner": winner if winner in ("A", "B") else None,
            "source_model_winner": position_source_winner(trial, winner),
        }
    return {
        "long_answer_won": long_answer_won(
            winner, trial["long_answer_position"]
        ),
    }


def make_parsed_row(trial, judge_model, winner, raw_ref, metadata):
    generated_at = utc_now()
    row = {
        "trial_id": trial["trial_id"],
        "judge_model": judge_model,
        "bias_type": trial.get("bias_type", "length"),
        "question_id": trial["question_id"],
        "category": trial["category"],
        "condition": trial["condition"],
        "prompt_condition": trial["prompt_condition"],
        "winner": winner,
        "raw_output_ref": raw_ref,
        "run_metadata": metadata,
        "generated_at": generated_at,
        "created_at": generated_at,
    }
    row.update(make_outcome_fields(trial, winner))
    return row


def validate_api_keys(configs, args):
    api_keys = {}
    for config in configs:
        judge_model = config["judge_model"]
        api_key_env = config.get("api_key_env", DEEPSEEK_API_KEY_ENV)
        api_key = get_api_key(api_key_env, args.env_file)
        if not api_key:
            raise SystemExit(
                f"Missing {api_key_env}. Set it in the environment or in {args.env_file}."
            )
        api_keys[judge_model] = api_key
    return api_keys


def log_result(result, stats_by_judge, args):
    stats = stats_by_judge[result["judge_model"]]
    stats[result["status"]] += 1
    append_jsonl(args.raw_output, result["raw_row"])
    append_jsonl(args.parsed_output, result["parsed_row"])
    if result["status"] == "failed":
        print(f"[{result['index']}/{result['total']}] {result['trial_id']} {result['judge_model']} failed: {result['error']}")
    elif args.verbose:
        print(f"[{result['index']}/{result['total']}] {result['trial_id']} {result['judge_model']} -> {result['winner']}")


def judge_trial(config, trial, index, total, api_key, trials_path, trials_sha256):
    judge_model = config["judge_model"]
    generated_at = utc_now()
    raw_ref = f"{judge_model}:{trial['trial_id']}:{generated_at}"
    raw_row = {
        "raw_output_ref": raw_ref,
        "trial_id": trial["trial_id"],
        "judge_model": judge_model,
        "bias_type": trial.get("bias_type", "length"),
        "generated_at": generated_at,
        "created_at": generated_at,
    }
    metadata = None
    try:
        payload = build_payload(trial, config)
        metadata = judgment_metadata(config, payload, trials_path, trials_sha256)
        raw_row["run_metadata"] = metadata
        endpoint = config.get("endpoint", DEEPSEEK_ENDPOINT)
        response = call_with_retries(endpoint, payload, api_key)
        content = extract_chat_content(response)
        winner = parse_winner(content)
        raw_row["response"] = response
        status = "completed"
    except Exception as exc:
        winner = "invalid"
        raw_row["error"] = error_details(exc)
        status = "failed"
        error = exc
    else:
        error = None

    return {
        "status": status,
        "index": index,
        "total": total,
        "trial_id": trial["trial_id"],
        "judge_model": judge_model,
        "winner": winner,
        "error": error,
        "raw_row": raw_row,
        "parsed_row": make_parsed_row(trial, judge_model, winner, raw_ref, metadata),
    }


def run(args):
    if args.parallel < 1:
        raise SystemExit("--parallel must be at least 1.")

    trials = read_jsonl(args.trials)
    if args.limit is not None:
        trials = trials[: args.limit]

    configs = load_judge_configs(args)
    if not configs:
        raise SystemExit(
            "No judge models selected. Set at least one of --deepseek, --gemini, "
            "or --xiaomi to 1, or provide --judge-config."
        )

    judge_names = [config["judge_model"] for config in configs]
    print(
        f"Judging {len(trials)} trial(s) with: {', '.join(judge_names)} "
        f"(parallel={args.parallel})"
    )

    if not trials:
        print("No trials to judge")
        return

    if args.dry_run:
        first = trials[0]
        print(f"Dry run first trial: trial_id={first.get('trial_id')}, category={first.get('category')}, prompt_condition={first.get('prompt_condition')}")
        if args.verbose:
            print("First trial prompt preview:")
            print(build_user_prompt(trials[0])[:1000])
        return

    if args.overwrite:
        truncate_file(args.raw_output)
        truncate_file(args.parsed_output)

    done = set() if args.overwrite else completed_keys(args.parsed_output)
    api_keys = validate_api_keys(configs, args)
    stats_by_judge = {
        config["judge_model"]: {"completed": 0, "skipped": 0, "failed": 0}
        for config in configs
    }
    trials_sha256 = maybe_file_sha256(args.trials)
    tasks = []
    retry_keys = set()
    for config in configs:
        judge_model = config["judge_model"]
        stats = stats_by_judge[judge_model]
        api_key = api_keys[judge_model]
        for index, trial in enumerate(trials, start=1):
            key = (trial["trial_id"], judge_model)
            if key in done:
                stats["skipped"] += 1
                if args.verbose:
                    print(f"[{index}/{len(trials)}] Skipping {trial['trial_id']} {judge_model}")
                continue
            retry_keys.add(key)
            tasks.append((config, trial, index, len(trials), api_key))

    removed = remove_invalid_parsed_rows(args.parsed_output, retry_keys)
    if removed:
        print(f"Removed {removed} previous invalid parsed row(s) before retrying")

    executor = ThreadPoolExecutor(max_workers=args.parallel)
    futures = [
        executor.submit(
            judge_trial,
            config,
            trial,
            index,
            total,
            api_key,
            args.trials,
            trials_sha256,
        )
        for config, trial, index, total, api_key in tasks
    ]
    print(f"Submitted {len(futures)} request(s)")
    handled = set()
    interrupted = False
    try:
        for future in as_completed(futures):
            log_result(future.result(), stats_by_judge, args)
            handled.add(future)
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted. Saving finished requests and cancelling pending work...")
        for future in futures:
            if future in handled:
                continue
            if future.done() and not future.cancelled():
                try:
                    log_result(future.result(), stats_by_judge, args)
                except Exception as exc:
                    print(f"Could not save a finished request: {exc}")
            else:
                future.cancel()
    finally:
        executor.shutdown(wait=not interrupted, cancel_futures=interrupted)

    total_stats = {"completed": 0, "skipped": 0, "failed": 0}
    for judge_model, stats in stats_by_judge.items():
        print(f"{judge_model}: completed={stats['completed']}, skipped={stats['skipped']}, failed={stats['failed']}")
        for key, value in stats.items():
            total_stats[key] += value
    print(f"Total: completed={total_stats['completed']}, skipped={total_stats['skipped']}, failed={total_stats['failed']}")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
