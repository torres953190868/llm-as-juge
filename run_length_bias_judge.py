import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from length_bias_common import (
    DEEPSEEK_API_KEY_ENV,
    DEEPSEEK_ENDPOINT,
    append_jsonl,
    call_with_retries,
    error_details,
    extract_chat_content,
    get_api_key,
    read_jsonl,
    truncate_file,
    utc_now,
    write_jsonl,
)


DEFAULT_TRIALS = "length_bias_trials.jsonl"
DEFAULT_RAW_OUTPUT = "raw_length_bias_judgments.jsonl"
DEFAULT_PARSED_OUTPUT = "parsed_length_bias_judgments.jsonl"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_ENV_FILE = ".env"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL = "gemini-3-flash-preview"
XIAOMI_ENDPOINT = "https://token-plan-cn.xiaomimimo.com/v1/chat/completions"
XIAOMI_API_KEY_ENV = "XIAOMI_API_KEY"
XIAOMI_MODEL = "mimo-v2-pro"


STANDARD_SYSTEM_PROMPT = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail. Avoid position bias and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible. After providing a short explanation, output your final verdict strictly as [[A]] if assistant A is better, [[B]] if assistant B is better, and [[C]] for a tie."""


NEUTRAL_SYSTEM_PROMPT = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail. Avoid position bias and ensure that the order in which the responses were presented does not influence your decision. Be as objective as possible. After providing a short explanation, output your final verdict strictly as [[A]] if assistant A is better, [[B]] if assistant B is better, and [[C]] for a tie."""


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


def deepseek_judge_config(model):
    return {
        "judge_model": model,
        "model": model,
        "endpoint": DEEPSEEK_ENDPOINT,
        "api_key_env": DEEPSEEK_API_KEY_ENV,
        "temperature": 0.0,
        "max_tokens": 4096,
        "extra_body": {"thinking": {"type": "disabled"}},
    }


def gemini_judge_config():
    return {
        "judge_model": GEMINI_MODEL,
        "model": GEMINI_MODEL,
        "endpoint": GEMINI_ENDPOINT,
        "api_key_env": GEMINI_API_KEY_ENV,
        "temperature": 1.0,
        "max_tokens": 4096,
        "extra_body": {"reasoning_effort": "low"},
    }


def xiaomi_judge_config():
    return {
        "judge_model": XIAOMI_MODEL,
        "model": XIAOMI_MODEL,
        "endpoint": XIAOMI_ENDPOINT,
        "api_key_env": XIAOMI_API_KEY_ENV,
        "temperature": 0.0,
        "max_tokens": 4096,
    }


def builtin_judge_configs(args):
    configs = []
    if args.deepseek:
        configs.append(deepseek_judge_config(args.judge_model))
    if args.gemini:
        configs.append(gemini_judge_config())
    if args.xiaomi:
        configs.append(xiaomi_judge_config())
    return configs


def load_judge_configs(args):
    path = args.judge_config
    if not path:
        return builtin_judge_configs(args)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    configs = data.get("judges", data) if isinstance(data, dict) else data
    if not isinstance(configs, list) or not configs:
        raise ValueError("judge config must be a non-empty list or {'judges': [...]}")
    return configs


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


def system_prompt(prompt_condition):
    if prompt_condition == "standard_anti_length":
        return STANDARD_SYSTEM_PROMPT
    if prompt_condition == "neutral_no_length":
        return NEUTRAL_SYSTEM_PROMPT
    raise ValueError(f"Unknown prompt_condition: {prompt_condition}")


def render_question(turns):
    if len(turns) == 1:
        return turns[0]
    return "\n\n".join(
        f"Turn {index}:\n{text}" for index, text in enumerate(turns, start=1)
    )


def build_user_prompt(trial):
    return (
        "[User Question]\n"
        f"{render_question(trial['question_turns'])}\n\n"
        "[The Start of Assistant A's Answer]\n"
        f"{trial['answer_a']}\n"
        "[The End of Assistant A's Answer]\n\n"
        "[The Start of Assistant B's Answer]\n"
        f"{trial['answer_b']}\n"
        "[The End of Assistant B's Answer]"
    )


def build_payload(trial, config):
    payload = {
        "model": config.get("model", config["judge_model"]),
        "messages": [
            {
                "role": "system",
                "content": system_prompt(trial["prompt_condition"]),
            },
            {"role": "user", "content": build_user_prompt(trial)},
        ],
        "temperature": config.get("temperature", 0.0),
        "max_tokens": config.get("max_tokens", 4096),
        "stream": False,
    }
    payload.update(config.get("extra_body", {}))
    return payload


def parse_winner(text):
    matches = re.findall(r"\[\[\s*([ABC])\s*\]\]", text)
    if not matches:
        return "invalid"
    verdict = matches[-1]
    if verdict == "C":
        return "tie"
    return verdict


def long_answer_won(winner, long_answer_position):
    if winner in ("invalid", "tie"):
        return None
    return winner == long_answer_position


def make_parsed_row(trial, judge_model, winner, raw_ref):
    return {
        "trial_id": trial["trial_id"],
        "judge_model": judge_model,
        "question_id": trial["question_id"],
        "category": trial["category"],
        "condition": trial["condition"],
        "prompt_condition": trial["prompt_condition"],
        "winner": winner,
        "long_answer_won": long_answer_won(
            winner, trial["long_answer_position"]
        ),
        "raw_output_ref": raw_ref,
        "created_at": utc_now(),
    }


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


def judge_trial(config, trial, index, total, api_key):
    judge_model = config["judge_model"]
    raw_ref = f"{judge_model}:{trial['trial_id']}:{utc_now()}"
    raw_row = {
        "raw_output_ref": raw_ref,
        "trial_id": trial["trial_id"],
        "judge_model": judge_model,
        "created_at": utc_now(),
    }
    try:
        payload = build_payload(trial, config)
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
        "parsed_row": make_parsed_row(trial, judge_model, winner, raw_ref),
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
    futures = [executor.submit(judge_trial, config, trial, index, total, api_key) for config, trial, index, total, api_key in tasks]
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
