from length_bias_common import DEEPSEEK_API_KEY_ENV, DEEPSEEK_ENDPOINT, utc_now
from length_bias_metadata import (
    prompt_hash_metadata,
    run_metadata,
    sanitize_judge_config,
    text_sha256,
)


DEFAULT_PADDING_TEMPERATURE = 0.2


def payload_prompt_hashes(payload):
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


def padding_config(args, prompt_version):
    return {
        "model": args.model,
        "endpoint": DEEPSEEK_ENDPOINT,
        "api_key_env": DEEPSEEK_API_KEY_ENV,
        "temperature": DEFAULT_PADDING_TEMPERATURE,
        "max_tokens": args.max_tokens,
        "min_ratio": args.min_ratio,
        "max_ratio": args.max_ratio,
        "padding_attempts": args.padding_attempts,
        "prompt_version": prompt_version,
    }


def make_base_run_metadata(args, prompt_version):
    return run_metadata(
        input_path=args.input,
        extra={
            "stage": "pad_answers_deepseek",
            "script": "02_pad_answers_deepseek.py",
            "input_format": args.input_format,
            "output_jsonl": args.output_jsonl,
            "output_txt": args.output_txt,
            "raw_output": args.raw_output,
            "failed_output": args.failed_output,
            "limit": args.limit,
            "dry_run": args.dry_run,
            "padding_config": sanitize_judge_config(
                padding_config(args, prompt_version)
            ),
        },
    )


def make_attempt_metadata(
    args,
    payload,
    base_metadata,
    prompt_version,
    attempt,
    turn_index,
    turn_count,
):
    generated_at = utc_now()
    metadata = dict(base_metadata)
    metadata.update(
        {
            "generated_at": generated_at,
            "created_at": generated_at,
            "attempt": attempt,
            "model": payload.get("model", args.model),
            "endpoint": DEEPSEEK_ENDPOINT,
            "temperature": payload.get(
                "temperature", DEFAULT_PADDING_TEMPERATURE
            ),
            "max_tokens": payload.get("max_tokens", args.max_tokens),
            "prompt_version": prompt_version,
        }
    )
    if turn_index is not None:
        metadata["turn_index"] = turn_index
        metadata["turn_count"] = turn_count
    metadata.update(payload_prompt_hashes(payload))
    return metadata


def prompt_hashes_from_metadata(metadata):
    return {
        "system_prompt_sha256": metadata.get("system_prompt_sha256"),
        "user_prompt_sha256": metadata.get("user_prompt_sha256"),
        "prompt_sha256": metadata.get("prompt_sha256"),
    }


def make_accepted_run_metadata(args, base_metadata, turn_metadatas, prompt_version):
    generated_at = utc_now()
    metadata = dict(base_metadata)
    metadata.update(
        {
            "generated_at": generated_at,
            "created_at": generated_at,
            "model": args.model,
            "endpoint": DEEPSEEK_ENDPOINT,
            "temperature": DEFAULT_PADDING_TEMPERATURE,
            "max_tokens": args.max_tokens,
            "prompt_version": prompt_version,
        }
    )
    prompt_hashes = [
        {
            "turn_index": item.get("turn_index"),
            "turn_count": item.get("turn_count"),
            "system_prompt_sha256": item.get("system_prompt_sha256"),
            "user_prompt_sha256": item.get("user_prompt_sha256"),
            "prompt_sha256": item.get("prompt_sha256"),
        }
        for item in turn_metadatas
    ]
    metadata["turn_prompt_hashes"] = prompt_hashes
    if len(turn_metadatas) == 1:
        metadata.update(prompt_hashes_from_metadata(turn_metadatas[0]))
    else:
        metadata["prompt_sha256"] = text_sha256(
            "\n".join(item["prompt_sha256"] for item in prompt_hashes)
        )
    return metadata
