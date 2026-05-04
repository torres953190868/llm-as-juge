import hashlib
import os

from length_bias_common import utc_now


SENSITIVE_CONFIG_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "bearer_token",
    "token",
    "access_token",
    "secret",
}


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def maybe_file_sha256(path):
    if not path or not os.path.exists(path):
        return None
    return file_sha256(path)


def sanitize_judge_config(config):
    sanitized = {}
    for key, value in config.items():
        if str(key).lower() in SENSITIVE_CONFIG_KEYS:
            continue
        if isinstance(value, dict):
            sanitized[key] = sanitize_judge_config(value)
        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_judge_config(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            sanitized[key] = value
    return sanitized


def sanitize_judge_configs(configs):
    return [sanitize_judge_config(config) for config in configs]


def prompt_hash_metadata(system_text="", user_text=""):
    combined_prompt = f"[system]\n{system_text}\n\n[user]\n{user_text}"
    return {
        "system_prompt_sha256": text_sha256(system_text),
        "user_prompt_sha256": text_sha256(user_text),
        "prompt_sha256": text_sha256(combined_prompt),
    }


def run_metadata(input_path=None, judge_configs=None, extra=None):
    generated_at = utc_now()
    metadata = {
        "generated_at": generated_at,
        "created_at": generated_at,
    }
    if input_path:
        metadata["input_path"] = input_path
        metadata["input_sha256"] = maybe_file_sha256(input_path)
    if judge_configs is not None:
        metadata["judge_configs"] = sanitize_judge_configs(judge_configs)
    if extra:
        metadata.update(extra)
    return metadata
