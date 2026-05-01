import datetime
import json
import os
import time
import urllib.error
import urllib.request


DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
OPENAI_RESPONSES_ENDPOINT = "https://api.openai.com/v1/responses"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def word_count(text):
    return len(text.split())


def length_ratio(short_text, long_text):
    short_count = word_count(short_text)
    if short_count == 0:
        return 0.0
    return word_count(long_text) / short_count


def read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def truncate_file(path):
    with open(path, "w", encoding="utf-8"):
        pass


def load_env_file(path):
    values = {}
    if not path or not os.path.exists(path):
        return values

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                values[key] = value
    return values


def get_api_key(env_name, env_file):
    return os.environ.get(env_name) or load_env_file(env_file).get(env_name)


def is_retryable_error(exc):
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or 500 <= exc.code < 600
    return isinstance(exc, urllib.error.URLError)


def error_details(exc):
    detail = {"type": type(exc).__name__, "message": str(exc)}
    if isinstance(exc, urllib.error.HTTPError):
        detail["status_code"] = exc.code
        try:
            detail["body"] = exc.read().decode("utf-8")
        except Exception as body_exc:
            detail["body_error"] = str(body_exc)
    return detail


def call_chat_completion(endpoint, payload, api_key, timeout=120):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout) as response:
        response_body = response.read().decode("utf-8")
        return json.loads(response_body)


def call_with_retries(endpoint, payload, api_key, attempts=3, timeout=120):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return call_chat_completion(endpoint, payload, api_key, timeout=timeout)
        except Exception as exc:
            last_error = exc
            if attempt == attempts or not is_retryable_error(exc):
                raise
            time.sleep(attempt * 2)
    raise last_error


def extract_chat_content(response):
    choices = response.get("choices", [])
    if not choices:
        raise ValueError("Chat completion response has no choices")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if not content.strip():
        raise ValueError("Chat completion response has empty content")
    return content.strip()


def extract_responses_text(response):
    output_text = response.get("output_text", "")
    if output_text.strip():
        return output_text.strip()

    parts = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            text = content.get("text", "")
            if text:
                parts.append(text)

    text = "\n".join(parts).strip()
    if not text:
        raise ValueError("Responses API response has empty text output")
    return text
