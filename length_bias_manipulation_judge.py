import json
import re

from length_bias_manipulation import SYSTEM_PROMPT, sample_sha256


CHECK_FIELDS = (
    "semantic_equivalence",
    "new_facts",
    "structure_improvement",
    "quality_improvement",
)


def normalize_trial(row):
    trial = dict(row)
    if not trial.get("sample_sha256"):
        trial["sample_sha256"] = sample_sha256(trial)
    if not trial.get("trial_id"):
        trial["trial_id"] = (
            f"q{trial['question_id']}_manipulation_"
            f"{trial['sample_sha256'][:12]}"
        )
    return trial


def strict_passed(parsed):
    return (
        parsed.get("semantic_equivalence") is True
        and parsed.get("new_facts") is False
        and parsed.get("structure_improvement") is False
        and parsed.get("quality_improvement") is False
    )


def parse_json_object(text):
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def parse_check_result(text):
    data = parse_json_object(text)
    if not isinstance(data, dict):
        raise ValueError("manipulation-check response must be a JSON object")

    parsed = {}
    for field in CHECK_FIELDS:
        value = data.get(field)
        if not isinstance(value, bool):
            raise ValueError(f"{field} must be true or false")
        parsed[field] = value
    explanation = data.get("explanation", "")
    parsed["explanation"] = str(explanation).strip()
    parsed["manipulation_passed"] = strict_passed(parsed)
    return parsed


def build_payload(trial, config):
    payload = {
        "model": config.get("model", config["judge_model"]),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": trial["user_prompt"]},
        ],
        "temperature": config.get("temperature", 0.0),
        "max_tokens": config.get("max_tokens", 1024),
        "stream": False,
    }
    payload.update(config.get("extra_body", {}))
    return payload
