import json
import re

from length_bias_common import DEEPSEEK_API_KEY_ENV, DEEPSEEK_ENDPOINT, get_env_value
from length_bias_judge_client import (
    API_MODE_ANTHROPIC_MESSAGES,
    API_MODE_OPENAI_CHAT,
)
from length_bias_metadata import sanitize_judge_configs


GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL = "gemini-3-flash-preview"
OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go/v1"
OPENCODE_GO_BASE_URL_ENV = "OPENCODE_GO_BASE_URL"
OPENCODE_GO_API_KEY_ENV = "OPENCODE_GO_API_KEY"
OPENCODE_GO_PROVIDER = "opencode-go"
OPENCODE_GO_CHAT_ENDPOINT = f"{OPENCODE_GO_BASE_URL}/chat/completions"
OPENCODE_GO_MESSAGES_ENDPOINT = f"{OPENCODE_GO_BASE_URL}/messages"
OPENCODE_GO_MODELS = ()
# Disabled OpenCode Go candidates:
# "qwen3.6-plus", "minimax-m2.7", "deepseek-v4-pro",
# "glm-5.1", "kimi-k2.6", "mimo-v2-pro"
OPENCODE_GO_MESSAGES_MODELS = {"minimax-m2.7"}
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_BASE_URL_ENV = "DEEPSEEK_BASE_URL"
DEEPSEEK_PROVIDER = "deepseek"
DEEPSEEK_MODEL = "deepseek-v4-pro"
XIAOMI_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
XIAOMI_BASE_URL_ENV = "XIAOMI_BASE_URL"
XIAOMI_ENDPOINT = f"{XIAOMI_BASE_URL}/chat/completions"
XIAOMI_API_KEY_ENV = "XIAOMI_API_KEY"
XIAOMI_PROVIDER = "xiaomi"
XIAOMI_MODEL = "mimo-v2-pro"


STANDARD_SYSTEM_PROMPT = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail. Avoid position bias and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible. After providing a short explanation, output your final verdict strictly as [[A]] if assistant A is better, [[B]] if assistant B is better, and [[C]] for a tie."""


NEUTRAL_SYSTEM_PROMPT = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail. Avoid position bias and ensure that the order in which the responses were presented does not influence your decision. Be as objective as possible. After providing a short explanation, output your final verdict strictly as [[A]] if assistant A is better, [[B]] if assistant B is better, and [[C]] for a tie."""


def opencode_go_endpoint(model):
    if model in OPENCODE_GO_MESSAGES_MODELS:
        return OPENCODE_GO_MESSAGES_ENDPOINT
    return OPENCODE_GO_CHAT_ENDPOINT


def opencode_go_endpoint_path(model):
    if model in OPENCODE_GO_MESSAGES_MODELS:
        return "/messages"
    return "/chat/completions"


def opencode_go_api_mode(model):
    if model in OPENCODE_GO_MESSAGES_MODELS:
        return API_MODE_ANTHROPIC_MESSAGES
    return API_MODE_OPENAI_CHAT


def opencode_go_judge_config(model):
    return {
        "judge_model": model,
        "model": model,
        "provider": OPENCODE_GO_PROVIDER,
        "endpoint": opencode_go_endpoint(model),
        "base_url": OPENCODE_GO_BASE_URL,
        "base_url_env": OPENCODE_GO_BASE_URL_ENV,
        "endpoint_path": opencode_go_endpoint_path(model),
        "api_key_env": OPENCODE_GO_API_KEY_ENV,
        "api_mode": opencode_go_api_mode(model),
        "temperature": 0.0,
        "max_tokens": 4096,
    }


def opencode_go_judge_configs():
    return [opencode_go_judge_config(model) for model in OPENCODE_GO_MODELS]


def deepseek_judge_config(model):
    return {
        "judge_model": model,
        "model": model,
        "provider": DEEPSEEK_PROVIDER,
        "endpoint": DEEPSEEK_ENDPOINT,
        "base_url": DEEPSEEK_BASE_URL,
        "base_url_env": DEEPSEEK_BASE_URL_ENV,
        "endpoint_path": "/chat/completions",
        "api_key_env": DEEPSEEK_API_KEY_ENV,
        "api_mode": API_MODE_OPENAI_CHAT,
        "temperature": 0.0,
        "max_tokens": 4096,
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
        "provider": XIAOMI_PROVIDER,
        "endpoint": XIAOMI_ENDPOINT,
        "base_url": XIAOMI_BASE_URL,
        "base_url_env": XIAOMI_BASE_URL_ENV,
        "endpoint_path": "/chat/completions",
        "api_key_env": XIAOMI_API_KEY_ENV,
        "api_mode": API_MODE_OPENAI_CHAT,
        "temperature": 0.0,
        "max_tokens": 4096,
    }


def append_unique_config(configs, config):
    judge_model = config["judge_model"]
    if all(existing["judge_model"] != judge_model for existing in configs):
        configs.append(config)


def builtin_judge_configs(args):
    configs = []
    if getattr(args, "gemini", 0):
        append_unique_config(configs, gemini_judge_config())
    if getattr(args, "opencode_go", 0):
        for config in opencode_go_judge_configs():
            append_unique_config(configs, config)
    if getattr(args, "deepseek", 0):
        append_unique_config(configs, deepseek_judge_config(args.judge_model))
    if getattr(args, "xiaomi", 0):
        append_unique_config(configs, xiaomi_judge_config())
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
    return sanitize_judge_configs(configs)


def resolve_judge_config(config, env_file):
    resolved = dict(config)
    base_url_env = resolved.get("base_url_env")
    endpoint_path = resolved.get("endpoint_path")
    if base_url_env and endpoint_path:
        base_url = get_env_value(base_url_env, env_file) or resolved.get("base_url")
        if base_url:
            resolved["endpoint"] = (
                base_url.rstrip("/") + "/" + str(endpoint_path).lstrip("/")
            )
    return resolved


def resolve_judge_configs(configs, env_file):
    return [resolve_judge_config(config, env_file) for config in configs]


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
