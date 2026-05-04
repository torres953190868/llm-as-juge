import json
import re

from length_bias_common import DEEPSEEK_API_KEY_ENV, DEEPSEEK_ENDPOINT
from length_bias_metadata import sanitize_judge_configs


GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL = "gemini-3-flash-preview"
DEEPSEEK_MODEL = "deepseek-v4-flash"
XIAOMI_ENDPOINT = "https://token-plan-cn.xiaomimimo.com/v1/chat/completions"
XIAOMI_API_KEY_ENV = "XIAOMI_API_KEY"
XIAOMI_MODEL = "mimo-v2-pro"


STANDARD_SYSTEM_PROMPT = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail. Avoid position bias and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible. After providing a short explanation, output your final verdict strictly as [[A]] if assistant A is better, [[B]] if assistant B is better, and [[C]] for a tie."""


NEUTRAL_SYSTEM_PROMPT = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail. Avoid position bias and ensure that the order in which the responses were presented does not influence your decision. Be as objective as possible. After providing a short explanation, output your final verdict strictly as [[A]] if assistant A is better, [[B]] if assistant B is better, and [[C]] for a tie."""


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
    return sanitize_judge_configs(configs)


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
