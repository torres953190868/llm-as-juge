from length_bias_common import call_json_with_retries, extract_chat_content


API_MODE_OPENAI_CHAT = "openai_chat"
API_MODE_ANTHROPIC_MESSAGES = "anthropic_messages"
ANTHROPIC_VERSION = "2023-06-01"


def anthropic_messages_payload(chat_payload):
    system_parts = []
    messages = []
    for message in chat_payload.get("messages", []):
        role = message.get("role")
        content = str(message.get("content", ""))
        if role == "system":
            system_parts.append(content)
        elif role in ("user", "assistant"):
            messages.append({"role": role, "content": content})
        else:
            messages.append({"role": "user", "content": f"[{role}]\n{content}"})

    payload = {
        "model": chat_payload["model"],
        "messages": messages,
        "max_tokens": chat_payload.get("max_tokens", 4096),
        "temperature": chat_payload.get("temperature", 0.0),
        "stream": bool(chat_payload.get("stream", False)),
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    return payload


def request_payload(config, chat_payload):
    if config.get("api_mode", API_MODE_OPENAI_CHAT) == API_MODE_ANTHROPIC_MESSAGES:
        return anthropic_messages_payload(chat_payload)
    return chat_payload


def request_headers(config, api_key):
    if config.get("api_mode", API_MODE_OPENAI_CHAT) == API_MODE_ANTHROPIC_MESSAGES:
        return {
            "x-api-key": api_key,
            "anthropic-version": config.get(
                "anthropic_version", ANTHROPIC_VERSION
            ),
        }
    return {"Authorization": f"Bearer {api_key}"}


def call_judge_model(config, chat_payload, api_key):
    return call_json_with_retries(
        config["endpoint"],
        request_payload(config, chat_payload),
        request_headers(config, api_key),
        attempts=config.get("attempts", 3),
        timeout=config.get("timeout", 120),
    )


def extract_anthropic_messages_content(response):
    content = response.get("content", "")
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text", "")
                if text:
                    parts.append(text)
        text = "\n".join(parts).strip()
    else:
        text = ""

    if not text:
        raise ValueError("Anthropic Messages response has empty content")
    return text


def extract_judge_content(config, response):
    if config.get("api_mode", API_MODE_OPENAI_CHAT) == API_MODE_ANTHROPIC_MESSAGES:
        try:
            return extract_anthropic_messages_content(response)
        except ValueError:
            return extract_chat_content(response)
    return extract_chat_content(response)
