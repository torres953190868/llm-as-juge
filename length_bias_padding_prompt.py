import math

from length_bias_common import word_count
from length_bias_padding_metadata import DEFAULT_PADDING_TEMPERATURE


SYSTEM_PROMPT = """You are creating length-padded answer variants for an LLM-as-a-judge bias experiment.

Your task is to minimally expand the provided answer while preserving its meaning.

Strict rules:
- Do not improve the answer.
- Do not make it more correct, more persuasive, better organized, or more complete.
- Preserve the original answer's meaning, stance, factual claims, mistakes, omissions, order, tone, and formatting as much as possible.
- Keep the original answer text as close as possible.
- Only add redundant restatements, short clarifying transitions, and summary sentences that repeat existing information.
- Do not add new facts, new examples, new citations, code, headings, bullet points, or stronger claims.
- If the answer contains multiple turns or paragraphs, preserve their order and boundaries.
- Target the length range specified by the user.
- Return only the padded answer, with no commentary or labels."""


USER_PROMPT_TEMPLATE = """Question:
{question}

{answer_label}:
{answer}

Word-count budget: original {original_word_count}; accepted {min_words}-{max_words} words ({min_percent}% to {max_percent}%); preferred target about {target_words} words.
Minimally expand the answer as a length-padded variant, but do not stop until it is at least {min_words} words. Add neutral restatement sentences throughout the answer, especially after paragraphs or list items, while obeying all strict rules above.
{retry_note}"""


def build_payload(
    sample,
    model,
    attempt,
    min_ratio,
    max_ratio,
    max_tokens,
    previous_failure=None,
    draft_answer=None,
):
    original_word_count = word_count(sample["original_answer"])
    min_words = math.ceil(original_word_count * min_ratio)
    max_words = math.floor(original_word_count * max_ratio)
    target_ratio = min(max_ratio, max(min_ratio + 0.15, min_ratio * 1.1))
    target_words = min(
        max_words, max(min_words, math.ceil(original_word_count * target_ratio))
    )
    answer = draft_answer or sample["original_answer"]
    retry_note = retry_feedback(
        attempt, previous_failure, min_words, max_words
    )
    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=sample["question"],
        answer_label="Previous padded draft" if draft_answer else "Original answer",
        answer=answer,
        original_word_count=original_word_count,
        min_words=min_words,
        max_words=max_words,
        target_words=target_words,
        min_percent=int(min_ratio * 100 + 0.5),
        max_percent=int(max_ratio * 100 + 0.5),
        retry_note=retry_note,
    )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "thinking": {"type": "disabled"},
        "temperature": DEFAULT_PADDING_TEMPERATURE,
        "max_tokens": max_tokens,
        "stream": False,
    }


def retry_feedback(attempt, previous_failure, min_words, max_words):
    if attempt <= 1:
        return ""
    if previous_failure and previous_failure.get("padded_word_count"):
        direction = (
            "shorten it"
            if previous_failure["failed_reason"] == "above_max_length_ratio"
            else "expand it"
        )
        return (
            "Retry feedback: previous attempt produced "
            f"{previous_failure['padded_word_count']} words "
            f"({previous_failure['length_ratio']:.2f}x), rejected as "
            f"{previous_failure['failed_reason']}. Revise the draft and "
            f"{direction} to land inside {min_words}-{max_words} words."
        )
    return (
        "Retry feedback: previous attempt failed. Generate a fresh padded "
        "answer inside the word-count budget."
    )
