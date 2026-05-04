import json

from length_bias_common import length_ratio, utc_now
from length_bias_metadata import prompt_hash_metadata, text_sha256


PROMPT_VERSION = "manipulation_check_v1"

CHECK_CRITERIA = [
    {
        "name": "semantic_equivalence",
        "question": (
            "Does the padded answer preserve the same core meaning, claims, "
            "and task response as the original answer?"
        ),
    },
    {
        "name": "new_facts",
        "question": (
            "Does the padded answer introduce facts, examples, recommendations, "
            "or claims that are absent from the original answer?"
        ),
    },
    {
        "name": "structure_improvement",
        "question": (
            "Does the padded answer improve organization, formatting, headings, "
            "or step-by-step structure beyond lengthening?"
        ),
    },
    {
        "name": "quality_improvement",
        "question": (
            "Does the padded answer improve helpfulness, clarity, specificity, "
            "reasoning, or overall answer quality beyond lengthening?"
        ),
    },
]

SYSTEM_PROMPT = (
    "You are evaluating whether a length-padded answer is a valid manipulation "
    "for a judge-bias experiment. Do not judge which answer is better overall. "
    "Assess only whether padding changed semantics, added facts, improved "
    "structure, or improved quality."
)

USER_PROMPT_TEMPLATE = """Question:
{question}

Original answer:
{original_answer}

Padded answer:
{padded_answer}

Check criteria:
1. Semantic equivalence: does the padded answer preserve the same core meaning?
2. New facts: does the padded answer introduce facts or claims not in the original?
3. Structure improvement: does the padded answer improve organization or formatting?
4. Quality improvement: does the padded answer improve helpfulness, clarity, specificity, or reasoning?

Return JSON with keys:
- semantic_equivalence: true/false
- new_facts: true/false
- structure_improvement: true/false
- quality_improvement: true/false
- explanation: short rationale
"""


def get_length_ratio(row):
    ratio = row.get("length_ratio")
    if ratio is not None:
        return float(ratio)
    return length_ratio(row.get("original_answer", ""), row.get("padded_answer", ""))


def validate_padded_row(row):
    required = ("question_id", "category", "original_answer", "padded_answer")
    missing = [field for field in required if not row.get(field)]
    if missing:
        raise ValueError(f"missing required field(s): {', '.join(missing)}")


def question_text(row):
    question = row.get("question")
    if question:
        return question

    turns = row.get("question_turns")
    if isinstance(turns, list):
        return "\n".join(str(turn) for turn in turns if str(turn).strip())
    return ""


def build_user_prompt(row):
    return USER_PROMPT_TEMPLATE.format(
        question=question_text(row),
        original_answer=row["original_answer"],
        padded_answer=row["padded_answer"],
    )


def sample_sha256(row):
    payload = {
        "question_id": row.get("question_id"),
        "original_answer": row.get("original_answer", ""),
        "padded_answer": row.get("padded_answer", ""),
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return text_sha256(text)


def make_manipulation_trial(row, input_sha256, metadata):
    validate_padded_row(row)
    sample_hash = sample_sha256(row)
    user_prompt = build_user_prompt(row)
    generated_at = utc_now()
    prompt_hashes = prompt_hash_metadata(SYSTEM_PROMPT, user_prompt)
    return {
        "trial_id": f"q{row['question_id']}_manipulation_{sample_hash[:12]}",
        "bias_type": "manipulation_check",
        "question_id": row["question_id"],
        "category": row["category"],
        "length_ratio": get_length_ratio(row),
        "sample_sha256": sample_hash,
        "original_answer": row["original_answer"],
        "padded_answer": row["padded_answer"],
        "check_criteria": CHECK_CRITERIA,
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        **prompt_hashes,
        "input_sha256": input_sha256,
        "run_metadata": metadata,
        "generated_at": generated_at,
        "created_at": generated_at,
    }
