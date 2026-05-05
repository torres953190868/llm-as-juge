from length_bias_judge import DEEPSEEK_MODEL, GEMINI_MODEL, OPENCODE_GO_MODELS
from length_bias_judge import XIAOMI_MODEL


JUDGE_MODEL_ALIASES = {
    DEEPSEEK_MODEL: ("deepseek-v4-flash",),
}


def selected_judge_models(args):
    models = []
    if args.gemini:
        models.append(GEMINI_MODEL)
    if args.opencode_go:
        models.extend(OPENCODE_GO_MODELS)
    if args.deepseek:
        models.append(DEEPSEEK_MODEL)
    if args.xiaomi:
        models.append(XIAOMI_MODEL)
    return list(dict.fromkeys(models))


def available_judge_models(rows):
    return sorted({
        row.get("judge_model")
        for row in rows
        if row.get("judge_model") and row.get("bias_type", "length") == "length"
    })


def expanded_judge_models(judge_models):
    selected = []
    for model in judge_models:
        selected.append(model)
        selected.extend(JUDGE_MODEL_ALIASES.get(model, ()))
    return list(dict.fromkeys(selected))


def missing_judge_models(judge_models, available_models):
    available = set(available_models)
    missing = []
    for model in judge_models:
        candidates = {model, *JUDGE_MODEL_ALIASES.get(model, ())}
        if not candidates & available:
            missing.append(model)
    return missing


def filter_rows_by_judge(rows, judge_models):
    selected = set(expanded_judge_models(judge_models))
    return [
        row for row in rows
        if row.get("judge_model") in selected
        and row.get("bias_type", "length") == "length"
    ]
