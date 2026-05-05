from xml.sax.saxutils import escape

from length_bias_pairing import PAIR_PATTERNS


MODEL_LABELS = {
    "deepseek-v4-flash": "DeepSeek V4 Flash",
    "deepseek-v4-pro": "DeepSeek V4 Pro",
    "gemini-3-flash-preview": "Gemini Flash",
    "glm-5.1": "GLM-5.1",
    "kimi-k2.6": "Kimi K2.6",
    "minimax-m2.7": "MiniMax M2.7",
    "mimo-v2-pro": "MiMo V2 Pro",
    "qwen3.6-plus": "Qwen3.6 Plus",
}

PROMPT_LABELS = {
    "neutral_no_length": "Neutral",
    "standard_anti_length": "Length-control",
}

OUTCOME_COLORS = {
    "long_wins": "#0072B2",
    "short_wins": "#D55E00",
    "ties": "#666666",
    "invalid": "#BDBDBD",
}

PAIR_COLORS = {
    "long_both_positions": "#332288",
    "short_both_positions": "#AA4499",
    "position_A_both": "#E69F00",
    "position_B_both": "#009E73",
    "tie_both": "#777777",
    "mixed_or_partial_tie": "#CC6677",
    "invalid_pair": "#BBBBBB",
    "missing_pair": "#88CCEE",
}

PAIR_LABELS = {
    "long_both_positions": "Long both",
    "short_both_positions": "Short both",
    "position_A_both": "A both",
    "position_B_both": "B both",
    "tie_both": "Tie both",
    "mixed_or_partial_tie": "Mixed",
    "invalid_pair": "Invalid",
    "missing_pair": "Missing",
}


def render_academic_svg(summary):
    groups = judge_prompt_groups(summary)
    judges = actual_judges(summary)
    shape = summary.get("statistical_analysis", {}).get("data_shape", {})
    bootstrap = (
        summary.get("statistical_analysis", {})
        .get("question_cluster", {})
        .get("bootstrap_95_ci", {})
    )
    subtitle = (
        f"{shape.get('questions', 0)} questions, "
        f"{summary.get('filtered_row_count', 0)} parsed judgments; "
        f"95% CI from question-cluster bootstrap "
        f"(seed={bootstrap.get('seed')}, iterations={bootstrap.get('iterations')})."
    )

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="1020" viewBox="0 0 1240 1020">',
        '<rect width="1240" height="1020" fill="#ffffff"/>',
        svg_text(56, 44, "Length-bias effects by judge and prompt", 24, "#111111", "700"),
        svg_text(56, 70, subtitle, 14, "#444444"),
    ]
    parts.extend(render_effect_panel(70, 120, 1100, 300, groups))
    parts.extend(render_outcome_panel(70, 465, 1100, 240, groups))
    parts.extend(render_pair_panel(70, 755, 1100, 190, judges, summary))
    parts.extend(render_pair_legend(70, 965))
    parts.append("</svg>")
    return "\n".join(parts)


def render_effect_panel(x, y, width, height, groups):
    label_width = 220
    plot_left = x + label_width
    plot_top = y + 48
    plot_width = width - label_width - 170
    row_step = 34
    axis_y = plot_top + row_step * max(len(groups), 1) + 10
    parts = [
        svg_text(x, y, "A. Net length preference with 95% CI", 17, "#111111", "700"),
        svg_text(
            x,
            y + 22,
            "Score: long win=1, short win=-1, tie=0; zero means no length preference.",
            13,
            "#444444",
        ),
    ]
    add_percent_axis(parts, plot_left, axis_y, plot_width, y + 42, axis_y, True)
    for index, group in enumerate(groups):
        row_y = plot_top + index * row_step
        stats = group.get("stats", {})
        ci = stats.get("bootstrap_95_ci", {})
        mean = stats.get("mean_net_length_preference")
        lower = ci.get("lower")
        upper = ci.get("upper")
        parts.append(svg_text(x, row_y + 4, group["label"], 12, "#111111"))
        if mean is None:
            parts.append(svg_text(plot_left, row_y + 4, "n/a", 12, "#666666"))
            continue
        mean_x = percent_x(plot_left, plot_width, mean)
        lower_x = percent_x(plot_left, plot_width, lower if lower is not None else mean)
        upper_x = percent_x(plot_left, plot_width, upper if upper is not None else mean)
        parts.append(
            f'<line x1="{lower_x:.1f}" y1="{row_y:.1f}" x2="{upper_x:.1f}" y2="{row_y:.1f}" '
            'stroke="#111111" stroke-width="1.5"/>'
        )
        parts.append(
            f'<line x1="{lower_x:.1f}" y1="{row_y - 5:.1f}" x2="{lower_x:.1f}" y2="{row_y + 5:.1f}" '
            'stroke="#111111" stroke-width="1.2"/>'
        )
        parts.append(
            f'<line x1="{upper_x:.1f}" y1="{row_y - 5:.1f}" x2="{upper_x:.1f}" y2="{row_y + 5:.1f}" '
            'stroke="#111111" stroke-width="1.2"/>'
        )
        parts.append(
            f'<circle cx="{mean_x:.1f}" cy="{row_y:.1f}" r="4.8" fill="#0072B2" stroke="#111111" stroke-width="0.7"/>'
        )
        parts.append(
            svg_text(
                plot_left + plot_width + 18,
                row_y + 4,
                f"{pct(mean)} [{pct(lower)}, {pct(upper)}]",
                12,
                "#111111",
            )
        )
    parts.append(svg_text(plot_left + plot_width / 2, axis_y + 34, "Net length preference", 13, "#111111", "400", "middle"))
    return parts


def render_outcome_panel(x, y, width, height, groups):
    label_width = 220
    bar_left = x + label_width
    bar_width = width - label_width - 210
    row_step = 27
    bar_height = 14
    top = y + 70
    parts = [
        svg_text(x, y, "B. Outcome composition", 17, "#111111", "700"),
        svg_text(x, y + 22, "Bars sum to 100% within each judge and prompt group.", 13, "#444444"),
    ]
    add_outcome_legend(parts, bar_left, y + 34)
    for index, group in enumerate(groups):
        row_y = top + index * row_step
        counts = group.get("counts", {})
        total = counts.get("total", 0)
        parts.append(svg_text(x, row_y + 11, group["label"], 12, "#111111"))
        cursor = bar_left
        for key in ("long_wins", "short_wins", "ties", "invalid"):
            value = counts.get(key, 0)
            segment_width = 0 if total == 0 else bar_width * value / total
            if segment_width:
                parts.append(
                    f'<rect x="{cursor:.1f}" y="{row_y:.1f}" width="{segment_width:.1f}" '
                    f'height="{bar_height}" fill="{OUTCOME_COLORS[key]}"/>'
                )
            cursor += segment_width
        parts.append(
            f'<rect x="{bar_left:.1f}" y="{row_y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height}" fill="none" stroke="#111111" stroke-width="0.6"/>'
        )
        parts.append(svg_text(bar_left + bar_width + 14, row_y + 11, f"n={total}", 12, "#111111"))
    add_percent_scale(parts, bar_left, top + row_step * len(groups) + 8, bar_width)
    return parts


def render_pair_panel(x, y, width, height, judges, summary):
    label_width = 160
    bar_left = x + label_width
    bar_width = width - label_width - 120
    row_step = 31
    bar_height = 15
    top = y + 42
    paired_by_judge = summary.get("swapped_pair_analysis", {}).get("by_judge", {})
    max_total = max(
        [paired_by_judge.get(judge, {}).get("total_pairs", 0) for judge in judges] or [1]
    )
    axis_max = nice_count(max_total)
    parts = [
        svg_text(x, y, "C. Swapped long_A / long_B pair patterns", 17, "#111111", "700"),
        svg_text(x, y + 22, "Counts are complete swapped pairs per judge.", 13, "#444444"),
    ]
    for index, judge in enumerate(judges):
        row_y = top + index * row_step
        counts = paired_by_judge.get(judge, {})
        total = counts.get("total_pairs", 0)
        cursor = bar_left
        parts.append(svg_text(x, row_y + 12, model_label(judge), 12, "#111111"))
        for key in PAIR_PATTERNS:
            value = counts.get(key, 0)
            segment_width = 0 if axis_max == 0 else bar_width * value / axis_max
            if segment_width:
                parts.append(
                    f'<rect x="{cursor:.1f}" y="{row_y:.1f}" width="{segment_width:.1f}" '
                    f'height="{bar_height}" fill="{PAIR_COLORS[key]}"/>'
                )
            cursor += segment_width
        parts.append(
            f'<rect x="{bar_left:.1f}" y="{row_y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height}" fill="none" stroke="#111111" stroke-width="0.6"/>'
        )
        parts.append(svg_text(bar_left + bar_width + 14, row_y + 12, f"{total} pairs", 12, "#111111"))
    add_count_axis(parts, bar_left, top + row_step * len(judges) + 8, bar_width, axis_max)
    return parts


def add_percent_axis(parts, left, axis_y, width, grid_top, grid_bottom, include_negative):
    ticks = (-1.0, -0.5, 0.0, 0.5, 1.0) if include_negative else (0.0, 0.5, 1.0)
    for tick in ticks:
        x = percent_x(left, width, tick)
        stroke = "#999999" if tick == 0 else "#DDDDDD"
        parts.append(
            f'<line x1="{x:.1f}" y1="{grid_top:.1f}" x2="{x:.1f}" y2="{grid_bottom:.1f}" '
            f'stroke="{stroke}" stroke-width="1"/>'
        )
        parts.append(svg_text(x, axis_y + 18, pct(tick), 11, "#111111", "400", "middle"))
    parts.append(f'<line x1="{left:.1f}" y1="{axis_y:.1f}" x2="{left + width:.1f}" y2="{axis_y:.1f}" stroke="#111111"/>')


def add_percent_scale(parts, left, y, width):
    for tick in (0.0, 0.5, 1.0):
        x = left + width * tick
        parts.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x:.1f}" y2="{y + 5:.1f}" stroke="#111111"/>')
        parts.append(svg_text(x, y + 18, pct(tick), 11, "#111111", "400", "middle"))
    parts.append(f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + width:.1f}" y2="{y:.1f}" stroke="#111111"/>')


def add_count_axis(parts, left, y, width, axis_max):
    step = 10 if axis_max > 20 else 5
    for tick in range(0, axis_max + 1, step):
        x = left + width * tick / axis_max
        parts.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x:.1f}" y2="{y + 5:.1f}" stroke="#111111"/>')
        parts.append(svg_text(x, y + 18, str(tick), 11, "#111111", "400", "middle"))
    parts.append(f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + width:.1f}" y2="{y:.1f}" stroke="#111111"/>')


def add_outcome_legend(parts, x, y):
    items = (
        ("Long", OUTCOME_COLORS["long_wins"]),
        ("Short", OUTCOME_COLORS["short_wins"]),
        ("Tie", OUTCOME_COLORS["ties"]),
        ("Invalid", OUTCOME_COLORS["invalid"]),
    )
    for index, (label, color) in enumerate(items):
        lx = x + index * 95
        parts.append(f'<rect x="{lx}" y="{y + 5}" width="12" height="12" fill="{color}"/>')
        parts.append(svg_text(lx + 18, y + 16, label, 12, "#111111"))


def render_pair_legend(x, y):
    parts = [svg_text(x, y - 8, "Swapped-pair legend", 13, "#111111", "700")]
    for index, key in enumerate(PAIR_PATTERNS):
        lx = x + (index % 4) * 250
        ly = y + (index // 4) * 22
        parts.append(f'<rect x="{lx}" y="{ly}" width="12" height="12" fill="{PAIR_COLORS[key]}"/>')
        parts.append(svg_text(lx + 18, ly + 11, PAIR_LABELS[key], 12, "#111111"))
    return parts


def judge_prompt_groups(summary):
    stats_by_group = summary.get("statistical_analysis", {}).get("by_judge_prompt", {})
    counts_by_group = summary.get("by_judge_prompt", {})
    judges = actual_judges(summary)
    prompts = prompt_order(stats_by_group.keys() | counts_by_group.keys())
    groups = []
    for judge in judges:
        for prompt in prompts:
            key = f"{judge}::{prompt}"
            if key not in stats_by_group and key not in counts_by_group:
                continue
            groups.append(
                {
                    "key": key,
                    "judge": judge,
                    "prompt": prompt,
                    "label": f"{model_label(judge)} - {prompt_label(prompt)}",
                    "stats": stats_by_group.get(key, {}),
                    "counts": counts_by_group.get(key, {}),
                }
            )
    return groups


def actual_judges(summary):
    judges = summary.get("filtered_judge_models") or sorted(summary.get("by_judge", {}))
    return [judge for judge in judges if summary.get("by_judge", {}).get(judge)]


def prompt_order(keys):
    prompts = {split_group_key(key)[1] for key in keys}
    preferred = ["neutral_no_length", "standard_anti_length"]
    ordered = [prompt for prompt in preferred if prompt in prompts]
    ordered.extend(sorted(prompts - set(ordered)))
    return ordered


def split_group_key(key):
    if "::" not in key:
        return key, "unknown"
    return key.split("::", 1)


def percent_x(left, width, value):
    value = max(-1.0, min(1.0, value))
    return left + ((value + 1.0) / 2.0) * width


def nice_count(value):
    if value <= 10:
        return 10
    if value <= 20:
        return 20
    return ((value + 9) // 10) * 10


def pct(value):
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def svg_text(x, y, text, size=14, color="#1f2933", weight="400", anchor="start"):
    return (
        f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
        f'font-family="Times New Roman, Times, serif" font-weight="{weight}" '
        f'text-anchor="{anchor}">{escape(str(text))}</text>'
    )


def model_label(model):
    return MODEL_LABELS.get(model, model)


def prompt_label(prompt):
    return PROMPT_LABELS.get(prompt, prompt)
