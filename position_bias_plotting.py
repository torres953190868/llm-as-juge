from xml.sax.saxutils import escape


MODEL_LABELS = {
    "deepseek-v4-pro": "DeepSeek V4 Pro",
    "gemini-3-flash-preview": "Gemini Flash",
    "mimo-v2-pro": "MiMo V2 Pro",
}

POSITION_COLORS = {
    "position_a_wins": "#E69F00",
    "position_b_wins": "#009E73",
    "ties": "#666666",
    "invalid": "#BDBDBD",
}

SOURCE_COLORS = {
    "model_a_wins": "#0072B2",
    "model_b_wins": "#D55E00",
    "ties": "#666666",
    "invalid": "#BDBDBD",
}

PAIR_KEYS = (
    "source_model_a_both",
    "source_model_b_both",
    "position_A_both",
    "position_B_both",
    "tie_both",
    "mixed_or_partial_tie",
    "invalid_pair",
    "missing_pair",
)

PAIR_COLORS = {
    "source_model_a_both": "#0072B2",
    "source_model_b_both": "#D55E00",
    "position_A_both": "#E69F00",
    "position_B_both": "#009E73",
    "tie_both": "#777777",
    "mixed_or_partial_tie": "#CC6677",
    "invalid_pair": "#BBBBBB",
    "missing_pair": "#88CCEE",
}

PAIR_LABELS = {
    "source_model_a_both": "Source A both",
    "source_model_b_both": "Source B both",
    "position_A_both": "Pos A both",
    "position_B_both": "Pos B both",
    "tie_both": "Tie both",
    "mixed_or_partial_tie": "Mixed",
    "invalid_pair": "Invalid",
    "missing_pair": "Missing",
}


def render_position_bias_svg(summary):
    judges = sorted(summary.get("by_judge", {}))
    pair_total = summary.get("swapped_pair_analysis", {}).get("total_pairs", 0)
    subtitle = (
        f"{summary.get('filtered_row_count', 0)} parsed judgments; "
        f"{pair_total} swapped pairs after default empty-response exclusions."
    )
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="1000" viewBox="0 0 1240 1000">',
        '<rect width="1240" height="1000" fill="#ffffff"/>',
        svg_text(56, 44, "Position-bias effects by judge", 24, "#111111", "700"),
        svg_text(56, 70, subtitle, 14, "#444444"),
    ]
    parts.extend(
        render_percent_panel(
            70,
            120,
            1100,
            "A. Display-position outcome composition",
            "Bars sum to 100% within each judge. Ties and invalid rows are shown explicitly.",
            judges,
            summary.get("by_judge", {}),
            (
                ("position_a_wins", "Position A", POSITION_COLORS["position_a_wins"]),
                ("position_b_wins", "Position B", POSITION_COLORS["position_b_wins"]),
                ("ties", "Tie", POSITION_COLORS["ties"]),
                ("invalid", "Invalid", POSITION_COLORS["invalid"]),
            ),
        )
    )
    parts.extend(
        render_percent_panel(
            70,
            400,
            1100,
            "B. Source-model preference composition",
            "Source A is the configured baseline model; here it is gpt-4.",
            judges,
            summary.get("by_judge", {}),
            (
                ("model_a_wins", "Source A", SOURCE_COLORS["model_a_wins"]),
                ("model_b_wins", "Source B", SOURCE_COLORS["model_b_wins"]),
                ("ties", "Tie", SOURCE_COLORS["ties"]),
                ("invalid", "Invalid", SOURCE_COLORS["invalid"]),
            ),
        )
    )
    parts.extend(render_pair_panel(70, 680, 1100, judges, summary))
    parts.extend(render_pair_legend(70, 950))
    parts.append("</svg>")
    return "\n".join(parts)


def render_percent_panel(x, y, width, title, subtitle, judges, counts_by_judge, keys):
    label_width = 170
    bar_left = x + label_width
    bar_width = width - label_width - 180
    row_step = 42
    bar_height = 18
    top = y + 78
    parts = [
        svg_text(x, y, title, 17, "#111111", "700"),
        svg_text(x, y + 24, subtitle, 13, "#444444"),
    ]
    add_legend(parts, bar_left, y + 42, [(label, color) for _, label, color in keys])
    for index, judge in enumerate(judges):
        counts = counts_by_judge.get(judge, {})
        total = counts.get("total", 0)
        row_y = top + index * row_step
        parts.append(svg_text(x, row_y + 14, model_label(judge), 12, "#111111"))
        cursor = bar_left
        for key, _, color in keys:
            value = counts.get(key, 0)
            segment_width = 0 if total == 0 else bar_width * value / total
            if segment_width:
                parts.append(rect(cursor, row_y, segment_width, bar_height, color))
            cursor += segment_width
        parts.append(outline_rect(bar_left, row_y, bar_width, bar_height))
        parts.append(svg_text(bar_left + bar_width + 14, row_y + 14, f"n={total}", 12, "#111111"))
    add_percent_axis(parts, bar_left, top + row_step * len(judges) + 10, bar_width)
    return parts


def render_pair_panel(x, y, width, judges, summary):
    label_width = 170
    bar_left = x + label_width
    bar_width = width - label_width - 180
    row_step = 42
    bar_height = 18
    top = y + 58
    paired_by_judge = summary.get("swapped_pair_analysis_by_judge", {})
    max_total = max([paired_by_judge.get(j, {}).get("total_pairs", 0) for j in judges] or [1])
    axis_max = nice_count(max_total)
    parts = [
        svg_text(x, y, "C. Swapped-pair pattern counts", 17, "#111111", "700"),
        svg_text(x, y + 24, "Counts are complete model_a_A/model_a_B pairs per judge.", 13, "#444444"),
    ]
    for index, judge in enumerate(judges):
        counts = paired_by_judge.get(judge, {})
        total = counts.get("total_pairs", 0)
        row_y = top + index * row_step
        cursor = bar_left
        parts.append(svg_text(x, row_y + 14, model_label(judge), 12, "#111111"))
        for key in PAIR_KEYS:
            value = counts.get(key, 0)
            segment_width = 0 if axis_max == 0 else bar_width * value / axis_max
            if segment_width:
                parts.append(rect(cursor, row_y, segment_width, bar_height, PAIR_COLORS[key]))
            cursor += segment_width
        parts.append(outline_rect(bar_left, row_y, bar_width, bar_height))
        parts.append(svg_text(bar_left + bar_width + 14, row_y + 14, f"{total} pairs", 12, "#111111"))
    add_count_axis(parts, bar_left, top + row_step * len(judges) + 10, bar_width, axis_max)
    return parts


def add_legend(parts, x, y, items):
    cursor = x
    for label, color in items:
        parts.append(rect(cursor, y + 6, 12, 12, color))
        parts.append(svg_text(cursor + 18, y + 17, label, 12, "#111111"))
        cursor += max(92, len(label) * 7 + 34)


def render_pair_legend(x, y):
    parts = []
    cursor = x
    for key in PAIR_KEYS:
        label = PAIR_LABELS[key]
        parts.append(rect(cursor, y, 12, 12, PAIR_COLORS[key]))
        parts.append(svg_text(cursor + 18, y + 11, label, 11, "#111111"))
        cursor += max(92, len(label) * 7 + 34)
    return parts


def add_percent_axis(parts, x, y, width):
    for tick in (0.0, 0.5, 1.0):
        tick_x = x + width * tick
        parts.append(f'<line x1="{tick_x:.1f}" y1="{y:.1f}" x2="{tick_x:.1f}" y2="{y + 5:.1f}" stroke="#111111"/>')
        parts.append(svg_text(tick_x, y + 19, pct(tick), 11, "#111111", "400", "middle"))
    parts.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + width:.1f}" y2="{y:.1f}" stroke="#111111"/>')


def add_count_axis(parts, x, y, width, axis_max):
    step = 20 if axis_max > 60 else 10
    for tick in range(0, axis_max + 1, step):
        tick_x = x + width * tick / axis_max
        parts.append(f'<line x1="{tick_x:.1f}" y1="{y:.1f}" x2="{tick_x:.1f}" y2="{y + 5:.1f}" stroke="#111111"/>')
        parts.append(svg_text(tick_x, y + 19, str(tick), 11, "#111111", "400", "middle"))
    parts.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + width:.1f}" y2="{y:.1f}" stroke="#111111"/>')


def nice_count(value):
    if value <= 10:
        return 10
    if value <= 50:
        return ((value + 9) // 10) * 10
    return ((value + 19) // 20) * 20


def pct(value):
    return f"{value * 100:.0f}%"


def model_label(model):
    return MODEL_LABELS.get(model, model)


def rect(x, y, width, height, color):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="{color}"/>'


def outline_rect(x, y, width, height):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="none" stroke="#111111" stroke-width="0.6"/>'


def svg_text(x, y, text, size, color, weight="400", anchor="start"):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{color}">{escape(str(text))}</text>'
    )
