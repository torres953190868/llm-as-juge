from xml.sax.saxutils import escape


MODEL_LABELS = {
    "deepseek-v4-flash": "DeepSeek",
    "gemini-3-flash-preview": "Gemini Flash",
    "mimo-v2-pro": "MiMo Pro",
}


def render_academic_svg(summary):
    judges = summary.get("selected_judge_models") or list(summary["by_judge"])
    by_judge = summary["by_judge"]
    count_keys = [
        ("long_wins", "#4c78a8"),
        ("short_wins", "#b55d60"),
        ("ties", "#6b6b6b"),
        ("invalid", "#b0b0b0"),
    ]
    max_count = max(
        [by_judge.get(judge, {}).get(key, 0) for judge in judges for key, _ in count_keys]
        or [1]
    )
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="620" viewBox="0 0 1120 620">',
        '<rect width="1120" height="620" fill="#ffffff"/>',
        svg_text(56, 44, "Length-bias outcomes by LLM judge", 24, "#111111", "700"),
        svg_text(
            56,
            70,
            "Filtered parsed judgments; bars report counts and rates by selected judge.",
            14,
            "#444444",
        ),
    ]
    parts.extend(
        render_panel(
            70,
            120,
            470,
            330,
            "A. Outcome counts",
            "Count",
            judges,
            count_values(judges, by_judge),
            max(max_count, 1),
            False,
        )
    )
    parts.extend(
        render_panel(
            620,
            120,
            410,
            330,
            "B. Length-bias rates",
            "Percent",
            judges,
            rate_values(judges, by_judge),
            100,
            True,
        )
    )
    parts.extend(render_legend(70, 500))
    parts.append("</svg>")
    return "\n".join(parts)


def svg_text(x, y, text, size=14, color="#1f2933", weight="400", anchor="start"):
    return (
        f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
        f'font-family="Times New Roman, Times, serif" font-weight="{weight}" '
        f'text-anchor="{anchor}">{escape(str(text))}</text>'
    )


def count_values(judges, by_judge):
    keys = [
        ("long_wins", "#4c78a8"),
        ("short_wins", "#b55d60"),
        ("ties", "#6b6b6b"),
        ("invalid", "#b0b0b0"),
    ]
    return [
        [(by_judge.get(judge, {}).get(key, 0), color) for key, color in keys]
        for judge in judges
    ]


def rate_values(judges, by_judge):
    values = []
    for judge in judges:
        counts = by_judge.get(judge, {})
        values.append(
            [
                ((counts.get("long_win_rate") or 0.0) * 100, "#4c78a8"),
                ((counts.get("net_length_preference") or 0.0) * 100, "#6a8f5a"),
            ]
        )
    return values


def render_panel(x, y, width, height, title, y_label, judges, values, max_value, allow_negative):
    left = x + 58
    top = y + 42
    chart_width = width - 80
    chart_height = height - 90
    baseline = top + chart_height if not allow_negative else top + chart_height / 2
    min_value = -100 if allow_negative else 0
    scale = chart_height / (max_value - min_value)
    group_width = chart_width / max(len(judges), 1)
    bar_count = max((len(item) for item in values), default=1)
    bar_width = min(18, group_width / (bar_count + 2))
    parts = [
        svg_text(x, y, title, 17, "#111111", "700"),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#111111"/>',
        f'<line x1="{left}" y1="{baseline:.1f}" x2="{left + chart_width}" y2="{baseline:.1f}" stroke="#111111"/>',
        svg_text(x, top + chart_height / 2, y_label, 13, "#111111", "400", "middle"),
    ]
    add_ticks(parts, left, top, chart_height, baseline, scale, max_value, allow_negative)
    for index, judge in enumerate(judges):
        center = left + group_width * index + group_width / 2
        start = center - (bar_count * bar_width) / 2
        for offset, (value, color) in enumerate(values[index] if index < len(values) else []):
            y0 = baseline - (value * scale)
            parts.append(
                f'<rect x="{start + offset * bar_width:.1f}" y="{min(y0, baseline):.1f}" '
                f'width="{bar_width - 2:.1f}" height="{abs(baseline - y0):.1f}" fill="{color}"/>'
            )
        parts.append(svg_text(center, top + chart_height + 24, model_label(judge), 12, "#111111", "400", "middle"))
    return parts


def add_ticks(parts, left, top, chart_height, baseline, scale, max_value, allow_negative):
    step = 25 if allow_negative else max(1, int(max_value / 4) or 1)
    ticks = list(range(0, int(max_value) + 1, step))
    if allow_negative:
        ticks.extend([-50, -100])
    for tick in ticks:
        y = baseline - (tick * scale)
        if top <= y <= top + chart_height:
            parts.append(f'<line x1="{left - 4}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#111111"/>')
            parts.append(svg_text(left - 8, y + 4, str(tick), 11, "#111111", "400", "end"))


def render_legend(x, y):
    items = [
        ("Long", "#4c78a8"),
        ("Short", "#b55d60"),
        ("Tie", "#6b6b6b"),
        ("Invalid", "#b0b0b0"),
        ("Net preference", "#6a8f5a"),
    ]
    parts = [svg_text(x, y, "Legend", 14, "#111111", "700")]
    for index, (label, color) in enumerate(items):
        lx = x + index * 150
        parts.append(f'<rect x="{lx}" y="{y + 18}" width="14" height="14" fill="{color}"/>')
        parts.append(svg_text(lx + 20, y + 30, label, 13, "#111111"))
    return parts


def model_label(model):
    return MODEL_LABELS.get(model, model)
