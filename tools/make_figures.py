"""Render the README figures from committed result records (light and dark SVG variants).

Add a new optimization step to STEPS (record file + arm) and rerun; figures never contain numbers that
are not in `results/`.
"""

import json
from pathlib import Path
import sys
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures"
FORMAT = "svg"
GALAXY = ROOT / "results/2026-10-06/galaxy"
TARGET_TOKENS_PER_SECOND = 5.0

# Comparable steps only: Qwen3.6-35B-A3B UD-Q4_K_M, 64 forced greedy tokens, identical routing per arm.
STEPS = [
    ("1 GiB cache\nbuffered, 4 I/O", "target-io-sweep.json", "buf_t4"),
    ("direct I/O\n16 I/O threads", "target-io-sweep.json", "dio_t16"),
    ("+ router\nlookahead", "target-prefetch-sweep.json", "prefetch"),
    ("+ 2 GiB\nexpert cache", "target-threads-budget.json", "t4_c2048"),
    ("+ zero-copy\ndirect reads", "target-zero-copy.json", "direct_zero_copy"),
    ("+ 2-bit experts\n(UD-IQ2_M)*", "../../2026-10-07/galaxy/iq2m-sweep.json", "plain_c2048"),
    ("+ expert slots\n(UD-IQ2_M)*", "../../2026-10-08/galaxy/slots-decode.json", "slots"),
]

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e", grid="#e4e3dd", s1="#2a78d6", s2="#eb6834"),
    "dark": dict(surface="#1a1a19", text="#ffffff", muted="#c3c2b7", grid="#3a3935", s1="#3987e5", s2="#d95926"),
}


def rows(record, arm):
    data = json.loads((GALAXY / record).read_text(encoding="utf-8"))
    selected = [r for r in data["rows"] if r.get("arm") == arm and "decode_ms" in r]
    if not selected:
        raise ValueError(f"No rows for {arm} in {record}")
    return selected


def style(ax, theme):
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0)
    ax.grid(axis="y", color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def figure(theme, width, height):
    fig, ax = plt.subplots(figsize=(width, height), dpi=100)
    fig.patch.set_facecolor(theme["surface"])
    style(ax, theme)
    return fig, ax


def title(fig, theme, text, subtitle):
    fig.text(0.02, 0.965, text, color=theme["text"], fontsize=13, fontweight="bold", va="top")
    fig.text(0.02, 0.895, subtitle, color=theme["muted"], fontsize=9, va="top")


def progress(theme, name):
    fig, ax = figure(theme, 10, 4.6)
    labels, means = [], []
    for index, (label, record, arm) in enumerate(STEPS):
        values = [r["decode_tokens_per_second"] for r in rows(record, arm)]
        thermal = sorted({r["thermal_before"] for r in rows(record, arm)} | {r["thermal_after"] for r in rows(record, arm)})
        labels.append(f"{label}\nthermal {thermal[0]}–{thermal[-1]}" if len(thermal) > 1 else f"{label}\nthermal {thermal[0]}")
        means.append(mean(values))
        ax.bar(index, means[-1], width=0.55, color=theme["s1"], zorder=2)
        ax.scatter([index] * len(values), values, s=22, color=theme["text"], zorder=3, linewidths=0)
        ax.text(index, max(values) + 0.12, f"{means[-1]:.2f}", ha="center", color=theme["text"], fontsize=10)
    ax.axhline(TARGET_TOKENS_PER_SECOND, color=theme["muted"], linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
    ax.text(len(STEPS) - 0.55, TARGET_TOKENS_PER_SECOND + 0.08, "5 tokens/s target", ha="right",
            color=theme["muted"], fontsize=9)
    ax.set_xticks(range(len(STEPS)), labels, color=theme["muted"], fontsize=8.5)
    ax.set_ylim(0, 6.6)
    ax.set_ylabel("decode tokens/s", color=theme["muted"], fontsize=9)
    title(fig, theme, "Qwen3.6-35B-A3B (22 GB) streamed on a Galaxy S24+ (Exynos 2400)",
          "Mean of 2–3 alternating runs (dots) · 64 greedy tokens · *different quantization, quality evaluated separately")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.2)
    fig.savefig(OUT / f"{name}-{theme_name(theme)}.{FORMAT}", facecolor=theme["surface"])
    plt.close(fig)


def breakdown(theme, name):
    fig, ax = figure(theme, 9, 4.2)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=theme["grid"], linewidth=0.8)
    for index, (label, record, arm) in enumerate(STEPS):
        selected = rows(record, arm)
        tokens = len(selected[0]["tokens"])
        wait = mean(r["decode_io_wait_ms"] for r in selected) / tokens
        other = mean(r["decode_ms"] for r in selected) / tokens - wait
        y = len(STEPS) - 1 - index
        # Surface-colored edges leave the 2 px gap between stacked segments.
        ax.barh(y, other, height=0.55, color=theme["s1"], edgecolor=theme["surface"], linewidth=2, zorder=2)
        ax.barh(y, wait, left=other, height=0.55, color=theme["s2"], edgecolor=theme["surface"], linewidth=2, zorder=2)
        ax.text(other + wait + 6, y, f"{other + wait:.0f} ms", va="center", color=theme["text"], fontsize=9)
    ax.axvline(1000 / TARGET_TOKENS_PER_SECOND, color=theme["muted"], linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(1000 / TARGET_TOKENS_PER_SECOND + 4, -0.62, "200 ms = 5 tokens/s", color=theme["muted"], fontsize=9)
    ax.set_ylim(-0.75, len(STEPS) - 0.5)
    ax.set_yticks(range(len(STEPS)), [label.replace("\n", " ") for label, _, _ in STEPS][::-1], color=theme["muted"])
    ax.set_xlabel("milliseconds per decoded token (mean)", color=theme["muted"], fontsize=9)
    ax.set_xlim(0, 460)
    handles = [plt.Rectangle((0, 0), 1, 1, color=theme["s1"]), plt.Rectangle((0, 0), 1, 1, color=theme["s2"])]
    legend = ax.legend(handles, ["compute, scheduling and cache bookkeeping", "blocked waiting for expert reads"],
                       loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=2, frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(theme["text"])
    title(fig, theme, "Where a decoded token's time goes",
          "Decode thread blocked on storage versus everything else; prefetch reads overlap compute when they hit")
    fig.subplots_adjust(left=0.24, right=0.97, top=0.74, bottom=0.14)
    fig.savefig(OUT / f"{name}-{theme_name(theme)}.{FORMAT}", facecolor=theme["surface"])
    plt.close(fig)


def models(theme, name):
    soak = json.loads((ROOT / "results/2026-10-05/soak-summary.json").read_text(encoding="utf-8"))
    best = max(rows("target-zero-copy.json", "direct_zero_copy"), key=lambda r: r["decode_tokens_per_second"])
    first = json.loads((GALAXY / "target-first-run.json").read_text(encoding="utf-8"))
    asym = max(rows("../../2026-10-07/galaxy/iq2m-sweep.json", "plain_c2048"), key=lambda r: r["decode_tokens_per_second"])
    gib = 1024**3
    entries = [
        (f'Qwen3-0.6B Q8_0, resident\n{soak["decode_tokens_per_second_weighted"]:.1f} tokens/s (15-min run)',
         639446688 / gib, soak["peak_rss_bytes"] / gib),
        (f'Qwen3.6-35B-A3B UD-Q4_K_M, streamed\n{best["decode_tokens_per_second"]:.2f} tokens/s best (2 GiB cache)',
         first["configuration"]["model_file_bytes"] / gib, best["peak_rss_bytes"] / gib),
        (f'Qwen3.6-35B-A3B UD-IQ2_M + MTP, streamed\n{asym["decode_tokens_per_second"]:.2f} tokens/s best (2 GiB cache)',
         11882969376 / gib, asym["peak_rss_bytes"] / gib),
    ]
    fig, ax = figure(theme, 9, 4.4)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=theme["grid"], linewidth=0.8)
    for index, (label, size, rss) in enumerate(entries):
        y = len(entries) - 1 - index
        ax.barh(y + 0.17, size, height=0.3, color=theme["s1"], zorder=2)
        ax.barh(y - 0.17, rss, height=0.3, color=theme["s2"], zorder=2)
        ax.text(size + 0.3, y + 0.17, f"{size:.1f} GiB weights", va="center", color=theme["text"], fontsize=9)
        ax.text(rss + 0.3, y - 0.17, f"{rss:.1f} GiB peak RSS", va="center", color=theme["text"], fontsize=9)
    memory_total = 11472024 * 1024 / gib  # MemTotal reported by the device (12 GB nominal)
    ax.axvline(memory_total, color=theme["muted"], linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(memory_total + 0.2, len(entries) - 0.52, "phone RAM (MemTotal)", color=theme["muted"], fontsize=9)
    ax.set_ylim(-0.5, len(entries) - 0.35)
    ax.set_yticks(range(len(entries)), [e[0] for e in entries][::-1], color=theme["muted"])
    ax.set_xlim(0, 30)
    ax.set_xlabel("GiB", color=theme["muted"], fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=theme["s1"]), plt.Rectangle((0, 0), 1, 1, color=theme["s2"])]
    legend = ax.legend(handles, ["model file", "process peak RSS"], loc="lower right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(theme["text"])
    title(fig, theme, "Models run on the Galaxy so far",
          f"The 35B Q4 file is {entries[1][1] / memory_total:.1f}x the phone's RAM; streamed processes peak at {entries[1][2]:.1f} and {entries[2][2]:.1f} GiB")
    fig.subplots_adjust(left=0.38, right=0.97, top=0.8, bottom=0.14)
    fig.savefig(OUT / f"{name}-{theme_name(theme)}.{FORMAT}", facecolor=theme["surface"])
    plt.close(fig)


def theme_name(theme):
    return next(name for name, value in THEMES.items() if value is theme)


def main():
    global OUT, FORMAT
    if len(sys.argv) == 3 and sys.argv[1] == "--preview":
        OUT, FORMAT = Path(sys.argv[2]), "png"  # raster previews for visual checks, never committed
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "svg.hashsalt": "eqt"})
    for theme in THEMES.values():
        progress(theme, "decode-progress")
        breakdown(theme, "token-time")
        models(theme, "models")
    print(sorted(p.name for p in OUT.glob("*." + FORMAT)))


if __name__ == "__main__":
    main()
