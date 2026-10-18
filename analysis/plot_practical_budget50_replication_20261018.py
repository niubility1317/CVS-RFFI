"""Plot only complete frozen aggregates for the fixed three-seed E50 study.

Usage:
  python plot_practical_budget50_replication_20261018.py \
      --aggregates <collected-analysis-directory> --output-dir <new-figure-directory>

Inputs are source_summary.json, formal_summary.json and complete.json only.
No IQ, checkpoints, packet predictions, truth, logs or remote resources are read.
All points use the fixed seeds 2026092701/02/03, in that order. Error bars are
sample standard deviations (ddof=1), never confidence intervals. Accuracy is
shown in percent; paired changes and budget interactions are absolute percentage
points (pp). This is descriptive plotting, not model selection or a release gate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

RUN = "20261018-phase1-FcrossRX-As-budget-e50-s0203-m4-r01"
SEEDS = (2026092701, 2026092702, 2026092703)
CONDITIONS = ("NOAS30", "AS30", "NOAS50", "AS50")
VIEWS = (
    "clean", "practical_high", "practical_mid", "practical_low_suburban",
    "practical_high_urban", "practical_mid_urban", "practical_low_urban",
)
VIEW_LABELS = (
    "Clean", "Practical\nhigh", "Practical\nmid", "Low\nsuburban",
    "High\nurban", "Mid\nurban", "Low\nurban",
)
PAIRS = {
    "AS50_minus_NOAS50": ("AS50", "NOAS50"),
    "AS30_minus_NOAS30": ("AS30", "NOAS30"),
    "AS50_minus_AS30": ("AS50", "AS30"),
    "NOAS50_minus_NOAS30": ("NOAS50", "NOAS30"),
}
COUNT = 168000
BLUE = "#0072B2"
GREY = "#737373"
PURPLE = "#8864A5"
STEMS = ("matched_e50_three_seed", "matched_e30_to_e50_budget")


def read_json(path):
    def invalid(value):
        raise ValueError("Non-finite JSON constant in " + path.name + ": " + value)
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=invalid)


def finite(value, location):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected numeric value at " + location)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite value at " + location)
    return result


def check_stats(saved, values, location):
    """Reconcile saved three-seed statistics with every plotted seed value."""
    values = np.asarray(values, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("All three finite seed values required at " + location)
    if saved["model_seed_count"] != 3:
        raise ValueError("Partial seed statistics at " + location)
    actual = np.array([finite(v, location + ".values") for v in saved["values"]])
    if actual.shape != (3,) or not np.allclose(actual, values, atol=1e-9, rtol=0):
        raise ValueError("Saved seed order/values differ at " + location)
    expected = {
        "mean": values.mean(), "sample_SD": values.std(ddof=1),
        "minimum": values.min(), "maximum": values.max(),
    }
    for key, value in expected.items():
        if not math.isclose(finite(saved[key], location + "." + key),
                            float(value), rel_tol=0, abs_tol=1e-9):
            raise ValueError("Saved three-seed statistic differs at " + location + "." + key)


def load_frozen(aggregates):
    """Read only saved public aggregates; reject incomplete three-seed results."""
    source = read_json(aggregates / "source_summary.json")
    formal = read_json(aggregates / "formal_summary.json")
    complete = read_json(aggregates / "complete.json")
    if (complete.get("status") != "ANALYZED"
            or complete.get("training_run") != RUN
            or complete.get("model_seed_count") != 3
            or not all(complete.get(k) is True for k in
                       ("source_complete", "formal_complete", "log_records_complete"))):
        raise ValueError("Complete frozen three-seed analysis is required before plotting")
    if (source.get("status") != "SOURCE_ANALYZED"
            or source.get("run_id") != RUN
            or source.get("model_seeds") != list(SEEDS)
            or source.get("target_scores_read") is not False):
        raise ValueError("Source-first aggregate or complete fixed seed set differs")
    if (formal.get("status") != "FORMAL_RECOUNT_COMPLETE"
            or formal.get("run_id") != RUN
            or formal.get("model_seed_count") != 3
            or formal.get("model_seeds") != list(SEEDS)
            or formal.get("independent_recount") != "VERIFIED"
            or formal.get("target_feedback") is not False):
        raise ValueError("Complete independently recounted formal aggregates are required")

    accuracy = {condition: np.empty((3, len(VIEWS))) for condition in CONDITIONS}
    deltas = {name: np.empty((3, len(VIEWS))) for name in PAIRS}
    interaction = np.empty((3, len(VIEWS)))
    for si, seed in enumerate(SEEDS):
        for condition in CONDITIONS:
            label = condition + "-s" + str(seed)
            row = formal["summary"][label]
            for vi, view in enumerate(VIEWS):
                metric = row["views"][view]
                value = finite(metric["accuracy"], label + "." + view + ".accuracy")
                if metric["count"] != COUNT or not 0 <= value <= 1:
                    raise ValueError("Expected fixed formal count and accuracy fraction at " + label)
                accuracy[condition][si, vi] = value * 100
            # Also require an actual finite source record for every logical row.
            finite(source["summary"][label]["full_V_clean"], "source." + label)
        for name, (new, control) in PAIRS.items():
            for vi, view in enumerate(VIEWS):
                expected = accuracy[new][si, vi] - accuracy[control][si, vi]
                actual = finite(formal["per_seed"][str(seed)]["paired_accuracy_pp"][name][view],
                                str(seed) + "." + name + "." + view)
                if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9):
                    raise ValueError("Paired accuracy does not match saved absolute accuracy")
                deltas[name][si, vi] = actual
        for vi, view in enumerate(VIEWS):
            expected = (deltas["AS50_minus_NOAS50"][si, vi]
                        - deltas["AS30_minus_NOAS30"][si, vi])
            actual = finite(formal["per_seed"][str(seed)]["budget_interaction_pp"][view],
                            str(seed) + ".interaction." + view)
            if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9):
                raise ValueError("Budget interaction differs from the paired difference of differences")
            interaction[si, vi] = actual
    for vi, view in enumerate(VIEWS):
        for condition in CONDITIONS:
            check_stats(formal["absolute_metrics_three_seed"][condition][view]["accuracy"],
                        accuracy[condition][:, vi] / 100, condition + "." + view)
        for name in PAIRS:
            check_stats(formal["paired_accuracy_pp_three_seed"][name][view],
                        deltas[name][:, vi], name + "." + view)
        check_stats(formal["budget_interaction_pp_three_seed"][view],
                    interaction[:, vi], "interaction." + view)
    return accuracy, deltas, interaction


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .8, "lines.linewidth": 1.5,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def delta_limits(values, include_guard=False):
    values = np.asarray(values, dtype=float)
    extras = [0., -.1] if include_guard else [0.]
    lo = min(float(values.min()), *extras)
    hi = max(float(values.max()), *extras)
    span = max(hi - lo, .3)
    return lo - .15 * span, hi + .22 * span


def save(fig, output, stem, dpi):
    # Exclusive creation preserves any earlier output, including a partial run.
    for extension in ("png", "pdf"):
        path = output / (stem + "." + extension)
        with path.open("xb") as stream:
            fig.savefig(stream, format=extension, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def matched_figure(deltas):
    selected = ("clean", "practical_mid_urban", "practical_low_urban")
    titles = ("Clean", "Mid urban", "Low urban")
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.2))
    values = deltas["AS50_minus_NOAS50"]
    for index, (ax, view, title) in enumerate(zip(axes, selected, titles)):
        y = values[:, VIEWS.index(view)]
        mean, sd = y.mean(), y.std(ddof=1)
        ax.scatter(np.arange(3), y, s=45, color=BLUE, zorder=4)
        ax.errorbar(3, mean, yerr=sd, fmt="D", color="#202020", capsize=5,
                    markersize=6, linewidth=1.6, zorder=5)
        ax.axhline(0, color="#989898", linewidth=.9, zorder=1)
        if view == "clean":
            ax.axhline(-.1, color="#A76634", linestyle="--", linewidth=1.1,
                       label="Stage bound: -0.1 pp", zorder=2)
            ax.legend(loc="lower left", fontsize=8)
        for x, value in enumerate((*y, mean)):
            ax.annotate(f"{value:+.3f}", (x, value), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=9)
        ax.set_xticks(range(4), ("Seed 01", "Seed 02", "Seed 03", "Mean +/- SD"))
        ax.tick_params(axis="x", labelsize=9)
        ax.set_xlim(-.45, 3.55)
        ax.set_ylim(*delta_limits([*y, mean - sd, mean + sd], view == "clean"))
        ax.set_title(chr(ord("a") + index) + "  " + title, loc="left", fontweight="bold")
        ax.set_ylabel("As - NoAs accuracy (absolute pp)")
        ax.grid(axis="y", color="#E5E5E5", linewidth=.65, zorder=0)
    fig.suptitle("Matched E50 practical augmentation: all three fixed seeds", y=.99, fontsize=13)
    fig.text(.5, .02,
             "Fixed seeds 2026092701/02/03. Diamond/error bars: mean +/- sample SD (ddof=1), not a CI.\n"
             "Same E50 budget, data roles and default head; each panel has its own continuous y scale.",
             ha="center", va="bottom", fontsize=9)
    fig.subplots_adjust(left=.07, right=.985, top=.84, bottom=.23, wspace=.34)
    return fig


def plot_seed_cloud(ax, x, values, color, offset=0.):
    for si, marker in enumerate(("o", "^", "s")):
        ax.scatter(x + offset + (si - 1) * .025, values[si], color=color,
                   marker=marker, s=19, alpha=.48, linewidths=0, zorder=3)


def budget_figure(accuracy, deltas, interaction):
    x = np.arange(len(VIEWS))
    fig = plt.figure(figsize=(12.4, 8.7))
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1), hspace=.52, wspace=.27)
    absolute = fig.add_subplot(grid[0, :])
    budget = fig.add_subplot(grid[1, 0])
    inter = fig.add_subplot(grid[1, 1])
    for condition, label, color, linestyle, marker in (
        ("NOAS30", "NoAs E30", GREY, "--", "o"),
        ("NOAS50", "NoAs E50", GREY, "-", "s"),
        ("AS30", "As E30", BLUE, "--", "o"),
        ("AS50", "As E50", BLUE, "-", "s"),
    ):
        y = accuracy[condition]
        absolute.errorbar(x, y.mean(axis=0), yerr=y.std(axis=0, ddof=1),
                          color=color, linestyle=linestyle, marker=marker,
                          capsize=3, markersize=4, label=label, zorder=3)
    absolute.set_ylim(0, 100)
    absolute.set_ylabel("Accuracy (%)")
    absolute.set_title("a  Absolute accuracy: matched E30 and E50", loc="left", fontweight="bold")
    absolute.set_xticks(x, VIEW_LABELS)
    absolute.legend(ncol=4, loc="lower center", fontsize=9)
    absolute.grid(axis="y", color="#E5E5E5", linewidth=.65, zorder=0)

    lows, highs = [], []
    for name, label, color, offset in (
        ("NOAS50_minus_NOAS30", "NoAs: E50 - E30", GREY, -.12),
        ("AS50_minus_AS30", "As: E50 - E30", BLUE, .12),
    ):
        y = deltas[name]; mean, sd = y.mean(axis=0), y.std(axis=0, ddof=1)
        plot_seed_cloud(budget, x, y, color, offset)
        budget.errorbar(x + offset, mean, yerr=sd, color=color, fmt="D-",
                        capsize=3, markersize=4, label=label, zorder=4)
        lows.extend(np.minimum(y.min(axis=0), mean - sd))
        highs.extend(np.maximum(y.max(axis=0), mean + sd))
    budget.set_ylim(*delta_limits(lows + highs))
    budget.set_title("b  Additional budget in each recipe", loc="left", fontweight="bold")
    budget.set_ylabel("E50 - E30 accuracy (absolute pp)")
    budget.legend(loc="best", fontsize=8)
    mean, sd = interaction.mean(axis=0), interaction.std(axis=0, ddof=1)
    plot_seed_cloud(inter, x, interaction, PURPLE)
    inter.errorbar(x, mean, yerr=sd, color=PURPLE, fmt="D-", capsize=3,
                   markersize=4, zorder=4)
    inter.set_ylim(*delta_limits([*interaction.ravel(), *(mean - sd), *(mean + sd)]))
    inter.set_title("c  Budget interaction", loc="left", fontweight="bold")
    inter.set_ylabel("Difference of paired changes (absolute pp)")
    for ax in (budget, inter):
        ax.axhline(0, color="#989898", linewidth=.9, zorder=1)
        ax.set_xticks(x, VIEW_LABELS, fontsize=8)
        ax.grid(axis="y", color="#E5E5E5", linewidth=.65, zorder=0)
        ax.set_xlim(-.45, len(VIEWS) - .55)
    fig.suptitle("E30 to E50: preserve the NoAs budget control", y=.995, fontsize=13)
    handles = [Line2D([], [], marker=marker, linestyle="None", color=GREY,
                      markersize=4, label="Seed " + str(si + 1).zfill(2))
               for si, marker in enumerate(("o", "^", "s"))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .052),
               ncol=3, fontsize=8)
    fig.text(.5, .012,
             "Lines/diamonds and error bars: three-seed mean +/- sample SD (ddof=1), not a CI. Faint points: every seed.\n"
             "Interaction = (As50 - NoAs50) - (As30 - NoAs30) = (As50 - As30) - (NoAs50 - NoAs30).\n"
             "Own FINAL30 + 20 epochs with reset optimizer/cosine; this is not a continuous cosine-50 experiment.",
             ha="center", va="bottom", fontsize=8.5)
    fig.subplots_adjust(left=.075, right=.98, top=.87, bottom=.19)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregates", required=True, type=Path,
                        help="Directory containing complete saved three-seed analysis JSON")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Local figure directory; existing figure files are never overwritten")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dpi < 150:
        parser.error("Use at least 150 dpi for a readable standalone scientific figure")
    accuracy, deltas, interaction = load_frozen(args.aggregates)
    for stem in STEMS:
        for extension in ("png", "pdf"):
            target = args.output_dir / (stem + "." + extension)
            if target.exists():
                raise FileExistsError("Existing figure preserved: " + str(target))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    save(matched_figure(deltas), args.output_dir, STEMS[0], args.dpi)
    save(budget_figure(accuracy, deltas, interaction), args.output_dir, STEMS[1], args.dpi)
    print("Saved two PNG/PDF figures from all three fixed seeds; mean +/- sample SD, not CI.")


if __name__ == "__main__":
    main()
