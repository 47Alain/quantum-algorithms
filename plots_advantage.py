"""Three figures on quantum vs classical algorithmic progress.

    9_improvement_rate_grid.png   compound annual improvement rates, as a
                                  distribution over problem families, faceted
                                  by category and problem size
    10_relative_speedup.png       25th/50th/75th percentile speedup over a
                                  fixed baseline, as a step function of year
    11_quantum_coverage.png       how the problem families divide into
                                  quantum-absent / classical-ahead /
                                  quantum-ahead, swept over problem size

All three read the parameter-consistent dataset assembled by
:mod:`src.dataset`, so every runtime inside a family is a function of the same
quantity.  Run ``python validate_dataset.py`` first to see the checks behind
that claim, and ``Plots/jun2026/runtime_dataset_audit.csv`` for the row-level
record of what was kept and dropped.

Run: ``python plots_advantage.py``
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.advantage import (  # noqa: E402
    BAND_COLORS,
    BAND_KEYS,
    BAND_LABELS,
    BUCKET_LABELS,
    CATEGORY_COLORS,
    CATEGORY_ORDER,
    PROBLEM_SIZES,
    SIZE_LABELS,
    best_by_category,
    bucket_rates,
    category_label,
    coverage_profile,
    improvement_rates,
    overhead_tolerance,
    problems_with,
    quantum_crossovers,
    runtime_table,
    speedup_series,
)
from src.dataset import build_dataset  # noqa: E402

OUT_DIR = REPO_ROOT / "Plots" / "jun2026"
OUT_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
})

_PERCENT = FuncFormatter(lambda v, _: f"{v * 100:.0f}%")


def _sci(size: float) -> str:
    return f"$10^{{{int(np.log10(size))}}}$"


# ---------------------------------------------------------------------------
# Figure 9 -- distribution of compound annual improvement rates
# ---------------------------------------------------------------------------

def plot_improvement_rate_grid(ds) -> Path:
    rates = {size: improvement_rates(ds, size) for size in PROBLEM_SIZES}

    fig, axes = plt.subplots(
        len(CATEGORY_ORDER), len(PROBLEM_SIZES),
        figsize=(9.0, 7.2), sharex=True, sharey=True)

    counts_note = {}
    for r, category in enumerate(CATEGORY_ORDER):
        for c, size in enumerate(PROBLEM_SIZES):
            ax = axes[r, c]
            frame = rates[size]
            subset = frame[(frame["category"] == category) & frame["rate"].notna()]
            shares = bucket_rates(subset["rate"])
            ax.bar(range(len(BUCKET_LABELS)), shares.values,
                   color=CATEGORY_COLORS[category], edgecolor="white", width=0.78)
            counts_note[(category, size)] = len(subset)

            if r == 0:
                ax.set_title(f"Problem size = {SIZE_LABELS[size]}\n({_sci(size)})",
                             fontsize=9.5)
            if c == 0:
                ax.set_ylabel(f"{category_label(category)}\n\nshare of problems",
                              fontsize=8.5)
            ax.text(0.97, 0.93, f"{len(subset)} problems", ha="right",
                    va="top", transform=ax.transAxes, fontsize=7.5, color="#555555")
            ax.yaxis.set_major_formatter(_PERCENT)
            ax.set_ylim(0, 0.6)
            ax.grid(axis="y", alpha=0.18, linewidth=0.6)
            ax.set_axisbelow(True)

    for ax in axes[-1]:
        ax.set_xticks(range(len(BUCKET_LABELS)))
        ax.set_xticklabels(BUCKET_LABELS, rotation=90, fontsize=8)

    fig.suptitle("Compound annual rate of algorithmic improvement",
                 fontsize=12, y=0.985)
    fig.text(0.5, 0.945,
             "From each problem's earliest catalogued algorithm to the one holding "
             "the record, at a fixed problem size",
             ha="center", fontsize=8.5, color="#444444")
    fig.text(0.5, -0.075,
             "A problem is a (family, variation) pair, counted only where it has at least two catalogued algorithms. "
             "Quantum rows are\ncircuit depth, classical serial is sequential running time, classical parallel is total "
             "work; each problem is measured in one\nconsistent parameter throughout. Two caveats on the quantum row: "
             "its problems are a set someone chose to attack, measured\nover a shorter history (median 11 years against "
             "23), and where both algorithms are exponential a small improvement in the\nbase produces an enormous "
             "ratio at fixed n.",
             ha="center", fontsize=7.5, color="#666666")
    fig.tight_layout(rect=(0, 0, 1, 0.935))

    path = OUT_DIR / "9_improvement_rate_grid.png"
    fig.savefig(path)
    plt.close(fig)
    _report_rate_grid(rates)
    return path


def _report_rate_grid(rates: dict[float, pd.DataFrame]) -> None:
    print("\nFigure 9 -- median compound annual improvement rate")
    for size in PROBLEM_SIZES:
        frame = rates[size]
        parts = []
        for category in CATEGORY_ORDER:
            sub = frame[(frame["category"] == category) & frame["rate"].notna()]
            if len(sub):
                parts.append(f"{category_label(category)} {sub['rate'].median():6.1%}"
                             f" (n={len(sub)})")
        print(f"   n = {size:>10.0e}:  " + "   ".join(parts))


# ---------------------------------------------------------------------------
# Figure 10 -- relative speedup percentiles over time
# ---------------------------------------------------------------------------

def plot_relative_speedup(ds, size: float = 1e6) -> Path:
    problems = problems_with(runtime_table(ds, size), "quantum", "serial")
    years = np.arange(1950, 2027)
    series = speedup_series(ds, problems, size, years)

    percentiles = (25, 50, 75)
    fig, axes = plt.subplots(len(percentiles), 1, figsize=(7.2, 8.0),
                             sharex=True, sharey=True)

    # How many of the problems actually have an algorithm in each category;
    # the rest sit at 1 and still count toward the percentiles.
    present = {c: int((f.values > 1.0).any(axis=1).sum())
               for c, f in series.items()}

    ceiling, first_move = 1.0, years[-1]
    for ax, pct in zip(axes, percentiles):
        for category in CATEGORY_ORDER:
            frame = series.get(category)
            if frame is None or frame.empty:
                continue
            curve = np.percentile(frame.values, pct, axis=0)
            ceiling = max(ceiling, curve.max())
            moved = np.flatnonzero(curve > 1.0)
            if moved.size:
                first_move = min(first_move, years[moved[0]])
            ax.step(years, curve, where="post", linewidth=1.9,
                    color=CATEGORY_COLORS[category],
                    label=f"{category_label(category)}  "
                          f"({present.get(category, 0)} of {len(frame)} have one)")
        ax.set_yscale("log")
        ax.set_title(f"{pct}th percentile", fontsize=9.5)
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.axhline(1.0, color="#bbbbbb", linewidth=0.8, zorder=0)

    # One shared range, so a curve's height means the same thing in all three.
    axes[0].set_ylim(0.3, ceiling * 8)
    axes[0].legend(loc="upper left", frameon=True, framealpha=0.95)
    axes[1].set_ylabel("Speedup over the problem's earliest classical algorithm")
    axes[-1].set_xlabel("Year")
    axes[-1].set_xlim(max(1950, first_move - 8), 2026)

    fig.suptitle("How much faster has each category made the median problem?",
                 fontsize=12, y=0.98)
    fig.text(0.5, 0.945,
             f"Percentiles across the {len(problems)} problems that have both a "
             f"quantum and a classical serial algorithm, at n = {_sci(size)}",
             ha="center", fontsize=8.5, color="#444444")
    fig.text(0.5, -0.055,
             "Each problem's baseline is its own earliest classical serial algorithm, so a problem with "
             "nothing published in a category sits\nat 1 and still counts toward that category's percentiles. "
             "The classical serial curve is flat here because these 18 problems are the\nones quantum was "
             "aimed at, and most carry only one or two classical entries; figure 9 measures classical progress "
             "over all 70.\nRatios between circuit depth and sequential running time are asymptotic statements, "
             "not wall-clock predictions.",
             ha="center", fontsize=7.5, color="#666666")
    fig.tight_layout(rect=(0, 0, 1, 0.935))

    path = OUT_DIR / "10_relative_speedup.png"
    fig.savefig(path)
    plt.close(fig)
    _report_speedup(series, years, problems)
    return path


def _report_speedup(series, years, problems) -> None:
    print(f"\nFigure 10 -- speedup at 2026 over each problem's earliest classical "
          f"algorithm ({len(problems)} problems)")
    for category in CATEGORY_ORDER:
        frame = series.get(category)
        if frame is None or frame.empty:
            continue
        final = frame[years[-1]]
        print(f"   {category_label(category):<20} "
              f"25th {final.quantile(0.25):>12.3g}   "
              f"median {final.median():>12.3g}   "
              f"75th {final.quantile(0.75):>12.3g}")


# ---------------------------------------------------------------------------
# Figure 11 -- where does quantum actually win?
# ---------------------------------------------------------------------------

_QUANTUM_BANDS = [key for key in BAND_KEYS if key != "no_quantum"]


def _stack(ax, x, counts: list[dict[str, int]]) -> None:
    """Stack the coverage bands, quantum's biggest lead at the bottom.

    Normalized over the problems that have a quantum algorithm at all. The
    problems that do not are a constant 86% of the catalogue and would
    otherwise flatten everything else into an unreadable strip; that share is
    reported in the subtitle instead.
    """
    totals = np.array([sum(c[k] for k in _QUANTUM_BANDS) for c in counts])
    totals[totals == 0] = 1
    bottom = np.zeros(len(x))
    for key in reversed(_QUANTUM_BANDS):
        share = np.array([c[key] for c in counts]) / totals
        ax.fill_between(x, bottom, bottom + share, color=BAND_COLORS[key],
                        label=BAND_LABELS[key], linewidth=0)
        bottom = bottom + share
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(_PERCENT)
    ax.set_xscale("log")
    ax.set_xlim(x[0], x[-1])
    ax.grid(alpha=0.15, linewidth=0.6, color="white")


def plot_quantum_coverage(ds, reference_size: float = 1e6) -> Path:
    sizes = np.logspace(1, 12, 45)
    by_size = [coverage_profile(best_by_category(ds, s)) for s in sizes]

    overheads = np.logspace(0, 15, 61)
    best_ref = best_by_category(ds, reference_size)
    by_overhead = [coverage_profile(best_ref, np.log10(o)) for o in overheads]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    _stack(axes[0], sizes, by_size)
    axes[0].set_xlabel("Problem size $n$ (each problem in its own parameter)")
    axes[0].set_ylabel("Share of the problems that have\na quantum algorithm")
    axes[0].set_title("The lead deepens as the problem grows", fontsize=10)

    _stack(axes[1], overheads, by_overhead)
    axes[1].set_xlabel("Cost of one quantum gate, in classical operations")
    axes[1].set_title("...and survives this much per-gate overhead, at $n=10^6$",
                      fontsize=10)

    reference = coverage_profile(best_ref)
    total = sum(reference.values())
    with_quantum = total - reference["no_quantum"]
    fig.suptitle("Where quantum algorithms are actually ahead", fontsize=12.5, y=1.0)
    fig.text(0.5, 0.90,
             f"Of the {total} problems with a catalogued classical serial algorithm, "
             f"{reference['no_quantum']} ({reference['no_quantum'] / total:.0%}) have no "
             f"quantum algorithm at all.\nThe panels below show the {with_quantum} that do.",
             ha="center", fontsize=8.5, color="#444444")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, -0.13))
    fig.text(0.5, -0.225,
             "Quantum circuit depth against classical sequential running time. The right panel charges every quantum "
             "algorithm a flat per-gate\ncost, which is where error correction and the gap between a gate and a CPU "
             "instruction would show up; only the problems whose quantum\nalgorithm is exponentially better "
             "(factoring, subset sum, maximum cut) keep a lead past about $10^4$.",
             ha="center", fontsize=7.5, color="#666666")
    fig.tight_layout(rect=(0, 0, 1, 0.87))

    path = OUT_DIR / "11_quantum_coverage.png"
    fig.savefig(path)
    plt.close(fig)
    _report_coverage(ds, sizes, reference_size)
    return path


def _report_coverage(ds, sizes, reference_size) -> None:
    print("\nFigure 11 -- coverage at three problem sizes")
    for size in PROBLEM_SIZES:
        counts = coverage_profile(best_by_category(ds, size))
        ahead = counts["lead_small"] + counts["lead_mid"] + counts["lead_large"]
        print(f"   n = {size:>10.0e}:  no quantum {counts['no_quantum']:>3}   "
              f"classical ahead {counts['classical']:>3}   "
              f"quantum ahead {ahead:>3}   "
              f"(of which >1e6x: {counts['lead_large']:>2})   "
              f"of {sum(counts.values())}")

    leads = overhead_tolerance(ds, reference_size)
    print(f"\n   quantum lead at n = {reference_size:.0e}, per problem "
          f"(orders of magnitude, = overhead per gate the lead can absorb):")
    for _, row in leads.iterrows():
        print(f"      {row['problem_label'][:52]:<54} {row['lead_log10']:>+8.1f}")

    crossovers = quantum_crossovers(ds, sizes)
    late = crossovers[crossovers["crossover_size"].notna()
                      & (crossovers["crossover_size"] > sizes[0])]
    never = crossovers[crossovers["crossover_size"].isna()]
    if len(late):
        print("\n   problems where quantum only pulls ahead above some size:")
        for _, row in late.iterrows():
            print(f"      {row['problem_label'][:52]:<54} n >= {row['crossover_size']:.1e}")
    if len(never):
        print("\n   problems where classical serial stays ahead at every swept size:")
        for _, row in never.iterrows():
            print(f"      {row['problem_label'][:52]:<54}")


# ---------------------------------------------------------------------------

def main() -> int:
    ds = build_dataset()
    written = [
        plot_improvement_rate_grid(ds),
        plot_relative_speedup(ds),
        plot_quantum_coverage(ds),
    ]
    print("\nwrote:")
    for path in written:
        print(f"   {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
