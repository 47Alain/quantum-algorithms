"""Shared helpers for plots: improvement detection + matplotlib styling.

Plot styling tries to match the look of dtontici/parallel-algorithms, which
uses a clean white background, a compact 6.55 x 3.5 in figure at 200 dpi,
horizontal multi-line y-labels (e.g. ``% Problem\\nFamilies with\\nImprovements``
positioned to the left with ``rotation=0, labelpad=40``), and inline coloured
text instead of a legend when there are multiple series.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FixedLocator, FixedFormatter

def set_big_o_complexity_axis(ax):
    """
    Display AlgoWiki's ordinal complexity scale using familiar
    Big-O notation while preserving the underlying numeric values.
    """

    tick_positions = [1, 2, 3, 4, 5, 6, 7, 8]

    tick_labels = [
        "O(1)",
        "O(log n)",
        "O(n)",
        "O(n log n)",
        "O(n²)",
        "O(n³)",
        "O(n⁴)",
        "O(2ⁿ)"
    ]

    ax.yaxis.set_major_locator(FixedLocator(tick_positions))
    ax.yaxis.set_major_formatter(FixedFormatter(tick_labels))

    ax.set_ylim(0.8, 8.2)

from .header import (
    CUR_YEAR,
    DECADES,
    DPI,
    EDGE_COLOR,
    FIGSIZE,
    GRID_ALPHA,
    MIN_YEAR,
    SAVE_LOC,
    STYLES,
)


def get_style(name: str = "parallel") -> dict:
    """Return the named style config from `STYLES` in header.py."""
    if name not in STYLES:
        raise ValueError(
            f"Unknown style {name!r}. Choose from {sorted(STYLES.keys())}.")
    return STYLES[name]


# ---------------------------------------------------------------------------
# Improvement detection
# ---------------------------------------------------------------------------
def mark_improvements(
    df: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = ("family", "variation"),
    metric_col: str = "time_class",
    year_col: str = "year",
) -> pd.DataFrame:
    """Annotate each algorithm row with whether it represents an improvement.

    Inside each group (default: a problem = (family, variation)), rows are
    sorted by year. The first algorithm in a group is always considered an
    improvement (introduces the problem). After that, an algorithm is an
    improvement iff its ``metric_col`` is strictly less than the best value
    seen so far in that group. ``time_class`` is used by default because the
    AlgoWiki convention encodes "lower class = faster algorithm".

    Adds two columns:
      * ``is_first``       - True for the earliest algorithm in the group
      * ``is_improvement`` - True if it strictly improved the best metric

    Rows missing the metric are kept but cannot trigger an improvement, so
    they will only be marked ``is_first`` when they are the only entry seen
    so far for the group.
    """
    work = df.copy()
    work = work.sort_values(list(group_cols) + [year_col], kind="stable")

    is_first = []
    is_improvement = []
    best_metric: dict[tuple, float] = {}
    seen: set[tuple] = set()

    for _, row in work.iterrows():
        key = tuple(row[c] for c in group_cols)
        metric = row.get(metric_col)
        first = key not in seen
        seen.add(key)

        if first:
            is_first.append(True)
            improved = True
            if pd.notna(metric):
                best_metric[key] = float(metric)
        else:
            is_first.append(False)
            if pd.isna(metric):
                improved = False
            else:
                m = float(metric)
                cur_best = best_metric.get(key)
                if cur_best is None or m < cur_best:
                    improved = True
                    best_metric[key] = m
                else:
                    improved = False
        is_improvement.append(improved)

    work["is_first"] = is_first
    work["is_improvement"] = is_improvement
    return work

def build_yearly_improvement_records(
    df: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = (
        "family",
        "variation",
    ),
    metric_col: str = "time_class",
    year_col: str = "year",
) -> pd.DataFrame:
    """Create one best complexity record per problem per year.

    Multiple algorithms for the same problem may appear in the same year.
    Because the dataset does not contain publication dates within a year,
    this function uses the best metric value from that year and counts at
    most one strict improvement per problem-year.

    A strict improvement occurs when the current year's best metric is lower
    than the best metric observed in all earlier years.

    Returns columns including:

    - previous_best
    - is_first_classified
    - is_strict_improvement
    - improvement_size
    """
    required_columns = [
        *group_cols,
        year_col,
        metric_col,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    classified_algorithms = df.dropna(
        subset=required_columns
    ).copy()

    classified_algorithms[year_col] = (
        pd.to_numeric(
            classified_algorithms[year_col],
            errors="coerce",
        )
    )

    classified_algorithms[metric_col] = (
        pd.to_numeric(
            classified_algorithms[metric_col],
            errors="coerce",
        )
    )

    classified_algorithms = (
        classified_algorithms.dropna(
            subset=[year_col, metric_col]
        )
    )

    classified_algorithms[year_col] = (
        classified_algorithms[year_col]
        .astype(int)
    )

    # Select the best recorded complexity for each problem-year.
    yearly_best = (
        classified_algorithms
        .groupby(
            [
                *group_cols,
                year_col,
            ],
            as_index=False,
        )[metric_col]
        .min()
        .sort_values(
            [
                *group_cols,
                year_col,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    # Best complexity observed up to and including the current year.
    yearly_best["best_so_far"] = (
        yearly_best
        .groupby(
            list(group_cols),
            sort=False,
        )[metric_col]
        .cummin()
    )

    # Best complexity observed strictly before the current year.
    yearly_best["previous_best"] = (
        yearly_best
        .groupby(
            list(group_cols),
            sort=False,
        )["best_so_far"]
        .shift(1)
    )

    yearly_best["is_first_classified"] = (
        yearly_best["previous_best"].isna()
    )

    yearly_best["is_strict_improvement"] = (
        yearly_best["previous_best"].notna()
        & (
            yearly_best[metric_col]
            < yearly_best["previous_best"]
        )
    )

    yearly_best["improvement_size"] = (
        yearly_best["previous_best"]
        - yearly_best[metric_col]
    ).where(
        yearly_best["is_strict_improvement"]
    )

    return yearly_best

def first_year_per_problem(
    df: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = ("family", "variation"),
    year_col: str = "year",
) -> pd.DataFrame:
    """One row per problem with the earliest year an algorithm appeared."""
    return (
        df.groupby(list(group_cols), as_index=False)[year_col]
        .min()
        .rename(columns={year_col: "first_year"})
    )


def restrict_to_known_problems(
    quantum: pd.DataFrame,
    problems: pd.DataFrame,
    *,
    on: tuple[str, ...] = ("family", "variation"),
) -> pd.DataFrame:
    """Inner-join to keep only quantum algos whose (family, variation) appears
    in the Problems sheet. Useful when you specifically want to restrict to
    canonical AlgoWiki problems.
    """
    pkey = problems[list(on)].drop_duplicates()
    return quantum.merge(pkey, on=list(on), how="inner")


def report_problems_coverage(quantum: pd.DataFrame, problems: pd.DataFrame) -> None:
    """Print a one-shot summary of how the Quantum sheet aligns with Problems.
    Helps surface mismatches (e.g. 'APSP (Adjacency Matrix Model)' is not in
    the Problems sheet, so plots restricted to the sheet would lose those).
    """
    qkeys = set(zip(quantum["family"], quantum["variation"]))
    pkeys = set(zip(problems["family"], problems["variation"]))
    qfam = set(quantum["family"])
    pfam = set(problems["family"])
    print(
        f"  problems-sheet coverage: "
        f"{len(qkeys & pkeys)}/{len(qkeys)} (family, variation) pairs match  |  "
        f"{len(qfam & pfam)}/{len(qfam)} families match"
    )


# ---------------------------------------------------------------------------
# Plot styling + IO
# ---------------------------------------------------------------------------
def make_figure(figsize=FIGSIZE, dpi: int = DPI):
    """Return (fig, ax) using the supplied size + DPI (style defaults)."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    return fig, ax


def style_year_axis(ax, *, min_year: int = MIN_YEAR, max_year: int = CUR_YEAR,
                    integer_y: bool = True, grid_alpha: float = GRID_ALPHA) -> None:
    """Common formatting for year-on-x bar charts."""
    ax.set_xlim(min_year - 0.5, max_year + 0.5)
    if grid_alpha > 0:
        ax.grid(True, axis="y", alpha=grid_alpha)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if integer_y:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def style_decade_axis(ax, *, decades: list[dict] | None = None,
                      integer_y: bool = True,
                      grid_alpha: float = GRID_ALPHA) -> None:
    """Common formatting for decade-on-x bar charts.
    Mirrors the x-axis treatment in parallel-algorithms `decade_progress.py`:
    integer positions on the x-axis with the decade label as the tick text.
    """
    decades = decades or DECADES
    ax.set_xticks(range(len(decades)))
    ax.set_xticklabels([d["label"] for d in decades])
    ax.set_xlim(-0.5, len(decades) - 0.5)
    if grid_alpha > 0:
        ax.grid(True, axis="y", alpha=grid_alpha)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if integer_y:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def save_plot(fig, name: str, *, save_dir: Path | None = None) -> Path:
    """Save ``fig`` as ``name.png`` under the style's ``Plots/`` subfolder and close it."""
    save_dir = save_dir or SAVE_LOC
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"{name}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------
def yearly_counts(
    series: pd.Series,
    *,
    min_year: int = MIN_YEAR,
    max_year: int = CUR_YEAR,
) -> pd.Series:
    """Count occurrences per year, reindexed over the full window (filling 0)."""
    counts = series.dropna().astype(int).value_counts().sort_index()
    full = pd.Series(0, index=range(min_year, max_year + 1), dtype=int)
    full.update(counts)
    return full


def decade_counts(
    series: pd.Series,
    *,
    decades: list[dict] | None = None,
) -> pd.Series:
    """Count occurrences per decade bucket. Mirrors the decade aggregation in
    parallel-algorithms `decade_progress.py` / `improved_families.py`.

    Each value y is placed in the first decade whose ``max`` is >= y. The
    returned series is indexed by integer positions 0..len(decades)-1 and
    has the decade label as the ``Series.name``-friendly metadata.
    """
    decades = decades or DECADES
    out = [0] * len(decades)
    for y in series.dropna().astype(int).tolist():
        for i, d in enumerate(decades):
            if y <= d["max"]:
                out[i] += 1
                break
    return pd.Series(out, index=range(len(decades)))


def decade_unique(
    df: pd.DataFrame,
    *,
    year_col: str = "year",
    key_cols: tuple[str, ...] = ("family",),
    decades: list[dict] | None = None,
) -> pd.Series:
    """Count *distinct* keys (e.g. families) per decade bucket. Used for the
    "share of families improved each decade" plot.
    """
    decades = decades or DECADES
    out = [set() for _ in decades]
    for _, row in df[[year_col, *key_cols]].dropna().iterrows():
        y = int(row[year_col])
        key = tuple(row[c] for c in key_cols)
        for i, d in enumerate(decades):
            if y <= d["max"]:
                out[i].add(key)
                break
    return pd.Series([len(s) for s in out], index=range(len(decades)))


# ---------------------------------------------------------------------------
# Bar drawing helpers
# ---------------------------------------------------------------------------
def bar_simple(ax, counts: pd.Series, *, color: str, label: str | None = None,
               width: float = 0.8) -> None:
    """Single-series bar chart, parallel-algorithms style: clean fill, no
    legend by default (callers add inline `text_label` instead when desired).
    """
    ax.bar(counts.index, counts.values, color=color, edgecolor=EDGE_COLOR,
           linewidth=0.5, label=label, width=width)


def bar_with_cumulative(ax, counts: pd.Series, *, color: str, label: str,
                        cumulative_label: str | None = None) -> None:
    """Bar chart with a cumulative line on a secondary y-axis.
    Kept for backwards compatibility; not used by the parallel-algorithms-style
    defaults. Pass ``show_cumulative=True`` to a plot function to opt in.
    """
    ax.bar(counts.index, counts.values, color=color, edgecolor=EDGE_COLOR,
           linewidth=0.5, label=label, width=0.8)
    ax2 = ax.twinx()
    ax2.plot(counts.index, counts.cumsum().values,
             color="black", linewidth=1.5, alpha=0.6,
             label=cumulative_label or f"Cumulative {label.lower()}")
    ax2.set_ylabel("Cumulative")
    ax2.spines["top"].set_visible(False)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)


# ---------------------------------------------------------------------------
# Label / axis cosmetics (matches parallel-algorithms' multi-line ylabel idiom)
# ---------------------------------------------------------------------------
def multiline_ylabel(ax, lines: list[str] | str, *, labelpad: float = 40.0) -> None:
    """Set a horizontal, left-aligned, multi-line y-label.

    Mirrors the parallel-algorithms idiom:
        ax.set_ylabel("% Problem\\nFamilies with\\nImprovements",
                      rotation=0, labelpad=40.0)
    """
    text = "\n".join(lines) if isinstance(lines, list) else lines
    ax.set_ylabel(text, rotation=0, labelpad=labelpad,
                  horizontalalignment="right", verticalalignment="center")


def percent_yaxis(ax) -> None:
    """Format the y-axis as ``NN%`` (parallel-algorithms uses
    ``ax.yaxis.set_major_formatter(mtick.PercentFormatter())``)."""
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0))


def inline_text_labels(ax, items: list[tuple[float, float, str, str]],
                       *, fontsize: int = 11) -> None:
    """Place inline coloured text labels (parallel-algorithms idiom for series
    annotation in lieu of a matplotlib legend).

    ``items`` is a list of ``(x, y, text, color)`` tuples.
    """
    for x, y, text, color in items:
        ax.text(x, y, text, color=color, fontsize=fontsize,
                horizontalalignment="center")
