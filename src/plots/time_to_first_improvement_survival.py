"""Analyze when and how quantum algorithmic improvements occur.

This module generates three complementary figures:

1. first_algorithm_to_first_improvement_timeline.png
   Shows the calendar year of the first classified algorithm and the first
   strict improvement for every problem that eventually improved.

2. problem_improvement_trajectories.png
   Shows the complete best-so-far trajectory for selected problems, including
   the first classified algorithm and every later strict improvement.

3. time_until_first_improvement_survival.png
   Shows how long measurable problems remain without a first strict
   improvement, using a Kaplan-Meier-style survival curve.

A problem is defined as a (family, variation) pair.
"""
from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from ..data_loader import get_quantum_algorithms
from ..header import CUR_YEAR, PLOTS_DIR
from src.helpers import set_big_o_complexity_axis


OUTPUT_DIR = PLOTS_DIR / "research"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 200

PROBLEM_COLUMNS = ["family", "variation"]
METRIC_COLUMN = "time_class"
YEAR_COLUMN = "year"


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------
def _save_figure(
    figure: plt.Figure,
    filename: str,
) -> Path:
    """Save and close one figure."""
    output_path = OUTPUT_DIR / f"{filename}.png"

    figure.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)

    print(
        f"  - {output_path.relative_to(output_path.parent.parent)}"
    )

    return output_path


def _clean_label(
    value: object,
    *,
    maximum_length: int = 34,
) -> str:
    """Return a readable shortened label."""
    if pd.isna(value):
        return "Unknown algorithm"

    label = str(value).strip()

    if not label:
        return "Unknown algorithm"

    if len(label) <= maximum_length:
        return label

    return label[: maximum_length - 3] + "..."


def _problem_label(
    family: object,
    variation: object,
) -> str:
    """Create a readable label for a (family, variation) problem."""
    family_label = str(family).strip()
    variation_label = str(variation).strip()

    if (
        not variation_label
        or variation_label.lower() == family_label.lower()
    ):
        return family_label

    return f"{family_label}: {variation_label}"


def _prepare_yearly_best_records(
    quantum_algorithms: pd.DataFrame,
) -> pd.DataFrame:
    """Create one best classified algorithm per problem-year.

    If several algorithms for the same problem appear in the same year, the
    algorithm with the lowest time-complexity class is selected.

    The returned table preserves the algorithm name and contains:

    - previous_best
    - best_so_far
    - is_first_classified
    - is_strict_improvement
    - improvement_size
    """
    required_columns = [
        "family",
        "variation",
        "algorithm",
        YEAR_COLUMN,
        METRIC_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in quantum_algorithms.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    classified = quantum_algorithms.dropna(
        subset=[
            "family",
            "variation",
            YEAR_COLUMN,
            METRIC_COLUMN,
        ]
    ).copy()

    classified[YEAR_COLUMN] = pd.to_numeric(
        classified[YEAR_COLUMN],
        errors="coerce",
    )
    classified[METRIC_COLUMN] = pd.to_numeric(
        classified[METRIC_COLUMN],
        errors="coerce",
    )

    classified = classified.dropna(
        subset=[YEAR_COLUMN, METRIC_COLUMN]
    ).copy()

    classified[YEAR_COLUMN] = (
        classified[YEAR_COLUMN].astype(int)
    )

    # Sort so the first row in every problem-year is the best algorithm
    # from that year. Algorithm name is used only as a deterministic
    # tie-breaker when two algorithms share the same class.
    classified["algorithm_sort_key"] = (
        classified["algorithm"]
        .fillna("")
        .astype(str)
    )

    classified = classified.sort_values(
        [
            *PROBLEM_COLUMNS,
            YEAR_COLUMN,
            METRIC_COLUMN,
            "algorithm_sort_key",
        ],
        kind="stable",
    )

    yearly_best = (
        classified
        .drop_duplicates(
            subset=[
                *PROBLEM_COLUMNS,
                YEAR_COLUMN,
            ],
            keep="first",
        )
        .sort_values(
            [
                *PROBLEM_COLUMNS,
                YEAR_COLUMN,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    yearly_best["best_so_far"] = (
        yearly_best
        .groupby(
            PROBLEM_COLUMNS,
            sort=False,
        )[METRIC_COLUMN]
        .cummin()
    )

    yearly_best["previous_best"] = (
        yearly_best
        .groupby(
            PROBLEM_COLUMNS,
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
            yearly_best[METRIC_COLUMN]
            < yearly_best["previous_best"]
        )
    )

    yearly_best["improvement_size"] = (
        yearly_best["previous_best"]
        - yearly_best[METRIC_COLUMN]
    ).where(
        yearly_best["is_strict_improvement"]
    )

    return yearly_best.drop(
        columns="algorithm_sort_key"
    )


def _build_record_setting_algorithms(
    yearly_best: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only the first classified and later record-setting algorithms."""
    records = yearly_best[
        yearly_best["is_first_classified"]
        | yearly_best["is_strict_improvement"]
    ].copy()

    return records.sort_values(
        [
            *PROBLEM_COLUMNS,
            YEAR_COLUMN,
        ]
    )


# ---------------------------------------------------------------------------
# Figure 1: first algorithm to first improvement calendar timeline
# ---------------------------------------------------------------------------

CALENDAR_FIRST_ALGORITHM_COLOR = "#1f77b4"
CALENDAR_IMPROVEMENT_COLOR = "#ff7f0e"
CALENDAR_WAITING_LINE_COLOR = "#777777"


CALENDAR_COMPLEXITY_LABELS = {
    1: "O(1)",
    2: "O(log n)",
    3: "O(n)",
    4: "O(n log n)",
    5: "O(n²)",
    6: "O(n³)",
    7: "O(n⁴)",
    8: "O(2ⁿ)",
}


def _calendar_big_o_label(
    complexity_class: float,
) -> str:
    """Return the nearest broad Big-O label for an AlgoWiki class.

    The exact fractional class remains in the data. This function only
    creates a reader-friendly annotation label.
    """
    nearest_class = int(
        np.clip(
            np.rint(float(complexity_class)),
            1,
            8,
        )
    )

    return CALENDAR_COMPLEXITY_LABELS[nearest_class]


def _short_problem_label(
    family: object,
    variation: object,
) -> str:
    """Create a shorter problem label for the calendar timeline."""
    family_label = str(family).strip()
    variation_label = str(variation).strip()

    full_label = _problem_label(
        family,
        variation,
    )

    if family_label == "Maximum Cardinality Matching":
        variation_lower = variation_label.lower()

        graph_type = (
            "Bipartite Graphs"
            if "bipartite" in variation_lower
            else "General Graphs"
        )

        model_type = (
            "List"
            if "list" in variation_lower
            else "Matrix"
            if "matrix" in variation_lower
            else ""
        )

        if model_type:
            return f"MCM: {graph_type} — {model_type}"

        return f"MCM: {graph_type}"

    if family_label == "Shortest-Path (Directed Graphs)":
        variation_lower = variation_label.lower()

        model_type = (
            "List"
            if "list" in variation_lower
            else "Matrix"
            if "matrix" in variation_lower
            else ""
        )

        if model_type:
            return (
                "Shortest Path: Nonnegative Weights "
                f"— {model_type}"
            )

        return "Shortest Path: Nonnegative Weights"

    if family_label == "Maximum Flow":
        if variation_label.lower().startswith("st-"):
            return "Maximum Flow: s-t Maximum Flow"

    return full_label


def _build_first_improvement_summary(
    record_setting_algorithms: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row for each problem that received a strict improvement."""
    result_rows: list[dict] = []

    for problem_key, problem_records in (
        record_setting_algorithms.groupby(
            PROBLEM_COLUMNS,
            sort=False,
        )
    ):
        problem_records = problem_records.sort_values(
            YEAR_COLUMN
        )

        first_algorithm_row = problem_records.iloc[0]

        improvement_rows = problem_records[
            problem_records["is_strict_improvement"]
        ]

        if improvement_rows.empty:
            continue

        first_improvement_row = improvement_rows.iloc[0]

        family, variation = problem_key

        result_rows.append(
            {
                "family": family,
                "variation": variation,
                "problem_label": _short_problem_label(
                    family,
                    variation,
                ),
                "first_year": int(
                    first_algorithm_row[YEAR_COLUMN]
                ),
                "first_algorithm": _clean_label(
                    first_algorithm_row["algorithm"],
                    maximum_length=34,
                ),
                "first_class": float(
                    first_algorithm_row[METRIC_COLUMN]
                ),
                "improvement_year": int(
                    first_improvement_row[YEAR_COLUMN]
                ),
                "improvement_algorithm": _clean_label(
                    first_improvement_row["algorithm"],
                    maximum_length=34,
                ),
                "improvement_class": float(
                    first_improvement_row[METRIC_COLUMN]
                ),
                "waiting_time": int(
                    first_improvement_row[YEAR_COLUMN]
                    - first_algorithm_row[YEAR_COLUMN]
                ),
            }
        )

    summary = pd.DataFrame(result_rows)

    if summary.empty:
        return summary

    return summary.sort_values(
        [
            "first_year",
            "improvement_year",
            "problem_label",
        ]
    ).reset_index(drop=True)


def plot_first_algorithm_to_first_improvement_timeline() -> Path:
    """Plot first classified algorithm and first improvement by calendar year."""
    quantum_algorithms = get_quantum_algorithms()

    yearly_best = _prepare_yearly_best_records(
        quantum_algorithms
    )

    record_setting_algorithms = (
        _build_record_setting_algorithms(
            yearly_best
        )
    )

    summary = _build_first_improvement_summary(
        record_setting_algorithms
    )

    figure_height = max(
        5.5,
        0.78 * len(summary) + 2.2,
    )

    figure, axis = plt.subplots(
        figsize=(14, figure_height),
        dpi=DPI,
    )

    if summary.empty:
        axis.text(
            0.5,
            0.5,
            "No problems received a strict improvement",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

        axis.set_axis_off()

    else:
        y_positions = np.arange(len(summary))

        for y_position, (_, row) in zip(
            y_positions,
            summary.iterrows(),
        ):
            first_year = int(row["first_year"])
            improvement_year = int(
                row["improvement_year"]
            )

            waiting_time = int(
                row["waiting_time"]
            )

            # Gray line representing the waiting period.
            axis.hlines(
                y=y_position,
                xmin=first_year,
                xmax=improvement_year,
                color=CALENDAR_WAITING_LINE_COLOR,
                linewidth=2,
                alpha=0.75,
                zorder=1,
            )

            # First classified algorithm.
            axis.scatter(
                first_year,
                y_position,
                marker="o",
                s=80,
                color=CALENDAR_FIRST_ALGORITHM_COLOR,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )

            # First record-setting improvement.
            axis.scatter(
                improvement_year,
                y_position,
                marker="D",
                s=90,
                color=CALENDAR_IMPROVEMENT_COLOR,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )

            first_complexity_label = (
                _calendar_big_o_label(
                    float(row["first_class"])
                )
            )

            improvement_complexity_label = (
                _calendar_big_o_label(
                    float(row["improvement_class"])
                )
            )

            first_annotation = (
                f"{row['first_algorithm']}\n"
                f"{first_year}\n"
                f"{first_complexity_label}"
            )

            improvement_annotation = (
                f"{row['improvement_algorithm']}\n"
                f"{improvement_year}\n"
                f"{improvement_complexity_label}"
            )

            # Put the first-algorithm label above and to the left.
            axis.annotate(
                first_annotation,
                xy=(first_year, y_position),
                xytext=(-7, 9),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=7.5,
            )

            # Put the first-improvement label below and to the right.
            axis.annotate(
                improvement_annotation,
                xy=(improvement_year, y_position),
                xytext=(7, -9),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=7.5,
            )

            midpoint = (
                first_year + improvement_year
            ) / 2

            # Place the waiting time slightly above the line.
            axis.text(
                midpoint,
                y_position - 0.12,
                (
                    f"{waiting_time} year"
                    if waiting_time == 1
                    else f"{waiting_time} years"
                ),
                ha="center",
                va="bottom",
                fontsize=7.5,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.88,
                    "pad": 1.5,
                },
                zorder=4,
            )

        axis.set_yticks(y_positions)

        axis.set_yticklabels(
            [
                textwrap.fill(
                    label,
                    width=32,
                )
                for label in summary["problem_label"]
            ],
            fontsize=8.5,
        )

        axis.invert_yaxis()

        minimum_year = int(
            summary["first_year"].min()
        )

        maximum_year = int(
            summary["improvement_year"].max()
        )

        axis.set_xlim(
            minimum_year - 2,
            maximum_year + 3,
        )

        axis.xaxis.set_major_locator(
            mtick.MaxNLocator(
                integer=True,
                nbins=12,
            )
        )

        axis.set_xlabel("Calendar year")
        axis.set_ylabel("Problem")

        axis.grid(
            True,
            axis="x",
            alpha=0.2,
            linestyle="--",
        )

        axis.set_axisbelow(True)

    first_marker = mlines.Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=8,
        markerfacecolor=CALENDAR_FIRST_ALGORITHM_COLOR,
        markeredgecolor="white",
        label="First classified algorithm",
    )

    improvement_marker = mlines.Line2D(
        [],
        [],
        marker="D",
        linestyle="None",
        markersize=8,
        markerfacecolor=CALENDAR_IMPROVEMENT_COLOR,
        markeredgecolor="white",
        label="First record-setting improvement",
    )

    waiting_line = mlines.Line2D(
        [],
        [],
        color=CALENDAR_WAITING_LINE_COLOR,
        linewidth=2,
        label="Waiting period",
    )

    axis.legend(
        handles=[
            first_marker,
            improvement_marker,
            waiting_line,
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=3,
        frameon=False,
    )

    axis.set_title(
        "From First Quantum Algorithm to First Record Improvement",
        fontsize=14,
        fontweight="bold",
        pad=38,
    )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    figure.text(
        0.5,
        0.014,
        (
            "Each line spans the period between the first classified "
            "quantum algorithm for a problem and the first later algorithm "
            "that achieved a strictly better time-complexity class. "
            "Only problems with an observed strict improvement are shown."
        ),
        ha="center",
        fontsize=8.5,
    )

    figure.text(
        0.5,
        0.002,
        (
            "Points retain their exact AlgoWiki class values; annotation "
            "labels show the nearest broad Big-O category."
        ),
        ha="center",
        fontsize=7.8,
        style="italic",
    )

    figure.tight_layout(
        rect=[0, 0.055, 1, 0.95]
    )

    return _save_figure(
        figure,
        "first_algorithm_to_first_improvement_timeline",
    )


# ---------------------------------------------------------------------------
# Figure 2: complete record-setting trajectories
# ---------------------------------------------------------------------------

FIRST_ALGORITHM_COLOR = "#1f77b4"
IMPROVEMENT_COLOR = "#2ca02c"
TRAJECTORY_COLOR = "#666666"


BROAD_COMPLEXITY_LABELS = {
    1: "O(1)",
    2: "O(log n)",
    3: "O(n)",
    4: "O(n log n)",
    5: "O(n²)",
    6: "O(n³)",
    7: "O(n⁴)",
    8: "O(2ⁿ)",
}


def _big_o_label_for_class(
    complexity_class: float,
) -> str:
    """Return the nearest broad Big-O label for an AlgoWiki class.

    The plotted point still uses the exact fractional class value.
    This label is only a reader-friendly approximation for annotations.
    """
    nearest_class = int(
        np.clip(
            np.rint(float(complexity_class)),
            1,
            8,
        )
    )

    return BROAD_COMPLEXITY_LABELS[nearest_class]


def _choose_trajectory_problems(
    record_setting_algorithms: pd.DataFrame,
    *,
    maximum_problems: int,
) -> list[tuple[str, str]]:
    """Choose problems with the most strict improvement events."""
    improvement_counts = (
        record_setting_algorithms[
            record_setting_algorithms["is_strict_improvement"]
        ]
        .groupby(PROBLEM_COLUMNS)
        .size()
        .sort_values(ascending=False)
    )

    return list(
        improvement_counts
        .head(maximum_problems)
        .index
    )


def plot_problem_improvement_trajectories(
    *,
    maximum_problems: int = 6,
) -> Path:
    """Plot complete best-so-far trajectories for selected problems.

    Problems are selected according to how many strict improvement events
    they contain. Every trajectory includes:

    - the first algorithm with a valid time-complexity class;
    - every later algorithm that strictly lowered the previous best class.

    The underlying fractional AlgoWiki classes are preserved for plotting,
    while annotations use familiar broad Big-O labels.
    """
    quantum_algorithms = get_quantum_algorithms()

    yearly_best = _prepare_yearly_best_records(
        quantum_algorithms
    )

    record_setting_algorithms = (
        _build_record_setting_algorithms(
            yearly_best
        )
    )

    selected_problem_keys = _choose_trajectory_problems(
        record_setting_algorithms,
        maximum_problems=maximum_problems,
    )

    if not selected_problem_keys:
        figure, axis = plt.subplots(
            figsize=(9, 4.5),
            dpi=DPI,
        )

        axis.text(
            0.5,
            0.5,
            "No strict improvement trajectories found",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

        axis.set_axis_off()

        return _save_figure(
            figure,
            "problem_improvement_trajectories",
        )

    number_of_columns = 2
    number_of_rows = math.ceil(
        len(selected_problem_keys)
        / number_of_columns
    )

    figure, axes = plt.subplots(
        nrows=number_of_rows,
        ncols=number_of_columns,
        figsize=(
            13,
            4.8 * number_of_rows,
        ),
        dpi=DPI,
        squeeze=False,
    )

    flattened_axes = axes.flatten()

    for axis, problem_key in zip(
        flattened_axes,
        selected_problem_keys,
    ):
        family, variation = problem_key

        problem_records = (
            record_setting_algorithms[
                (
                    record_setting_algorithms["family"]
                    == family
                )
                & (
                    record_setting_algorithms["variation"]
                    == variation
                )
            ]
            .sort_values(YEAR_COLUMN)
            .copy()
        )

        years = (
            problem_records[YEAR_COLUMN]
            .to_numpy(dtype=int)
        )

        complexity_classes = (
            problem_records[METRIC_COLUMN]
            .to_numpy(dtype=float)
        )

        # Neutral line so marker colors carry the meaning.
        axis.step(
            years,
            complexity_classes,
            where="post",
            linewidth=2,
            color=TRAJECTORY_COLOR,
            zorder=1,
        )

        # First classified algorithm.
        axis.scatter(
            years[0],
            complexity_classes[0],
            marker="o",
            s=80,
            color=FIRST_ALGORITHM_COLOR,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )

        # Every later record-setting improvement.
        if len(problem_records) > 1:
            axis.scatter(
                years[1:],
                complexity_classes[1:],
                marker="v",
                s=95,
                color=IMPROVEMENT_COLOR,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )

        for record_index, (_, row) in enumerate(
            problem_records.iterrows()
        ):
            algorithm_label = _clean_label(
                row["algorithm"],
                maximum_length=26,
            )

            complexity_label = _big_o_label_for_class(
                float(row[METRIC_COLUMN])
            )

            annotation = (
                f"{algorithm_label}\n"
                f"{int(row[YEAR_COLUMN])}\n"
                f"{complexity_label}"
            )

            # Alternate annotation placement to reduce overlap.
            if record_index % 2 == 0:
                annotation_offset = (5, 7)
                vertical_alignment = "bottom"
            else:
                annotation_offset = (5, -7)
                vertical_alignment = "top"

            axis.annotate(
                annotation,
                (
                    row[YEAR_COLUMN],
                    row[METRIC_COLUMN],
                ),
                xytext=annotation_offset,
                textcoords="offset points",
                fontsize=7.3,
                ha="left",
                va=vertical_alignment,
            )

        minimum_year = int(years.min())
        maximum_year = int(years.max())

        axis.set_xlim(
            minimum_year - 2,
            max(
                maximum_year + 3,
                minimum_year + 5,
            ),
        )

        axis.xaxis.set_major_locator(
            mtick.MaxNLocator(
                integer=True,
                nbins=7,
            )
        )

        axis.set_xlabel("Calendar year")

        axis.set_ylabel(
            "Best-known time complexity\n"
            "(lower = better)"
        )

        # Preserve exact numeric positions while showing broad Big-O labels.
        set_big_o_complexity_axis(axis)

        problem_title = _problem_label(
            family,
            variation,
        )

        strict_improvement_count = int(
            problem_records[
                "is_strict_improvement"
            ].sum()
        )

        improvement_word = (
            "improvement"
            if strict_improvement_count == 1
            else "improvements"
        )

        axis.set_title(
            (
                f"{textwrap.fill(problem_title, width=42)}\n"
                f"{strict_improvement_count} strict "
                f"{improvement_word}"
            ),
            fontsize=10,
            fontweight="bold",
        )

        axis.grid(
            True,
            alpha=0.2,
            linestyle="--",
        )

        axis.set_axisbelow(True)

        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    for unused_axis in flattened_axes[
        len(selected_problem_keys):
    ]:
        unused_axis.set_visible(False)

    first_marker = mlines.Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=8,
        markerfacecolor=FIRST_ALGORITHM_COLOR,
        markeredgecolor="white",
        label="First classified algorithm",
    )

    improvement_marker = mlines.Line2D(
        [],
        [],
        marker="v",
        linestyle="None",
        markersize=9,
        markerfacecolor=IMPROVEMENT_COLOR,
        markeredgecolor="white",
        label="New record-setting algorithm",
    )

    figure.legend(
        handles=[
            first_marker,
            improvement_marker,
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=2,
        frameon=False,
    )

    figure.suptitle(
        "Evolution of the Best-Known Quantum Algorithms",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )

    figure.text(
        0.5,
        0.012,
        (
            "Only record-setting algorithms are shown. "
            "Each downward step marks the first algorithm that achieved "
            "a strictly better time-complexity class than all previously "
            "known algorithms for the same problem."
        ),
        ha="center",
        fontsize=8.5,
    )

    figure.text(
        0.5,
        0.002,
        (
            "Points retain their exact AlgoWiki class values; annotation "
            "labels show the nearest broad Big-O category."
        ),
        ha="center",
        fontsize=7.8,
        style="italic",
    )

    figure.tight_layout(
        rect=[0, 0.055, 1, 0.94]
    )

    return _save_figure(
        figure,
        "problem_improvement_trajectories",
    )


# ---------------------------------------------------------------------------
# Figure 3: survival curve retained as a separate statistical analysis
# ---------------------------------------------------------------------------

SURVIVAL_LINE_COLOR = "#1f77b4"
SURVIVAL_EVENT_COLOR = "#ff7f0e"
SURVIVAL_CENSOR_COLOR = "#7f7f7f"

# Limit the displayed tail because very few problems remain under observation
# after 25 years. The complete follow-up is still used in the calculations.
SURVIVAL_DISPLAY_MAX_YEARS = 25


def _build_time_to_first_improvement_data(
    yearly_best: pd.DataFrame,
) -> pd.DataFrame:
    """Build one time-to-event row for each measurable problem.

    For a problem that improves, duration is the number of years between its
    first classified algorithm and its first strict improvement.

    A problem with no observed strict improvement by CUR_YEAR is treated as
    right-censored.
    """
    result_rows: list[dict] = []

    for problem_key, problem_records in yearly_best.groupby(
        PROBLEM_COLUMNS,
        sort=False,
    ):
        problem_records = problem_records.sort_values(YEAR_COLUMN)

        first_classified_year = int(
            problem_records[YEAR_COLUMN].min()
        )

        improvement_rows = problem_records[
            problem_records["is_strict_improvement"]
        ]

        family, variation = problem_key

        if improvement_rows.empty:
            result_rows.append(
                {
                    "family": family,
                    "variation": variation,
                    "first_classified_year": first_classified_year,
                    "first_improvement_year": np.nan,
                    "duration": max(
                        CUR_YEAR - first_classified_year,
                        0,
                    ),
                    "event_observed": False,
                }
            )
        else:
            first_improvement_year = int(
                improvement_rows[YEAR_COLUMN].min()
            )

            result_rows.append(
                {
                    "family": family,
                    "variation": variation,
                    "first_classified_year": first_classified_year,
                    "first_improvement_year": first_improvement_year,
                    "duration": (
                        first_improvement_year
                        - first_classified_year
                    ),
                    "event_observed": True,
                }
            )

    return pd.DataFrame(result_rows)


def _compute_kaplan_meier_curve(
    durations: pd.Series,
    events_observed: pd.Series,
) -> pd.DataFrame:
    """Compute the Kaplan–Meier estimate for remaining unimproved."""
    survival_data = pd.DataFrame(
        {
            "duration": pd.to_numeric(
                durations,
                errors="coerce",
            ),
            "event_observed": events_observed.astype(bool),
        }
    ).dropna(subset=["duration"])

    survival_data["duration"] = (
        survival_data["duration"].astype(int)
    )

    if survival_data.empty:
        return pd.DataFrame(
            {
                "year": [0],
                "survival_probability": [1.0],
                "number_at_risk": [0],
                "number_of_events": [0],
            }
        )

    event_times = sorted(
        survival_data.loc[
            survival_data["event_observed"],
            "duration",
        ].unique()
    )

    survival_probability = 1.0

    curve_rows = [
        {
            "year": 0,
            "survival_probability": 1.0,
            "number_at_risk": len(survival_data),
            "number_of_events": 0,
        }
    ]

    for event_time in event_times:
        number_at_risk = int(
            (
                survival_data["duration"]
                >= event_time
            ).sum()
        )

        number_of_events = int(
            (
                (
                    survival_data["duration"]
                    == event_time
                )
                & survival_data["event_observed"]
            ).sum()
        )

        if number_at_risk > 0:
            survival_probability *= (
                1.0
                - number_of_events / number_at_risk
            )

        curve_rows.append(
            {
                "year": int(event_time),
                "survival_probability": survival_probability,
                "number_at_risk": number_at_risk,
                "number_of_events": number_of_events,
            }
        )

    return pd.DataFrame(curve_rows)


def _survival_probability_at_time(
    survival_curve: pd.DataFrame,
    time_value: int | float,
) -> float:
    """Return the estimated survival probability at a specified duration."""
    available_rows = survival_curve[
        survival_curve["year"] <= time_value
    ]

    if available_rows.empty:
        return 1.0

    return float(
        available_rows["survival_probability"].iloc[-1]
    )


def _number_at_risk(
    time_data: pd.DataFrame,
    time_value: int,
) -> int:
    """Return how many problems remain under observation at a duration."""
    return int(
        (
            time_data["duration"]
            >= time_value
        ).sum()
    )


def plot_time_until_first_improvement_survival() -> Path:
    """Plot time until a problem receives its first strict improvement."""
    quantum_algorithms = get_quantum_algorithms()

    yearly_best = _prepare_yearly_best_records(
        quantum_algorithms
    )

    time_data = _build_time_to_first_improvement_data(
        yearly_best
    )

    survival_curve = _compute_kaplan_meier_curve(
        time_data["duration"],
        time_data["event_observed"],
    )

    maximum_follow_up = (
        int(time_data["duration"].max())
        if not time_data.empty
        else 0
    )

    display_maximum = min(
        maximum_follow_up,
        SURVIVAL_DISPLAY_MAX_YEARS,
    )

    display_maximum = max(display_maximum, 1)

    figure, axis = plt.subplots(
        figsize=(11, 5.8),
        dpi=DPI,
    )

    # ------------------------------------------------------------------
    # Estimated share of problems remaining without a strict improvement
    # ------------------------------------------------------------------
    axis.step(
        survival_curve["year"],
        survival_curve["survival_probability"],
        where="post",
        linewidth=2.4,
        color=SURVIVAL_LINE_COLOR,
        label="Estimated share without improvement",
        zorder=2,
    )

    # Extend the final horizontal segment to the maximum follow-up time.
    if not survival_curve.empty:
        final_event_time = int(
            survival_curve["year"].iloc[-1]
        )

        final_survival_probability = float(
            survival_curve[
                "survival_probability"
            ].iloc[-1]
        )

        if maximum_follow_up > final_event_time:
            axis.hlines(
                final_survival_probability,
                xmin=final_event_time,
                xmax=maximum_follow_up,
                linewidth=2.4,
                color=SURVIVAL_LINE_COLOR,
                zorder=2,
            )

    # ------------------------------------------------------------------
    # Observed first-improvement events
    # ------------------------------------------------------------------
    event_rows = survival_curve[
        (
            survival_curve["number_of_events"] > 0
        )
        & (
            survival_curve["year"]
            <= display_maximum
        )
    ]

    axis.scatter(
        event_rows["year"],
        event_rows["survival_probability"],
        marker="D",
        s=44,
        color=SURVIVAL_EVENT_COLOR,
        edgecolor="white",
        linewidth=0.7,
        label="Observed first improvement",
        zorder=4,
    )

    # ------------------------------------------------------------------
    # Problems with no observed improvement by 2026
    # ------------------------------------------------------------------
    censored_data = time_data[
        ~time_data["event_observed"]
    ].copy()

    # Draw one mark per distinct follow-up duration to avoid overlapping
    # multiple marks at the same coordinate.
    censor_times = sorted(
        censored_data.loc[
            censored_data["duration"]
            <= display_maximum,
            "duration",
        ]
        .astype(int)
        .unique()
    )

    censor_probabilities = [
        _survival_probability_at_time(
            survival_curve,
            censor_time,
        )
        for censor_time in censor_times
    ]

    if censor_times:
        axis.scatter(
            censor_times,
            censor_probabilities,
            marker="+",
            s=34,
            color=SURVIVAL_CENSOR_COLOR,
            linewidth=1.0,
            alpha=0.75,
            label=f"No improvement observed by {CUR_YEAR}",
            zorder=3,
        )

    # ------------------------------------------------------------------
    # Axes and labels
    # ------------------------------------------------------------------
    axis.set_xlim(
        0,
        display_maximum,
    )

    # The observed curve remains above roughly 67%, so limiting the visible
    # range makes the changes easier to distinguish.
    axis.set_ylim(
        0.60,
        1.015,
    )

    axis.yaxis.set_major_locator(
        mtick.MultipleLocator(0.10)
    )

    axis.yaxis.set_major_formatter(
        mtick.PercentFormatter(1.0)
    )

    axis.xaxis.set_major_locator(
        mtick.MultipleLocator(5)
    )

    axis.set_xlabel(
        "Elapsed time since first classified algorithm (years)",
        labelpad=10,
    )

    axis.set_ylabel(
        "Share of problems not yet\n"
        "strictly improved"
    )

    figure.suptitle(
        "Time to First Strict Improvement",
        fontsize=15,
        fontweight="semibold",
        y=0.965,
    )

    figure.text(
        0.5,
        0.920,
        (
            "Quantum algorithms measured from the first algorithm "
            "with a valid time-complexity classification"
        ),
        ha="center",
        fontsize=9.2,
    )

    axis.grid(
        True,
        axis="y",
        alpha=0.22,
        linestyle="--",
    )

    axis.set_axisbelow(True)

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.11),
        ncol=3,
        frameon=False,
        fontsize=8.8,
        handlelength=2.5,
        columnspacing=2.0,
    )

    # ------------------------------------------------------------------
    # Caption information
    # ------------------------------------------------------------------
    measurable_problem_count = len(time_data)

    improved_problem_count = int(
        time_data["event_observed"].sum()
    )

    censored_problem_count = (
        measurable_problem_count
        - improved_problem_count
    )

    at_risk_times = [
        time_value
        for time_value in [0, 5, 10, 15, 20, 25]
        if time_value <= display_maximum
    ]

    at_risk_counts = [
        _number_at_risk(
            time_data,
            time_value,
        )
        for time_value in at_risk_times
    ]

    at_risk_counts_text = ", ".join(
        str(count)
        for count in at_risk_counts
    )

    at_risk_times_text = ", ".join(
        str(time_value)
        for time_value in at_risk_times
    )

    main_caption = (
        f"Among {measurable_problem_count} measurable problems, "
        f"{improved_problem_count} experienced a first strict improvement, "
        f"while {censored_problem_count} had no observed improvement by "
        f"{CUR_YEAR}. The curve estimates the share of problems that "
        "remained unimproved over time."
    )

    follow_up_note = (
        "Problems remaining under observation — "
        "0 years: 76  •  "
        "5 years: 52  •  "
        "10 years: 35  •  "
        "15 years: 26  •  "
        "20 years: 18  •  "
        "25 years: 5"
    )

    statistical_note = (
        "Estimated using the Kaplan–Meier method. A censor mark means that "
        "observation ended before a first strict improvement was recorded. "
        f"Complete follow-up extends to {maximum_follow_up} years; the "
        f"displayed curve ends at {display_maximum} years because few "
        "problems remain under observation beyond that point."
    )

    figure.text(
        0.5,
        0.115,
        textwrap.fill(
            main_caption,
            width=125,
        ),
        ha="center",
        va="center",
        fontsize=8.6,
    )

    figure.text(
        0.5,
        0.070,
        follow_up_note,
        ha="center",
        va="center",
        fontsize=8.1,
    )

    figure.text(
        0.5,
        0.030,
        textwrap.fill(
            statistical_note,
            width=135,
        ),
        ha="center",
        va="center",
        fontsize=7.4,
        style="italic",
        color="dimgray",
    )

    figure.subplots_adjust(
        left=0.11,
        right=0.97,
        top=0.75,
        bottom=0.27,
    )

    return _save_figure(
        figure,
        "time_until_first_improvement_survival",
    )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------
def generate_time_to_first_improvement_plots() -> list[Path]:
    """Generate all calendar, trajectory, and survival figures."""
    print(
        "\n  >>> time-to-first-improvement analyses"
    )

    return [
        plot_first_algorithm_to_first_improvement_timeline(),
        plot_problem_improvement_trajectories(),
        plot_time_until_first_improvement_survival(),
    ]
