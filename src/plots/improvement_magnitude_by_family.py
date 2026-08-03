"""Improvement magnitude by quantum problem family.

This module compares two related measures of quantum algorithmic progress:

1. Total complexity-class reduction:
   The sum of all strict reductions achieved by algorithms in a family.

2. Average reduction per improvement event:
   The total reduction divided by the number of strict improvement events.

These measures answer different questions:

- Total reduction identifies families that accumulated the most progress.
- Average reduction identifies families whose individual improvements tended
  to make the largest jumps.

AlgoWiki complexity classes form an ordinal scale. Differences between classes
describe movement across the scale, not literal runtime speedup factors.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import textwrap

from ..data_loader import get_quantum_algorithms
from ..header import DPI, PLOTS_DIR


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROBLEM_COLUMNS = ["family", "variation"]
YEAR_COLUMN = "year"
METRIC_COLUMN = "time_class"

RESEARCH_PLOTS_DIR = PLOTS_DIR / "research"
OUTPUT_FILENAME = "improvement_magnitude_by_family"

TOTAL_COLOR = "#1f77b4"
AVERAGE_COLOR = "#ff7f0e"

MAXIMUM_FAMILIES = 12


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _prepare_yearly_best_algorithms(
    quantum_algorithms: pd.DataFrame,
) -> pd.DataFrame:
    """Retain the best classified algorithm for each problem and year.

    Several algorithms can be recorded for the same problem in the same year.
    Because their within-year order may not be reliable, the algorithm with
    the lowest time-complexity class is retained as that year's best result.

    This avoids treating arbitrary spreadsheet order as historical progress.
    """
    required_columns = {
        "family",
        "variation",
        "algorithm",
        YEAR_COLUMN,
        METRIC_COLUMN,
    }

    missing_columns = required_columns.difference(
        quantum_algorithms.columns
    )

    if missing_columns:
        missing_text = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            "Quantum algorithm data is missing required columns: "
            f"{missing_text}"
        )

    classified_algorithms = quantum_algorithms.dropna(
        subset=[
            *PROBLEM_COLUMNS,
            YEAR_COLUMN,
            METRIC_COLUMN,
        ]
    ).copy()

    classified_algorithms[YEAR_COLUMN] = pd.to_numeric(
        classified_algorithms[YEAR_COLUMN],
        errors="coerce",
    )

    classified_algorithms[METRIC_COLUMN] = pd.to_numeric(
        classified_algorithms[METRIC_COLUMN],
        errors="coerce",
    )

    classified_algorithms = classified_algorithms.dropna(
        subset=[
            YEAR_COLUMN,
            METRIC_COLUMN,
        ]
    ).copy()

    classified_algorithms[YEAR_COLUMN] = (
        classified_algorithms[YEAR_COLUMN]
        .astype(int)
    )

    classified_algorithms = classified_algorithms.sort_values(
        [
            *PROBLEM_COLUMNS,
            YEAR_COLUMN,
            METRIC_COLUMN,
            "algorithm",
        ],
        ascending=True,
        na_position="last",
    )

    yearly_best = classified_algorithms.drop_duplicates(
        subset=[
            *PROBLEM_COLUMNS,
            YEAR_COLUMN,
        ],
        keep="first",
    ).copy()

    return yearly_best.reset_index(drop=True)


def _detect_strict_improvement_events(
    yearly_best: pd.DataFrame,
) -> pd.DataFrame:
    """Return every strict record-setting improvement.

    The first classified algorithm for a problem establishes its baseline.
    A later algorithm is a strict improvement only when its time-complexity
    class is lower than the best class previously observed for that problem.

    Improvement magnitude is:

        previous best class - new best class
    """
    event_rows: list[dict] = []

    grouped_problems = yearly_best.groupby(
        PROBLEM_COLUMNS,
        sort=False,
        dropna=False,
    )

    for problem_key, problem_records in grouped_problems:
        family, variation = problem_key

        problem_records = problem_records.sort_values(
            [
                YEAR_COLUMN,
                METRIC_COLUMN,
            ]
        )

        best_class_so_far: float | None = None

        for _, algorithm_row in problem_records.iterrows():
            current_class = float(
                algorithm_row[METRIC_COLUMN]
            )

            if best_class_so_far is None:
                best_class_so_far = current_class
                continue

            if current_class < best_class_so_far:
                event_rows.append(
                    {
                        "family": family,
                        "variation": variation,
                        "algorithm": algorithm_row.get(
                            "algorithm"
                        ),
                        "year": int(
                            algorithm_row[YEAR_COLUMN]
                        ),
                        "previous_best_class": best_class_so_far,
                        "new_best_class": current_class,
                        "improvement_magnitude": (
                            best_class_so_far
                            - current_class
                        ),
                    }
                )

                best_class_so_far = current_class

    return pd.DataFrame(event_rows)


def _summarize_improvements_by_family(
    improvement_events: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate strict improvement events by problem family."""
    if improvement_events.empty:
        return pd.DataFrame(
            columns=[
                "family",
                "total_reduction",
                "event_count",
                "average_reduction",
                "largest_single_reduction",
            ]
        )

    family_summary = (
        improvement_events.groupby(
            "family",
            as_index=False,
        )
        .agg(
            total_reduction=(
                "improvement_magnitude",
                "sum",
            ),
            event_count=(
                "improvement_magnitude",
                "size",
            ),
            average_reduction=(
                "improvement_magnitude",
                "mean",
            ),
            largest_single_reduction=(
                "improvement_magnitude",
                "max",
            ),
        )
    )

    family_summary = family_summary.sort_values(
        [
            "total_reduction",
            "average_reduction",
            "family",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    return family_summary


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_reduction(
    value: float,
) -> str:
    """Format an ordinal complexity-class reduction."""
    return f"{value:.1f}"


def _event_label(
    event_count: int,
) -> str:
    """Return a grammatically correct event label."""
    event_word = (
        "event"
        if event_count == 1
        else "events"
    )

    return f"{event_count} {event_word}"


def _empty_plot() -> Path:
    """Create an informative empty figure if no improvements are present."""
    RESEARCH_PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(9, 4.5),
        dpi=DPI,
    )

    axis.text(
        0.5,
        0.5,
        "No strict quantum improvement events were found.",
        ha="center",
        va="center",
        transform=axis.transAxes,
        fontsize=11,
    )

    axis.set_axis_off()

    output_path = (
        RESEARCH_PLOTS_DIR
        / f"{OUTPUT_FILENAME}.png"
    )

    figure.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


# ---------------------------------------------------------------------------
# Public plotting function
# ---------------------------------------------------------------------------

def plot_improvement_magnitude_by_family(
    *,
    maximum_families: int = MAXIMUM_FAMILIES,
) -> Path:
    """Plot total and average improvement magnitude by family.

    The left panel shows cumulative complexity-class reduction.

    The right panel shows average complexity-class reduction per strict
    improvement event.

    Families are selected according to total cumulative reduction.
    """
    quantum_algorithms = get_quantum_algorithms()

    yearly_best = _prepare_yearly_best_algorithms(
        quantum_algorithms
    )

    improvement_events = (
        _detect_strict_improvement_events(
            yearly_best
        )
    )

    family_summary = (
        _summarize_improvements_by_family(
            improvement_events
        )
    )

    if family_summary.empty:
        return _empty_plot()

    displayed_summary = (
        family_summary.head(
            maximum_families
        )
        .copy()
    )

    # Reverse the rows so the largest family appears at the top of a
    # horizontal bar chart.
    displayed_summary = (
        displayed_summary.iloc[::-1]
        .reset_index(drop=True)
    )

    families = displayed_summary["family"]
    total_reductions = displayed_summary[
        "total_reduction"
    ]
    average_reductions = displayed_summary[
        "average_reduction"
    ]
    event_counts = displayed_summary[
        "event_count"
    ].astype(int)

    number_of_families = len(
        displayed_summary
    )

    figure_height = max(
        6.2,
        0.62 * number_of_families + 2.8,
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(
            14,
            figure_height,
        ),
        dpi=DPI,
        sharey=True,
    )

    total_axis = axes[0]
    average_axis = axes[1]

    y_positions = np.arange(
        number_of_families
    )

    # ------------------------------------------------------------------
    # Left panel: total reduction
    # ------------------------------------------------------------------
    total_bars = total_axis.barh(
        y_positions,
        total_reductions,
        color=TOTAL_COLOR,
        alpha=0.9,
    )

    total_axis.set_yticks(
        y_positions
    )

    total_axis.set_yticklabels(
        families,
        fontsize=9,
    )

    total_axis.set_xlabel(
        "Total reduction in complexity class"
    )

    total_axis.set_ylabel(
        "Problem family"
    )

    total_axis.set_title(
        "Cumulative reduction",
        fontsize=12,
        fontweight="semibold",
        pad=12,
    )

    maximum_total = float(
        total_reductions.max()
    )

    total_axis.set_xlim(
        0,
        max(
            maximum_total * 1.27,
            0.5,
        ),
    )

    total_axis.xaxis.set_major_locator(
        mtick.MultipleLocator(0.5)
    )

    total_axis.xaxis.set_major_formatter(
        mtick.FormatStrFormatter("%.1f")
    )

    for bar, total_value, event_count in zip(
        total_bars,
        total_reductions,
        event_counts,
    ):
        total_axis.text(
            bar.get_width() + maximum_total * 0.025,
            bar.get_y() + bar.get_height() / 2,
            (
                f"{_format_reduction(total_value)} "
                f"({_event_label(event_count)})"
            ),
            ha="left",
            va="center",
            fontsize=8.3,
        )

    # ------------------------------------------------------------------
    # Right panel: average reduction
    # ------------------------------------------------------------------
    average_bars = average_axis.barh(
        y_positions,
        average_reductions,
        color=AVERAGE_COLOR,
        alpha=0.9,
    )

    average_axis.set_xlabel(
        "Average reduction per improvement event"
    )

    average_axis.set_title(
        "Average size of each improvement",
        fontsize=12,
        fontweight="semibold",
        pad=12,
    )

    maximum_average = float(
        average_reductions.max()
    )

    average_axis.set_xlim(
        0,
        max(
            maximum_average * 1.22,
            0.5,
        ),
    )

    average_axis.xaxis.set_major_locator(
        mtick.MultipleLocator(0.5)
    )

    average_axis.xaxis.set_major_formatter(
        mtick.FormatStrFormatter("%.1f")
    )

    for bar, average_value in zip(
        average_bars,
        average_reductions,
    ):
        average_axis.text(
            bar.get_width() + maximum_average * 0.025,
            bar.get_y() + bar.get_height() / 2,
            _format_reduction(
                average_value
            ),
            ha="left",
            va="center",
            fontsize=8.3,
        )

    # ------------------------------------------------------------------
    # Shared styling
    # ------------------------------------------------------------------
    for axis in axes:
        axis.grid(
            True,
            axis="x",
            linestyle="--",
            alpha=0.22,
        )

        axis.set_axisbelow(True)

        axis.spines["top"].set_visible(
            False
        )

        axis.spines["right"].set_visible(
            False
        )

    figure.suptitle(
        (
            "Time-Complexity Class Reduction "
            "by Quantum Problem Family"
        ),
        fontsize=15,
        fontweight="semibold",
        y=0.975,
    )

    figure.text(
        0.5,
        0.055,
        (
            "Left: cumulative reduction across all strict improvements. "
            "Right: average reduction per strict improvement event."
        ),
        ha="center",
        va="center",
        fontsize=8.8,
    )

    figure.text(
        0.5,
        0.022,
        (
            "Reductions are measured on AlgoWiki's ordinal complexity-class "
            "scale and are not literal runtime speedup factors."
        ),
        ha="center",
        va="center",
        fontsize=7.8,
        style="italic",
        color="dimgray",
    )

    figure.subplots_adjust(
        left=0.25,
        right=0.96,
        top=0.87,
        bottom=0.23,
        wspace=0.30,
    )

    RESEARCH_PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESEARCH_PLOTS_DIR
        / f"{OUTPUT_FILENAME}.png"
    )

    figure.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"  - {output_path.relative_to(PLOTS_DIR.parent)}"
    )

    return output_path


if __name__ == "__main__":
    plot_improvement_magnitude_by_family()
