"""Family-level quantum algorithm improvements by decade.

This module generates two complementary heatmaps:

1. ``family_improvements_by_decade.png``
   Counts strict improvement events by problem family and conventional decade.

2. ``family_improvement_magnitude_by_decade.png``
   Sums the reduction in AlgoWiki time-complexity class produced by those
   strict improvement events.

A strict improvement occurs when an algorithm achieves a lower time-complexity
class than every previously recorded algorithm for the same
(family, variation) problem.

Lower AlgoWiki complexity-class values represent better asymptotic complexity.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data_loader import get_quantum_algorithms
from ..header import DPI, PLOTS_DIR


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROBLEM_COLUMNS = ["family", "variation"]
YEAR_COLUMN = "year"
METRIC_COLUMN = "time_class"

RESEARCH_PLOTS_DIR = PLOTS_DIR / "research"

COUNT_FILENAME = "family_improvements_by_decade"
MAGNITUDE_FILENAME = "family_improvement_magnitude_by_decade"

COUNT_COLOR_MAP = "viridis"
MAGNITUDE_COLOR_MAP = "plasma"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _prepare_yearly_best_algorithms(
    quantum_algorithms: pd.DataFrame,
) -> pd.DataFrame:
    """Return the best classified algorithm for each problem and year.

    Multiple algorithms can appear for the same problem in the same year.
    Because the spreadsheet does not necessarily provide a reliable
    within-year ordering, only the lowest time-complexity class from each
    year is retained.

    This prevents arbitrary spreadsheet row ordering from creating false
    within-year improvement events.
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

    # Sorting by metric first ensures that the first row kept for a given
    # problem-year is the algorithm with the lowest complexity class.
    classified_algorithms = classified_algorithms.sort_values(
        [
            *PROBLEM_COLUMNS,
            YEAR_COLUMN,
            METRIC_COLUMN,
            "algorithm",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            True,
        ],
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
    """Detect record-setting algorithms and calculate improvement magnitude.

    For each problem, the first classified algorithm establishes the baseline.
    A later algorithm is a strict improvement only when its complexity class
    is lower than the best class previously observed for that problem.

    Improvement magnitude is:

        previous best class - new best class

    This is an ordinal class reduction, not a literal runtime speedup factor.
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

            # The first valid algorithm establishes the baseline and is not
            # counted as an improvement event.
            if best_class_so_far is None:
                best_class_so_far = current_class
                continue

            if current_class < best_class_so_far:
                improvement_magnitude = (
                    best_class_so_far
                    - current_class
                )

                year = int(
                    algorithm_row[YEAR_COLUMN]
                )

                decade_start = (
                    year // 10
                ) * 10

                event_rows.append(
                    {
                        "family": family,
                        "variation": variation,
                        "algorithm": algorithm_row.get(
                            "algorithm"
                        ),
                        "year": year,
                        "decade_start": decade_start,
                        "decade": f"{decade_start}s",
                        "previous_best_class": (
                            best_class_so_far
                        ),
                        "new_best_class": current_class,
                        "improvement_magnitude": (
                            improvement_magnitude
                        ),
                    }
                )

                best_class_so_far = current_class

    return pd.DataFrame(event_rows)


# ---------------------------------------------------------------------------
# Heatmap table construction
# ---------------------------------------------------------------------------

def _build_family_decade_table(
    improvement_events: pd.DataFrame,
    *,
    value_mode: str,
) -> pd.DataFrame:
    """Create a family-by-decade table with a final Total column.

    Parameters
    ----------
    improvement_events:
        Strict improvement-event records.

    value_mode:
        ``"count"`` counts strict improvement events.

        ``"magnitude"`` sums class reductions.
    """
    if improvement_events.empty:
        return pd.DataFrame()

    if value_mode == "count":
        table = pd.pivot_table(
            improvement_events,
            index="family",
            columns="decade_start",
            values="algorithm",
            aggfunc="size",
            fill_value=0,
        )

    elif value_mode == "magnitude":
        table = pd.pivot_table(
            improvement_events,
            index="family",
            columns="decade_start",
            values="improvement_magnitude",
            aggfunc="sum",
            fill_value=0.0,
        )

    else:
        raise ValueError(
            "value_mode must be either 'count' or 'magnitude'"
        )

    # Ensure decades appear in chronological order.
    table = table.reindex(
        sorted(table.columns),
        axis=1,
    )

    # Remove decades that contain no improvement activity.
    nonempty_columns = (
        table.sum(axis=0) > 0
    )

    table = table.loc[
        :,
        nonempty_columns,
    ]

    # Add a total before sorting.
    table["Total"] = table.sum(axis=1)

    # Sort families by total progress, then alphabetically for stable output.
    table = (
        table.reset_index()
        .sort_values(
            ["Total", "family"],
            ascending=[False, True],
        )
        .set_index("family")
    )

    # Rename decade columns only after sorting.
    renamed_columns: list[str] = []

    for column in table.columns:
        if column == "Total":
            renamed_columns.append("Total")
        else:
            renamed_columns.append(
                f"{int(column)}s"
            )

    table.columns = renamed_columns

    return table


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _annotation_text(
    value: float,
    *,
    value_mode: str,
) -> str:
    """Format a heatmap cell annotation."""
    if np.isclose(value, 0.0):
        return ""

    if value_mode == "count":
        return str(int(round(value)))

    if np.isclose(
        value,
        round(value),
    ):
        return f"{value:.1f}"

    return f"{value:.2f}"


def _annotation_color(
    value: float,
    *,
    maximum_value: float,
) -> str:
    """Choose readable text color based on cell intensity."""
    if maximum_value <= 0:
        return "black"

    normalized_value = (
        value / maximum_value
    )

    return (
        "white"
        if normalized_value >= 0.55
        else "black"
    )


def _plot_family_decade_heatmap(
    table: pd.DataFrame,
    *,
    value_mode: str,
    title: str,
    colorbar_label: str,
    caption: str,
    filename: str,
) -> Path:
    """Render and save one compact annotated heatmap."""
    RESEARCH_PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if table.empty:
        figure, axis = plt.subplots(
            figsize=(8, 4),
            dpi=DPI,
        )

        axis.text(
            0.5,
            0.5,
            "No strict improvement events were found.",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )

        axis.set_axis_off()

        output_path = (
            RESEARCH_PLOTS_DIR
            / f"{filename}.png"
        )

        figure.savefig(
            output_path,
            bbox_inches="tight",
        )

        plt.close(figure)

        return output_path

    number_of_families = len(table)
    number_of_columns = len(table.columns)

    figure_width = max(
        7.5,
        1.1 * number_of_columns + 4.8,
    )

    figure_height = max(
        4.2,
        0.55 * number_of_families + 2.3,
    )

    figure, axis = plt.subplots(
        figsize=(
            figure_width,
            figure_height,
        ),
        dpi=DPI,
    )

    matrix = table.to_numpy(
        dtype=float
    )

    color_map = (
        COUNT_COLOR_MAP
        if value_mode == "count"
        else MAGNITUDE_COLOR_MAP
    )

    image = axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=color_map,
        vmin=0,
    )

    axis.set_xticks(
        np.arange(number_of_columns)
    )

    axis.set_xticklabels(
        table.columns,
        fontsize=9,
    )

    axis.set_yticks(
        np.arange(number_of_families)
    )

    axis.set_yticklabels(
        table.index,
        fontsize=9,
    )

    axis.set_xlabel(
        "Decade",
        labelpad=8,
    )

    axis.set_ylabel(
        "Problem family",
        labelpad=8,
    )

    axis.set_title(
        title,
        fontsize=14,
        fontweight="semibold",
        pad=14,
    )

    maximum_value = float(
        np.nanmax(matrix)
    )

    # Draw annotations. Zero cells are intentionally left blank to reduce
    # visual noise in this sparse matrix.
    for row_index in range(
        number_of_families
    ):
        for column_index in range(
            number_of_columns
        ):
            cell_value = matrix[
                row_index,
                column_index,
            ]

            cell_text = _annotation_text(
                cell_value,
                value_mode=value_mode,
            )

            if not cell_text:
                continue

            axis.text(
                column_index,
                row_index,
                cell_text,
                ha="center",
                va="center",
                fontsize=9,
                fontweight=(
                    "bold"
                    if table.columns[column_index]
                    == "Total"
                    else "normal"
                ),
                color=_annotation_color(
                    cell_value,
                    maximum_value=maximum_value,
                ),
            )

    # Add thin boundaries so the chart reads more like a compact table.
    axis.set_xticks(
        np.arange(
            -0.5,
            number_of_columns,
            1,
        ),
        minor=True,
    )

    axis.set_yticks(
        np.arange(
            -0.5,
            number_of_families,
            1,
        ),
        minor=True,
    )

    axis.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=1.5,
    )

    axis.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    # Draw a stronger divider before the Total column.
    total_column_index = (
        number_of_columns - 1
    )

    axis.axvline(
        total_column_index - 0.5,
        linewidth=2.2,
        color="white",
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.035,
        pad=0.025,
    )

    colorbar.set_label(
        colorbar_label,
        rotation=90,
        labelpad=10,
    )

    if value_mode == "count":
        colorbar_ticks = np.arange(
            0,
            int(np.ceil(maximum_value)) + 1,
            1,
        )

        colorbar.set_ticks(
            colorbar_ticks
        )

    figure.text(
        0.5,
        0.02,
        caption,
        ha="center",
        va="bottom",
        fontsize=8.2,
        wrap=True,
    )

    figure.tight_layout(
        rect=[
            0,
            0.07,
            1,
            1,
        ]
    )

    output_path = (
        RESEARCH_PLOTS_DIR
        / f"{filename}.png"
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


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def plot_family_improvements_by_decade() -> Path:
    """Plot strict improvement-event counts by family and decade."""
    quantum_algorithms = get_quantum_algorithms()

    yearly_best = _prepare_yearly_best_algorithms(
        quantum_algorithms
    )

    improvement_events = (
        _detect_strict_improvement_events(
            yearly_best
        )
    )

    count_table = _build_family_decade_table(
        improvement_events,
        value_mode="count",
    )

    caption = (
        "Cells count record-setting algorithms that achieved a strictly "
        "lower time-complexity class than the previous best for the same "
        "problem. Families are sorted by total event count, and decades "
        "use conventional calendar boundaries."
    )

    return _plot_family_decade_heatmap(
        count_table,
        value_mode="count",
        title=(
            "Strict Quantum Algorithm Improvements "
            "by Family and Decade"
        ),
        colorbar_label=(
            "Number of strict improvement events"
        ),
        caption=caption,
        filename=COUNT_FILENAME,
    )


def plot_family_improvement_magnitude_by_decade() -> Path:
    """Plot cumulative improvement magnitude by family and decade."""
    quantum_algorithms = get_quantum_algorithms()

    yearly_best = _prepare_yearly_best_algorithms(
        quantum_algorithms
    )

    improvement_events = (
        _detect_strict_improvement_events(
            yearly_best
        )
    )

    magnitude_table = _build_family_decade_table(
        improvement_events,
        value_mode="magnitude",
    )

    caption = (
        "Cells sum reductions in AlgoWiki's ordinal time-complexity class "
        "across strict improvement events. Larger values represent movement "
        "across more complexity classes, not literal runtime speedup factors."
    )

    return _plot_family_decade_heatmap(
        magnitude_table,
        value_mode="magnitude",
        title=(
            "Magnitude of Quantum Algorithmic Progress "
            "by Family and Decade"
        ),
        colorbar_label=(
            "Cumulative reduction in complexity class"
        ),
        caption=caption,
        filename=MAGNITUDE_FILENAME,
    )


def generate_family_improvement_decade_plots() -> tuple[Path, Path]:
    """Generate both family-by-decade improvement figures."""
    count_plot = (
        plot_family_improvements_by_decade()
    )

    magnitude_plot = (
        plot_family_improvement_magnitude_by_decade()
    )

    return (
        count_plot,
        magnitude_plot,
    )


if __name__ == "__main__":
    generate_family_improvement_decade_plots()
