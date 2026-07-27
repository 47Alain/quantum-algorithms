"""Plot quantum algorithmic improvements by family and decade.

The heatmap contains:

    rows    = problem families
    columns = decades
    values  = number of strict improvement events

Only families with at least one strict improvement are included.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from ..data_loader import get_quantum_algorithms
from ..header import DECADES, PLOTS_DIR
from ..helpers import build_yearly_improvement_records


OUTPUT_DIR = PLOTS_DIR / "research"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 200


def _save_figure(
    figure: plt.Figure,
    filename: str,
) -> Path:
    """Save a figure in the research plots directory."""
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


def _assign_decade_label(
    year: int,
) -> str | None:
    """Assign a year to one of the configured decade buckets."""
    for decade in DECADES:
        if year <= decade["max"]:
            return decade["label"]

    return None


def _build_family_decade_counts(
    improvement_records: pd.DataFrame,
) -> pd.DataFrame:
    """Count strict improvement events by family and decade."""
    strict_improvements = improvement_records[
        improvement_records["is_strict_improvement"]
    ].copy()

    strict_improvements["decade"] = (
        strict_improvements["year"]
        .astype(int)
        .map(_assign_decade_label)
    )

    strict_improvements = (
        strict_improvements.dropna(
            subset=["family", "decade"]
        )
    )

    decade_labels = [
        decade["label"]
        for decade in DECADES
    ]

    family_decade_counts = (
        strict_improvements
        .groupby(["family", "decade"])
        .size()
        .unstack(fill_value=0)
        .reindex(
            columns=decade_labels,
            fill_value=0,
        )
    )

    family_decade_counts["total"] = (
        family_decade_counts.sum(axis=1)
    )

    family_decade_counts = (
        family_decade_counts[
            family_decade_counts["total"] > 0
        ]
        .sort_values(
            by=[
                "total",
                *decade_labels,
            ],
            ascending=False,
        )
        .drop(columns="total")
    )

    return family_decade_counts


def plot_family_improvements_by_decade() -> Path:
    """Generate a family-by-decade improvement heatmap."""
    quantum_algorithms = get_quantum_algorithms()

    improvement_records = build_yearly_improvement_records(
        quantum_algorithms,
        group_cols=("family", "variation"),
        metric_col="time_class",
        year_col="year",
    )

    family_decade_counts = _build_family_decade_counts(
        improvement_records
    )

    number_of_families = len(
        family_decade_counts
    )

    figure_height = max(
        4.5,
        0.38 * number_of_families + 1.8,
    )

    figure, axis = plt.subplots(
        figsize=(8.5, figure_height),
        dpi=DPI,
    )

    if family_decade_counts.empty:
        axis.text(
            0.5,
            0.5,
            "No strict improvement events found",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axis.transAxes,
        )

        axis.set_axis_off()
    else:
        decade_labels = list(
            family_decade_counts.columns
        )

        improvement_matrix = (
            family_decade_counts.to_numpy()
        )

        heatmap = axis.imshow(
            improvement_matrix,
            aspect="auto",
            interpolation="nearest",
        )

        axis.set_xticks(
            np.arange(len(decade_labels))
        )
        axis.set_xticklabels(
            decade_labels
        )

        axis.set_yticks(
            np.arange(number_of_families)
        )
        axis.set_yticklabels(
            family_decade_counts.index,
            fontsize=8.5,
        )

        maximum_count = int(
            improvement_matrix.max()
        )

        # Display nonzero values inside the heatmap cells.
        for row_index in range(
            improvement_matrix.shape[0]
        ):
            for column_index in range(
                improvement_matrix.shape[1]
            ):
                improvement_count = int(
                    improvement_matrix[
                        row_index,
                        column_index,
                    ]
                )

                if improvement_count == 0:
                    continue

                text_color = (
                    "white"
                    if maximum_count > 0
                    and improvement_count
                    >= maximum_count * 0.6
                    else "black"
                )

                axis.text(
                    column_index,
                    row_index,
                    str(improvement_count),
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=8.5,
                    color=text_color,
                )

        colorbar = figure.colorbar(
            heatmap,
            ax=axis,
            pad=0.02,
        )

        colorbar.set_label(
            "Strict improvement events"
        )

        colorbar.locator = mtick.MaxNLocator(
            integer=True
        )
        colorbar.update_ticks()

        axis.set_xlabel("Decade")
        axis.set_ylabel("Problem family")

    axis.set_title(
        "Quantum algorithmic improvements by family and decade",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    figure.tight_layout()

    return _save_figure(
        figure,
        "family_improvements_by_decade",
    )
