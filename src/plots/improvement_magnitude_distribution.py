"""Plot the magnitude of quantum algorithmic improvement events.

An improvement magnitude is calculated as:

    previous best time class - new best time class

Because lower AlgoWiki time-complexity classes indicate faster algorithms,
a positive value represents an improvement.

This difference is an ordinal complexity-class change. It should not be
interpreted as a literal runtime speedup factor.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data_loader import get_quantum_algorithms
from ..header import PLOTS_DIR
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


def _build_improvement_magnitude_data(
    improvement_records: pd.DataFrame,
) -> pd.DataFrame:
    """Return only strict improvement events with measurable magnitudes."""
    return (
        improvement_records[
            improvement_records[
                "is_strict_improvement"
            ]
        ]
        .dropna(
            subset=[
                "previous_best",
                "time_class",
                "improvement_size",
            ]
        )
        .copy()
    )


def plot_improvement_magnitude_distribution() -> Path:
    """Generate a histogram of improvement magnitudes."""
    quantum_algorithms = get_quantum_algorithms()

    improvement_records = build_yearly_improvement_records(
        quantum_algorithms,
        group_cols=("family", "variation"),
        metric_col="time_class",
        year_col="year",
    )

    improvements = _build_improvement_magnitude_data(
        improvement_records
    )

    figure, axis = plt.subplots(
        figsize=(8.5, 4.8),
        dpi=DPI,
    )

    if improvements.empty:
        axis.text(
            0.5,
            0.5,
            "No strict improvement events found",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axis.transAxes,
        )
    else:
        largest_improvement = float(
            improvements["improvement_size"].max()
        )

        # The dataset contains fractional classes, so bins of 0.5 retain
        # more information than integer-only bins.
        histogram_bins = np.arange(
            0,
            largest_improvement + 0.51,
            0.5,
        )

        if len(histogram_bins) < 2:
            histogram_bins = np.array(
                [0.0, 0.5]
            )

        axis.hist(
            improvements["improvement_size"],
            bins=histogram_bins,
            edgecolor="white",
        )

        median_improvement = float(
            improvements["improvement_size"].median()
        )

        mean_improvement = float(
            improvements["improvement_size"].mean()
        )

        axis.axvline(
            median_improvement,
            linestyle="--",
            linewidth=1.5,
            label=(
                f"Median reduction: "
                f"{median_improvement:.2f} classes"
            ),
        )

        axis.axvline(
            mean_improvement,
            linestyle=":",
            linewidth=1.5,
            label=(
                f"Mean reduction: "
                f"{mean_improvement:.2f} classes"
            ),
        )

        axis.legend(
            frameon=False,
            fontsize=9,
        )

    axis.set_xlabel(
        "Reduction in time-complexity class"
    )
    axis.set_ylabel(
        "Number of strict improvement events"
    )

    axis.set_title(
        "Magnitude of quantum algorithmic improvements",
        fontsize=13,
        fontweight="bold",
    )

    axis.grid(
        True,
        axis="y",
        alpha=0.2,
        linestyle="--",
    )
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    axis.text(
        0.01,
        -0.18,
        (
            "A larger value indicates that the new yearly best moved "
            "farther down AlgoWiki's ordinal complexity-class scale. "
            "This is a class difference, not a literal speedup factor."
        ),
        transform=axis.transAxes,
        fontsize=9,
        verticalalignment="top",
    )

    figure.tight_layout()

    return _save_figure(
        figure,
        "improvement_magnitude_distribution",
    )
