"""Plot a heatmap of strict time-complexity transitions.

Rows represent the previous best time-complexity class.

Columns represent the new, improved time-complexity class.

Each cell contains the number of strict improvement events that moved from
the row class to the column class.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from ..data_loader import get_quantum_algorithms
from ..header import PLOTS_DIR
from ..helpers import build_yearly_improvement_records


OUTPUT_DIR = PLOTS_DIR / "research"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 200

CLASS_LABELS = {
    1: "O(1)",
    2: "O(log n)",
    3: "O(n)",
    4: "O(n log n)",
    5: "O(n²)",
    6: "O(n³)",
    7: "O(n⁴)",
    8: "O(2ⁿ)",
}


def _save_figure(
    figure: plt.Figure,
    filename: str,
) -> Path:
    """Save and close a figure."""
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


def _class_bin(
    value: float,
) -> int:
    """Map a fractional AlgoWiki class to its nearest integer class.

    The heatmap uses broad complexity classes for readability. The exact
    fractional values remain available in the magnitude analyses.
    """
    return int(
        np.clip(
            np.rint(float(value)),
            1,
            8,
        )
    )


def _build_transition_matrix(
    improvement_records: pd.DataFrame,
) -> pd.DataFrame:
    """Build a previous-class by new-class transition matrix."""
    transitions = improvement_records[
        improvement_records["is_strict_improvement"]
    ].dropna(
        subset=["previous_best", "time_class"]
    ).copy()

    if transitions.empty:
        return pd.DataFrame(
            0,
            index=range(1, 9),
            columns=range(1, 9),
        )

    transitions["previous_class_bin"] = (
        transitions["previous_best"]
        .map(_class_bin)
    )

    transitions["new_class_bin"] = (
        transitions["time_class"]
        .map(_class_bin)
    )

    transition_matrix = pd.crosstab(
        transitions["previous_class_bin"],
        transitions["new_class_bin"],
    ).reindex(
        index=range(1, 9),
        columns=range(1, 9),
        fill_value=0,
    )

    return transition_matrix


def plot_complexity_transition_heatmap() -> Path:
    """Generate the strict complexity-transition heatmap."""
    quantum_algorithms = get_quantum_algorithms()

    improvement_records = build_yearly_improvement_records(
        quantum_algorithms,
        group_cols=("family", "variation"),
        metric_col="time_class",
        year_col="year",
    )

    transition_matrix = _build_transition_matrix(
        improvement_records
    )

    matrix_values = transition_matrix.to_numpy()

    figure, axis = plt.subplots(
        figsize=(8.7, 7.2),
        dpi=DPI,
    )

    heatmap = axis.imshow(
        matrix_values,
        aspect="equal",
        interpolation="nearest",
    )

    tick_positions = np.arange(8)

    tick_labels = [
        f"{class_number}\n{CLASS_LABELS[class_number]}"
        for class_number in range(1, 9)
    ]

    axis.set_xticks(tick_positions)
    axis.set_xticklabels(
        tick_labels,
        fontsize=8.5,
    )

    axis.set_yticks(tick_positions)
    axis.set_yticklabels(
        tick_labels,
        fontsize=8.5,
    )

    maximum_count = int(
        matrix_values.max()
    )

    for row_index in range(
        matrix_values.shape[0]
    ):
        for column_index in range(
            matrix_values.shape[1]
        ):
            transition_count = int(
                matrix_values[
                    row_index,
                    column_index,
                ]
            )

            if transition_count == 0:
                continue

            text_color = (
                "white"
                if maximum_count > 0
                and transition_count >= maximum_count * 0.6
                else "black"
            )

            axis.text(
                column_index,
                row_index,
                str(transition_count),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=10,
                fontweight="bold",
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

    axis.set_xlabel(
        "New best time-complexity class"
    )
    axis.set_ylabel(
        "Previous best time-complexity class"
    )

    axis.set_title(
        "How quantum algorithmic complexity classes improve",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    axis.text(
        0.0,
        -0.12,
        (
            "Only strict improvements are shown. Fractional AlgoWiki classes "
            "are rounded to their nearest broad class for this summary."
        ),
        transform=axis.transAxes,
        fontsize=8.5,
        verticalalignment="top",
    )

    figure.tight_layout()

    return _save_figure(
        figure,
        "complexity_transition_heatmap",
    )
