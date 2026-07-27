"""Plot time-space Pareto frontiers for quantum algorithm families.

Each point represents a quantum algorithm with both a valid time-complexity
class and space-complexity class.

The analysis is performed separately within each problem family. Comparing
Pareto dominance globally across unrelated problem families would not be
meaningful because the algorithms solve different tasks.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data_loader import get_quantum_algorithms
from ..header import PLOTS_DIR


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


def _mark_pareto_optimal(
    family_algorithms: pd.DataFrame,
) -> pd.DataFrame:
    """Mark nondominated algorithms within one family.

    Lower time_class and lower space_class are both preferred.
    """
    marked_algorithms = family_algorithms.copy()

    time_values = (
        marked_algorithms["time_class"]
        .to_numpy(dtype=float)
    )
    space_values = (
        marked_algorithms["space_class"]
        .to_numpy(dtype=float)
    )

    pareto_flags = np.ones(
        len(marked_algorithms),
        dtype=bool,
    )

    for algorithm_index in range(
        len(marked_algorithms)
    ):
        current_time = time_values[
            algorithm_index
        ]
        current_space = space_values[
            algorithm_index
        ]

        dominated_by_another = (
            (time_values <= current_time)
            & (space_values <= current_space)
            & (
                (time_values < current_time)
                | (space_values < current_space)
            )
        )

        dominated_by_another[
            algorithm_index
        ] = False

        if dominated_by_another.any():
            pareto_flags[
                algorithm_index
            ] = False

    marked_algorithms["is_pareto"] = pareto_flags

    return marked_algorithms


def _prepare_pareto_data(
    quantum_algorithms: pd.DataFrame,
    *,
    minimum_family_entries: int,
    maximum_families: int,
) -> pd.DataFrame:
    """Prepare valid time-space observations for selected families."""
    valid_algorithms = quantum_algorithms.dropna(
        subset=[
            "family",
            "time_class",
            "space_class",
        ]
    ).copy()

    valid_algorithms["time_class"] = pd.to_numeric(
        valid_algorithms["time_class"],
        errors="coerce",
    )
    valid_algorithms["space_class"] = pd.to_numeric(
        valid_algorithms["space_class"],
        errors="coerce",
    )

    valid_algorithms = valid_algorithms.dropna(
        subset=[
            "time_class",
            "space_class",
        ]
    )

    family_counts = (
        valid_algorithms
        .groupby("family")
        .size()
        .sort_values(
            ascending=False
        )
    )

    selected_families = (
        family_counts[
            family_counts >= minimum_family_entries
        ]
        .head(maximum_families)
        .index
    )

    valid_algorithms = valid_algorithms[
        valid_algorithms["family"].isin(
            selected_families
        )
    ].copy()

    marked_groups = []

    for _, family_algorithms in (
        valid_algorithms.groupby(
            "family",
            sort=False,
        )
    ):
        marked_groups.append(
            _mark_pareto_optimal(
                family_algorithms
            )
        )

    if not marked_groups:
        valid_algorithms["is_pareto"] = False
        return valid_algorithms

    return pd.concat(
        marked_groups,
        ignore_index=True,
    )


def _clean_algorithm_label(
    value: object,
) -> str:
    """Create a short point label."""
    label = str(value or "").strip()

    if not label:
        return ""

    if len(label) > 24:
        return label[:21] + "..."

    return label


def plot_time_space_pareto_frontier(
    *,
    minimum_family_entries: int = 3,
    maximum_families: int = 6,
) -> Path:
    """Generate separate time-space Pareto panels for selected families."""
    quantum_algorithms = get_quantum_algorithms()

    pareto_data = _prepare_pareto_data(
        quantum_algorithms,
        minimum_family_entries=minimum_family_entries,
        maximum_families=maximum_families,
    )

    selected_families = (
        pareto_data
        .groupby("family")
        .size()
        .sort_values(
            ascending=False
        )
        .index
        .tolist()
    )

    if not selected_families:
        figure, axis = plt.subplots(
            figsize=(8, 4.5),
            dpi=DPI,
        )

        axis.text(
            0.5,
            0.5,
            (
                "No families have enough algorithms with both "
                "time and space classes."
            ),
            horizontalalignment="center",
            verticalalignment="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()

        return _save_figure(
            figure,
            "time_space_pareto_frontier",
        )

    number_of_columns = 2
    number_of_rows = math.ceil(
        len(selected_families)
        / number_of_columns
    )

    figure, axes = plt.subplots(
        number_of_rows,
        number_of_columns,
        figsize=(
            11,
            4.5 * number_of_rows,
        ),
        dpi=DPI,
        squeeze=False,
    )

    flattened_axes = axes.flatten()

    for axis, family in zip(
        flattened_axes,
        selected_families,
    ):
        family_algorithms = pareto_data[
            pareto_data["family"] == family
        ].copy()

        dominated_algorithms = family_algorithms[
            ~family_algorithms["is_pareto"]
        ]

        pareto_algorithms = family_algorithms[
            family_algorithms["is_pareto"]
        ].sort_values(
            [
                "time_class",
                "space_class",
            ]
        )

        axis.scatter(
            dominated_algorithms["time_class"],
            dominated_algorithms["space_class"],
            alpha=0.45,
            marker="o",
            label="Dominated",
        )

        axis.scatter(
            pareto_algorithms["time_class"],
            pareto_algorithms["space_class"],
            s=85,
            marker="D",
            label="Pareto optimal",
            zorder=3,
        )

        if len(pareto_algorithms) >= 2:
            axis.plot(
                pareto_algorithms["time_class"],
                pareto_algorithms["space_class"],
                linestyle="--",
                linewidth=1.2,
                alpha=0.7,
            )

        for _, algorithm_row in (
            pareto_algorithms.iterrows()
        ):
            algorithm_label = _clean_algorithm_label(
                algorithm_row.get(
                    "algorithm",
                    "",
                )
            )

            if not algorithm_label:
                continue

            axis.annotate(
                algorithm_label,
                (
                    algorithm_row["time_class"],
                    algorithm_row["space_class"],
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.5,
            )

        observation_count = len(
            family_algorithms
        )
        pareto_count = int(
            family_algorithms["is_pareto"].sum()
        )

        axis.set_title(
            (
                f"{family}\n"
                f"{pareto_count}/{observation_count} "
                "algorithms on frontier"
            ),
            fontsize=10,
            fontweight="bold",
        )

        axis.set_xlabel(
            "Time-complexity class\n(lower = faster)"
        )
        axis.set_ylabel(
            "Space-complexity class\n(lower = fewer qubits)"
        )

        all_values = pd.concat(
            [
                family_algorithms["time_class"],
                family_algorithms["space_class"],
            ]
        )

        minimum_class = max(
            1,
            int(np.floor(all_values.min())),
        )
        maximum_class = min(
            8,
            int(np.ceil(all_values.max())),
        )

        tick_values = list(
            range(
                minimum_class,
                maximum_class + 1,
            )
        )

        axis.set_xticks(tick_values)
        axis.set_yticks(tick_values)

        axis.set_xticklabels(
            [
                f"{value}\n{CLASS_LABELS.get(value, '')}"
                for value in tick_values
            ],
            fontsize=7,
        )

        axis.set_yticklabels(
            [
                f"{value}\n{CLASS_LABELS.get(value, '')}"
                for value in tick_values
            ],
            fontsize=7,
        )

        axis.grid(
            True,
            alpha=0.18,
            linestyle="--",
        )
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    for unused_axis in flattened_axes[
        len(selected_families):
    ]:
        unused_axis.set_visible(False)

    legend_handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            alpha=0.45,
            label="Dominated algorithm",
        ),
        mlines.Line2D(
            [],
            [],
            marker="D",
            linestyle="None",
            markersize=8,
            label="Pareto-optimal algorithm",
        ),
    ]

    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )

    figure.suptitle(
        "Time-space Pareto frontiers for quantum algorithms",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    figure.text(
        0.5,
        0.005,
        (
            "Pareto-optimal means that no other recorded algorithm in the "
            "same family has both an equal-or-lower time class and an "
            "equal-or-lower space class, with at least one strictly lower."
        ),
        horizontalalignment="center",
        fontsize=8.5,
    )

    figure.tight_layout(
        rect=[0, 0.035, 1, 0.94]
    )

    return _save_figure(
        figure,
        "time_space_pareto_frontier",
    )
