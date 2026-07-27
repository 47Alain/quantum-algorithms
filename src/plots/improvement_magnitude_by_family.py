"""Plot cumulative quantum improvement magnitude by problem family.

For every strict improvement event:

    improvement_size = previous_best - new_best

The values are then summed across every variation belonging to a family.

Because AlgoWiki's complexity classes are ordinal, the resulting total should
be interpreted as cumulative movement across complexity classes, not as a
literal runtime speedup factor.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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


def _build_family_improvement_summary(
    improvement_records: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize strict improvement magnitudes by family."""
    strict_improvements = improvement_records[
        improvement_records["is_strict_improvement"]
    ].dropna(
        subset=["family", "improvement_size"]
    ).copy()

    if strict_improvements.empty:
        return pd.DataFrame(
            columns=[
                "family",
                "total_improvement",
                "mean_improvement",
                "largest_improvement",
                "improvement_events",
                "problems_improved",
            ]
        )

    family_summary = (
        strict_improvements
        .groupby("family")
        .agg(
            total_improvement=(
                "improvement_size",
                "sum",
            ),
            mean_improvement=(
                "improvement_size",
                "mean",
            ),
            largest_improvement=(
                "improvement_size",
                "max",
            ),
            improvement_events=(
                "improvement_size",
                "size",
            ),
            problems_improved=(
                "variation",
                "nunique",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "total_improvement",
                "improvement_events",
            ],
            ascending=False,
        )
    )

    return family_summary


def plot_improvement_magnitude_by_family(
    *,
    top_n: int | None = 20,
) -> Path:
    """Plot cumulative strict-improvement magnitude by family.

    Parameters
    ----------
    top_n:
        Maximum number of families to show. Pass None to show every family
        that has at least one strict improvement.
    """
    quantum_algorithms = get_quantum_algorithms()

    improvement_records = build_yearly_improvement_records(
        quantum_algorithms,
        group_cols=("family", "variation"),
        metric_col="time_class",
        year_col="year",
    )

    family_summary = _build_family_improvement_summary(
        improvement_records
    )

    if top_n is not None:
        family_summary = family_summary.head(top_n)

    # Reverse the order so the largest bar appears at the top.
    plot_data = family_summary.sort_values(
        "total_improvement",
        ascending=True,
    )

    figure_height = max(
        4.8,
        0.48 * len(plot_data) + 1.8,
    )

    figure, axis = plt.subplots(
        figsize=(9.5, figure_height),
        dpi=DPI,
    )

    if plot_data.empty:
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
        bars = axis.barh(
            plot_data["family"],
            plot_data["total_improvement"],
        )

        maximum_value = float(
            plot_data["total_improvement"].max()
        )

        label_offset = max(
            maximum_value * 0.015,
            0.02,
        )

        for bar, (_, row) in zip(
            bars,
            plot_data.iterrows(),
        ):
            total_improvement = float(
                row["total_improvement"]
            )
            event_count = int(
                row["improvement_events"]
            )

            axis.text(
                total_improvement + label_offset,
                bar.get_y() + bar.get_height() / 2,
                (
                    f"{total_improvement:.2f} "
                    f"({event_count} events)"
                ),
                verticalalignment="center",
                fontsize=8.5,
            )

        axis.set_xlim(
            right=maximum_value * 1.28
        )

        axis.set_xlabel(
            "Cumulative reduction in time-complexity class"
        )
        axis.set_ylabel(
            "Problem family"
        )

        axis.grid(
            True,
            axis="x",
            alpha=0.2,
            linestyle="--",
        )
        axis.set_axisbelow(True)

    axis.set_title(
        "Quantum problem families with the largest cumulative improvements",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    axis.text(
        0.0,
        -0.09,
        (
            "Totals sum ordinal complexity-class reductions across strict "
            "improvement events. They are not literal runtime speedup factors."
        ),
        transform=axis.transAxes,
        fontsize=8.5,
        verticalalignment="top",
    )

    figure.tight_layout()

    return _save_figure(
        figure,
        "improvement_magnitude_by_family",
    )
