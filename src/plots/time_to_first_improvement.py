"""Analyze how long quantum problems take to receive a first improvement.

This module produces a two-panel figure:

1. A histogram showing the observed number of years between the first
   classifiable algorithm for a problem and its first strict improvement.
2. A Kaplan-Meier survival curve showing the fraction of problems that have
   not yet received a strict improvement after a given number of years.

A problem is defined as a (family, variation) pair.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from ..data_loader import get_quantum_algorithms
from ..header import CUR_YEAR, PLOTS_DIR
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


def _build_time_to_first_improvement_data(
    improvement_records: pd.DataFrame,
) -> pd.DataFrame:
    """Create one time-to-event row for every measurable problem.

    For problems that receive a strict improvement, duration is the number
    of years from the first classifiable algorithm to the first improvement.

    Problems that do not receive an improvement are right-censored at
    CUR_YEAR.
    """
    problem_columns = ["family", "variation"]
    result_rows: list[dict] = []

    grouped_problems = improvement_records.groupby(
        problem_columns,
        sort=False,
    )

    for problem_key, problem_records in grouped_problems:
        problem_records = problem_records.sort_values("year")

        first_classified_year = int(
            problem_records["year"].min()
        )

        strict_improvements = problem_records[
            problem_records["is_strict_improvement"]
        ]

        if strict_improvements.empty:
            first_improvement_year = np.nan
            duration = max(
                CUR_YEAR - first_classified_year,
                0,
            )
            event_observed = False
        else:
            first_improvement_year = int(
                strict_improvements["year"].min()
            )
            duration = (
                first_improvement_year
                - first_classified_year
            )
            event_observed = True

        family, variation = problem_key

        result_rows.append(
            {
                "family": family,
                "variation": variation,
                "first_classified_year": first_classified_year,
                "first_improvement_year": first_improvement_year,
                "duration": duration,
                "event_observed": event_observed,
            }
        )

    return pd.DataFrame(result_rows)


def _compute_kaplan_meier_curve(
    durations: pd.Series,
    events_observed: pd.Series,
) -> pd.DataFrame:
    """Compute a Kaplan-Meier estimate.

    Survival means that a problem has not yet received its first strict
    time-complexity improvement.
    """
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


def plot_time_to_first_improvement() -> Path:
    """Generate the time-to-first-improvement figure."""
    quantum_algorithms = get_quantum_algorithms()

    improvement_records = build_yearly_improvement_records(
        quantum_algorithms,
        group_cols=("family", "variation"),
        metric_col="time_class",
        year_col="year",
    )

    time_data = _build_time_to_first_improvement_data(
        improvement_records
    )

    observed_improvements = time_data[
        time_data["event_observed"]
    ].copy()

    survival_curve = _compute_kaplan_meier_curve(
        time_data["duration"],
        time_data["event_observed"],
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(11, 4.6),
        dpi=DPI,
    )

    histogram_axis = axes[0]
    survival_axis = axes[1]

    # ------------------------------------------------------------------
    # Left panel: histogram of observed first-improvement delays
    # ------------------------------------------------------------------
    if observed_improvements.empty:
        histogram_axis.text(
            0.5,
            0.5,
            "No strict improvements found",
            horizontalalignment="center",
            verticalalignment="center",
            transform=histogram_axis.transAxes,
        )
    else:
        maximum_delay = int(
            observed_improvements["duration"].max()
        )

        histogram_bins = np.arange(
            -0.5,
            maximum_delay + 1.5,
            1,
        )

        histogram_axis.hist(
            observed_improvements["duration"],
            bins=histogram_bins,
            edgecolor="white",
        )

        median_delay = float(
            observed_improvements["duration"].median()
        )

        histogram_axis.axvline(
            median_delay,
            linestyle="--",
            linewidth=1.5,
            label=(
                f"Median observed delay: "
                f"{median_delay:g} years"
            ),
        )

        histogram_axis.legend(
            frameon=False,
            fontsize=9,
        )

    histogram_axis.set_xlabel(
        "Years from first classified algorithm"
    )
    histogram_axis.set_ylabel(
        "Number of problems"
    )
    histogram_axis.set_title(
        "Observed time to first improvement"
    )

    histogram_axis.grid(
        True,
        axis="y",
        alpha=0.2,
        linestyle="--",
    )
    histogram_axis.set_axisbelow(True)
    histogram_axis.spines["top"].set_visible(False)
    histogram_axis.spines["right"].set_visible(False)

    # ------------------------------------------------------------------
    # Right panel: Kaplan-Meier curve
    # ------------------------------------------------------------------
    survival_axis.step(
        survival_curve["year"],
        survival_curve["survival_probability"],
        where="post",
        linewidth=2.2,
    )

    maximum_follow_up = (
        int(time_data["duration"].max())
        if not time_data.empty
        else 0
    )

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
            survival_axis.hlines(
                final_survival_probability,
                xmin=final_event_time,
                xmax=maximum_follow_up,
                linewidth=2.2,
            )

    survival_axis.set_xlim(
        left=0,
        right=max(maximum_follow_up, 1),
    )
    survival_axis.set_ylim(0, 1.03)

    survival_axis.yaxis.set_major_formatter(
        mtick.PercentFormatter(1.0)
    )

    survival_axis.set_xlabel(
        "Years since first classified algorithm"
    )
    survival_axis.set_ylabel(
        "Problems not yet improved"
    )
    survival_axis.set_title(
        "Time until first strict improvement"
    )

    survival_axis.grid(
        True,
        axis="y",
        alpha=0.2,
        linestyle="--",
    )
    survival_axis.set_axisbelow(True)
    survival_axis.spines["top"].set_visible(False)
    survival_axis.spines["right"].set_visible(False)

    measurable_problem_count = len(time_data)

    improved_problem_count = int(
        time_data["event_observed"].sum()
    )

    censored_problem_count = (
        measurable_problem_count
        - improved_problem_count
    )

    figure.suptitle(
        "How long does quantum algorithmic improvement take?",
        fontsize=13,
        fontweight="bold",
    )

    figure.text(
        0.5,
        0.01,
        (
            f"{measurable_problem_count} measurable problems; "
            f"{improved_problem_count} received a strict improvement; "
            f"{censored_problem_count} had not improved by {CUR_YEAR}. "
            "The baseline is the first algorithm with a valid time class."
        ),
        horizontalalignment="center",
        fontsize=9,
    )

    figure.tight_layout(
        rect=[0, 0.06, 1, 0.94]
    )

    return _save_figure(
        figure,
        "time_to_first_improvement",
    )
