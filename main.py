"""
Entry point for the quantum-algorithms analysis.

Usage:
    python main.py

Comment / uncomment plot calls below to choose which figures to generate.

Generated figures are organized into three groups:

    parallel/
        Parallel-algorithms style
        Compact figures, year + decade variants.

    classic/
        Original visualization style
        Larger figures, year variants only.

    research/
        Additional analyses exploring quantum algorithmic progress:
            • time to first improvement
            • improvement magnitude distribution
            • family improvements by decade
"""
from __future__ import annotations

from src.data_loader import build_processed_dataset
from src.header import PLOTS_DIR
from src.helpers import report_problems_coverage

# Timeline plots
from src.plots.family_improvements_per_year import plot_family_improvements_per_year
from src.plots.improvements_per_year import plot_improvements_per_year
from src.plots.problems_per_year import plot_problems_per_year

# Additional research analyses
from src.plots.time_to_first_improvement_survival import generate_time_to_first_improvement_plots
from src.plots.improvement_magnitude_distribution import plot_improvement_magnitude_distribution
from src.plots.family_improvements_by_decade import plot_family_improvements_by_decade
from src.plots.family_improvements_by_decade import plot_family_improvement_magnitude_by_decade
from src.plots.improvement_magnitude_by_family import plot_improvement_magnitude_by_family
from src.plots.complexity_transition_heatmap import plot_complexity_transition_heatmap
from src.plots.time_space_pareto_frontier import plot_time_space_pareto_frontier


def main() -> None:
    print("[1/2] (Re)building cleaned dataset cache from the AlgoWiki workbook ...")
    qdf, pdf = build_processed_dataset()
    print(f"      quantum algorithms : {len(qdf):>4} rows, "
          f"{qdf.groupby(['family','variation']).ngroups} problems, "
          f"{qdf['family'].nunique()} families")
    print(f"      problems sheet     : {len(pdf):>4} rows")
    report_problems_coverage(qdf, pdf)

    print(f"\n[2/2] Generating plots into {PLOTS_DIR} ...")

    # ---------- Parallel-algorithms style (year + decade variants) -------
    print("\n  >>> parallel-algorithms style")
    plot_problems_per_year(by="problem", bin_by="year",   style="parallel")
    plot_problems_per_year(by="problem", bin_by="decade", style="parallel")
    plot_problems_per_year(by="family",  bin_by="year",   style="parallel")
    plot_problems_per_year(by="family",  bin_by="decade", style="parallel")
    plot_improvements_per_year(bin_by="year",   style="parallel")
    plot_improvements_per_year(bin_by="decade", style="parallel")
    plot_family_improvements_per_year(bin_by="year",   style="parallel")
    plot_family_improvements_per_year(bin_by="decade", style="parallel")
    plot_family_improvements_per_year(bin_by="year",   style="parallel", as_fraction=True)
    plot_family_improvements_per_year(bin_by="decade", style="parallel", as_fraction=True)

    # ---------- Classic style (original look, year-binned) ---------------
    print("\n  >>> classic style  (filenames carry a _classic suffix)")
    plot_problems_per_year(by="problem", style="classic")
    plot_problems_per_year(by="family",  style="classic")
    plot_improvements_per_year(style="classic")
    plot_family_improvements_per_year(style="classic")
    plot_family_improvements_per_year(style="classic", as_fraction=True)

    # ---------- Additional quantum-improvement analyses ------------------
    print("\n  >>> additional quantum-improvement analyses")
    generate_time_to_first_improvement_plots()
    plot_improvement_magnitude_by_family()
    plot_improvement_magnitude_distribution()
    plot_complexity_transition_heatmap()
    plot_time_space_pareto_frontier()

    # Existing family-by-decade analysis
    plot_family_improvements_by_decade()
    plot_family_improvement_magnitude_by_decade()

    print("\nDone.")


if __name__ == "__main__":
    main()
