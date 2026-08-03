"""Turn the parsed runtime dataset into the quantities the figures plot.

Everything here works on log10 operation counts produced by
:mod:`src.dataset`, evaluated at an explicit problem size.  Three derived
quantities:

``annual_improvement_rate``
    How fast a problem family got faster, as a compound annual percentage,
    measured from the earliest catalogued algorithm to the one that set the
    record.  This is the quantity behind "How Fast Do Algorithms Improve".

``speedup_series``
    For one family and category, the speedup over a fixed baseline as a
    function of year -- a step function, since it only moves when someone
    publishes a faster algorithm.

``coverage_bands``
    For one problem size, how the problem families divide into "no quantum
    algorithm catalogued", "quantum catalogued but classical still fastest",
    and "quantum faster".

A note on comparing categories.  Quantum runtimes are circuit depths and
classical serial runtimes are sequential step counts, so a ratio between them
is an asymptotic statement, not a wall-clock prediction: it says how the
operation counts diverge as the problem grows, and ignores the constant
factor between a gate and a CPU instruction.  Parallel rows are kept in terms
of total *work* for the same reason -- a span would silently assume unlimited
processors.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dataset import CATEGORY_LABELS, Dataset
from .problems import GENERAL, variation_label

# The three problem sizes every figure is faceted over, matching the reference
# figures in dtontici/parallel-algorithms.
PROBLEM_SIZES = (1e3, 1e6, 1e9)
SIZE_LABELS = {1e3: "1 thousand", 1e6: "1 million", 1e9: "1 billion"}

CATEGORY_ORDER = ("quantum", "serial", "parallel")
CATEGORY_COLORS = {
    "quantum": "#7B3FA0",
    "serial": "#9A9A9A",
    "parallel": "#58D68D",
}

# Compound-annual-rate buckets, as used in the reference figure.
RATE_BUCKETS = [
    (0.00, 0.03, "0-3%"),
    (0.03, 0.10, "3-10%"),
    (0.10, 0.30, "10-30%"),
    (0.30, 1.00, "30-100%"),
    (1.00, 3.00, "100-300%"),
    (3.00, 10.0, "300-1000%"),
    (10.0, np.inf, ">1000%"),
]
BUCKET_LABELS = [label for _, _, label in RATE_BUCKETS]

# Two runtimes within this many orders of magnitude are called a tie; it is
# well below the precision the source formulas actually carry.
TIE_TOLERANCE = 1e-9


def runtime_table(ds: Dataset, size: float) -> pd.DataFrame:
    """One row per (algorithm, variation it solves), with its log10 cost.

    A row whose "Variation" cell lists two problems is counted under both,
    because it genuinely solves both. The ``problem`` column -- family plus
    variation -- is the unit every figure aggregates over; the family alone
    would put quantum chromatic-number algorithms in the same bucket as
    classical 3-colouring ones.
    """
    records = []
    for row in ds.rows:
        try:
            value = row.log10_at(size)
        except Exception:                             # noqa: BLE001
            continue
        for variation in row.variations:
            records.append({
                "family": row.family, "family_label": row.family_label,
                "variation": variation,
                "problem": f"{row.family}||{variation}",
                "problem_label": _problem_label(row.family_label, variation),
                "category": row.category, "year": row.year, "name": row.name,
                "log10_ops": value, "formula": row.cleaned_formula,
            })
    return pd.DataFrame(records)


def _problem_label(family_label: str, variation: str) -> str:
    if variation == GENERAL:
        return family_label
    return f"{family_label} - {variation_label(variation)}"


def problems_with(table: pd.DataFrame, *categories: str) -> list[str]:
    """Problems carrying an algorithm in every one of ``categories``."""
    present = table.groupby("problem")["category"].agg(set)
    wanted = set(categories)
    return sorted(problem for problem, cats in present.items() if wanted <= cats)


def annual_improvement_rate(entries: list[tuple[int, float]]) -> float | None:
    """Compound annual improvement rate implied by a family's history.

    ``entries`` are ``(year, log10 operations)``.  The rate runs from the best
    algorithm known in the first catalogued year to the one that set the
    record, so a long quiet stretch after the last breakthrough does not
    dilute it.  Returns 0.0 when nothing ever improved, and None when the
    record was set in the same year the family first appears -- that is a
    speedup over no elapsed time, which has no annual rate.
    """
    if not entries:
        return None
    ordered = sorted(entries)
    first_year = ordered[0][0]
    base = min(value for year, value in ordered if year == first_year)

    best, best_year = base, first_year
    for year, value in ordered:
        if value < best - TIE_TOLERANCE:
            best, best_year = value, year

    if best_year == first_year:
        return 0.0 if best >= base - TIE_TOLERANCE else None
    span = best_year - first_year
    exponent = (base - best) / span
    # Replacing a 2^n algorithm with an n^2 one inside a few years is a rate
    # far beyond anything the top bucket distinguishes, and 10**exponent
    # overflows. The bucketing only needs to know it is off the scale.
    if exponent > 12:
        return float("inf")
    return float(10.0 ** exponent - 1.0)


def improvement_rates(ds: Dataset, size: float,
                      min_algorithms: int = 2) -> pd.DataFrame:
    """Compound annual improvement rate per (family, category) at ``size``.

    A family represented by a single algorithm has no rate of change to
    measure and is excluded rather than counted as 0%, which would otherwise
    pile up in the lowest bucket and make progress look slower than it was.
    """
    table = runtime_table(ds, size)
    records = []
    for (problem, category), group in table.groupby(["problem", "category"]):
        if len(group) < min_algorithms:
            continue
        entries = list(zip(group["year"], group["log10_ops"]))
        rate = annual_improvement_rate(entries)
        records.append({
            "problem": problem, "problem_label": group["problem_label"].iloc[0],
            "family": group["family"].iloc[0],
            "category": category, "rate": rate, "n_algorithms": len(group),
            "first_year": int(group["year"].min()),
            "span_years": int(group["year"].max() - group["year"].min()),
        })
    return pd.DataFrame(records)


def bucket_rates(rates: pd.Series) -> pd.Series:
    """Share of families falling in each compound-annual-rate bucket."""
    counts = []
    values = rates.dropna()
    for low, high, _ in RATE_BUCKETS:
        if high is RATE_BUCKETS[-1][1]:
            counts.append(int(((values >= low)).sum()))
        else:
            counts.append(int(((values >= low) & (values < high)).sum()))
    total = sum(counts)
    if not total:
        return pd.Series([0.0] * len(RATE_BUCKETS), index=BUCKET_LABELS)
    return pd.Series([c / total for c in counts], index=BUCKET_LABELS)


def frontier_at(entries: list[tuple[int, float]], year: int) -> float:
    """Best log10 runtime known by ``year``; ``inf`` if nothing exists yet."""
    known = [value for y, value in entries if y <= year]
    return min(known) if known else float("inf")


def speedup_series(ds: Dataset, problems: list[str], size: float,
                   years: np.ndarray, baseline_category: str = "serial",
                   ) -> dict[str, pd.DataFrame]:
    """Speedup over a fixed per-problem baseline, per category, per year.

    The baseline is the problem's earliest catalogued classical serial
    algorithm, so every category is measured against the same starting point
    and a category with nothing published yet sits at 1.
    """
    table = runtime_table(ds, size)
    table = table[table["problem"].isin(problems)]

    entries: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in table.itertuples():
        entries[(row.problem, row.category)].append((row.year, row.log10_ops))

    baselines: dict[str, float] = {}
    for problem in problems:
        base_entries = entries.get((problem, baseline_category), [])
        if not base_entries:
            continue
        first_year = min(y for y, _ in base_entries)
        baselines[problem] = min(v for y, v in base_entries if y == first_year)

    out: dict[str, pd.DataFrame] = {}
    for category in CATEGORY_ORDER:
        matrix, kept = [], []
        for problem in problems:
            if problem not in baselines:
                continue
            own = entries.get((problem, category), [])
            base = baselines[problem]
            row = [10.0 ** min(base - frontier_at(own, int(y)), 300)
                   if own else 1.0 for y in years]
            # Before the first algorithm in this category, speedup is 1.
            row = [1.0 if not np.isfinite(v) or v < 1.0 else v for v in row]
            matrix.append(row)
            kept.append(problem)
        if matrix:
            out[category] = pd.DataFrame(matrix, index=kept, columns=years)
    return out


# How each family is classified at a given problem size. The quantum bands are
# graded by the size of the lead, because a binary "quantum wins" hides the
# difference between a square-root speedup and an exponential one.
COVERAGE_BANDS = [
    ("no_quantum", "No quantum algorithm catalogued", "#BFD8E8"),
    ("classical", "Classical serial still fastest", "#F5C8AF"),
    ("lead_small", "Quantum ahead by up to $10^3$x", "#D6BCE8"),
    ("lead_mid", "Quantum ahead by $10^3$-$10^6$x", "#A874C8"),
    ("lead_large", "Quantum ahead by more than $10^6$x", "#5B2C7A"),
]
BAND_KEYS = [key for key, _, _ in COVERAGE_BANDS]
BAND_LABELS = {key: label for key, label, _ in COVERAGE_BANDS}
BAND_COLORS = {key: color for key, _, color in COVERAGE_BANDS}


def best_by_category(ds: Dataset, size: float) -> pd.DataFrame:
    """Best (smallest) log10 runtime per problem and category at ``size``.

    Restricted to problems with a classical serial entry: a problem with no
    classical algorithm on record gives quantum nothing to be compared with.
    """
    table = runtime_table(ds, size)
    best = table.groupby(["problem", "category"])["log10_ops"].min().unstack()
    for category in CATEGORY_ORDER:
        if category not in best.columns:
            best[category] = np.nan
    best = best[best["serial"].notna()]
    labels = table.drop_duplicates("problem").set_index("problem")["problem_label"]
    best = best.join(labels)
    return best


def coverage_profile(best: pd.DataFrame, overhead_log10: float = 0.0
                     ) -> dict[str, int]:
    """Count families in each coverage band.

    ``overhead_log10`` charges every quantum algorithm a constant factor
    before the comparison -- the cost of one logical gate relative to one
    classical instruction, which for an error-corrected machine is many orders
    of magnitude and is invisible in an asymptotic runtime.
    """
    counts = dict.fromkeys(BAND_KEYS, 0)
    for _, row in best.iterrows():
        quantum = row.get("quantum", np.nan)
        if pd.isna(quantum):
            counts["no_quantum"] += 1
            continue
        lead = row["serial"] - (quantum + overhead_log10)
        if lead <= TIE_TOLERANCE:
            counts["classical"] += 1
        elif lead < 3:
            counts["lead_small"] += 1
        elif lead < 6:
            counts["lead_mid"] += 1
        else:
            counts["lead_large"] += 1
    return counts


def coverage_fractions(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values()) or 1
    return {key: counts[key] / total for key in BAND_KEYS}


def quantum_crossovers(ds: Dataset, sizes: np.ndarray) -> pd.DataFrame:
    """Per problem, the smallest swept size from which quantum stays ahead.

    "Stays ahead" matters: a quantum algorithm can lead at small n and fall
    behind at large n, and reporting its first win would misrepresent that.
    """
    profiles = {size: best_by_category(ds, size) for size in sizes}
    problems = sorted(set().union(*(set(p.index) for p in profiles.values())))
    records = []
    for problem in problems:
        leads, label = [], problem
        for size in sizes:
            best = profiles[size]
            if problem not in best.index:
                leads.append(None)
                continue
            row = best.loc[problem]
            label = row["problem_label"]
            quantum = row.get("quantum", np.nan)
            leads.append(None if pd.isna(quantum) else row["serial"] - quantum)
        if any(v is None for v in leads):
            continue
        sustained = None
        for i in range(len(sizes)):
            if all(v > TIE_TOLERANCE for v in leads[i:]):
                sustained = sizes[i]
                break
        records.append({"problem": problem, "problem_label": label,
                        "crossover_size": sustained})
    return pd.DataFrame(records)


def overhead_tolerance(ds: Dataset, size: float) -> pd.DataFrame:
    """How much constant-factor overhead each problem's quantum lead absorbs."""
    best = best_by_category(ds, size)
    records = []
    for problem, row in best.iterrows():
        quantum = row.get("quantum", np.nan)
        if pd.isna(quantum):
            continue
        records.append({"problem": problem, "problem_label": row["problem_label"],
                        "lead_log10": row["serial"] - quantum})
    if not records:
        return pd.DataFrame(columns=["problem", "problem_label", "lead_log10"])
    return pd.DataFrame(records).sort_values("lead_log10", ascending=False)


def category_label(category: str) -> str:
    return CATEGORY_LABELS.get(category, category.title())
