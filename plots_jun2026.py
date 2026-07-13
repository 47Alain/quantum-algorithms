"""
Jun 2026 quantum-algorithm analysis plots.
Output: Plots/jun2026/*.png

Plots generated
---------------
1. best_by_family             — best time-complexity class (quantum vs serial vs
                                parallel) for every problem family that appears in
                                two or more categories.
2. runtime_distribution       — box-plot of time_class values for the most-studied
                                quantum problem families, with classical medians
                                overlaid.
3. best_histogram             — distribution of *best* (minimum) time_class across
                                all shared families: quantum vs serial.
4. improvement_over_time      — cumulative algorithmic improvement events by year
                                for serial, parallel and quantum algorithms.
5. time_space_tradeoff        — scatter of circuit-depth class vs qubit-space class
                                for the top quantum families.
6. family_improvement_timeline — per-family step-function of best known time-class
                                 over the years (quantum vs classical). Mirrors the
                                 style of Figure 3 in andrewlucasgs/approximation-
                                 algorithms: each step = a new algorithm that beats
                                 the prior best, annotated with the algorithm name.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
XLSX = REPO_ROOT / "data" / "AlgoWiki algorithms (our copy) (3).xlsx"
OUT_DIR = REPO_ROOT / "Plots" / "jun2026"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────────────
C_QUANTUM  = "#4C72B0"   # blue
C_SERIAL   = "#DD8452"   # orange
C_PARALLEL = "#55A868"   # green
ALPHA      = 0.85
DPI        = 200


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm_family(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.title().replace({"": None})


def load_quantum(xls: pd.ExcelFile) -> pd.DataFrame:
    raw = xls.parse("Quantum Algorithms")
    df = raw[["Family Name", "Variation", "Year",
               "Time Complexity Class",
               "Time Complexity / Circuit Depth (Worst Only)",
               "Space Complexity Class",
               "Space (QBit) Complexity (Auxiliary)"]].copy()
    df.columns = ["family", "variation", "year",
                  "time_class", "time_expr",
                  "space_class", "space_expr"]
    df["family"] = _norm_family(df["family"])
    for col in ["time_class", "space_class", "year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["family", "year"]).copy()
    df["year"] = df["year"].astype(int)
    df["category"] = "quantum"
    return df


def load_serial(xls: pd.ExcelFile) -> pd.DataFrame:
    raw = xls.parse("Sheet1")
    df = raw[["Family Name", "Variation", "Year",
               "Time Complexity Class", "Time Complexity (Worst Only)",
               "Parallel?", "Quantum?"]].copy()
    df.columns = ["family", "variation", "year",
                  "time_class", "time_expr", "parallel", "quantum"]
    df["family"] = _norm_family(df["family"])
    for col in ["time_class", "year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["family", "year"]).copy()
    df["year"] = df["year"].astype(int)
    # serial = not parallel, not quantum
    df = df[(df["parallel"] == 0) & (df["quantum"] != 1)].copy()
    df["category"] = "serial"
    return df[["family", "variation", "year", "time_class", "time_expr", "category"]]


def load_parallel(xls: pd.ExcelFile) -> pd.DataFrame:
    raw = xls.parse("Parallel Algos")
    df = raw[["Family Name", "Variation", "Year",
               "Time Complexity Class", "Time Complexity (Worst Only)"]].copy()
    df.columns = ["family", "variation", "year", "time_class", "time_expr"]
    df["family"] = _norm_family(df["family"])
    for col in ["time_class", "year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["family", "year"]).copy()
    df["year"] = df["year"].astype(int)
    df["category"] = "parallel"
    return df[["family", "variation", "year", "time_class", "time_expr", "category"]]


def mark_improvements(df: pd.DataFrame,
                      group_cols=("family", "variation"),
                      metric_col: str = "time_class",
                      year_col: str = "year") -> pd.DataFrame:
    """Return df with boolean columns `is_first` and `is_improvement`."""
    work = df.copy().sort_values(list(group_cols) + [year_col], kind="stable")
    is_first, is_improvement = [], []
    best_metric: dict = {}
    seen: set = set()

    for _, row in work.iterrows():
        key = tuple(row[c] for c in group_cols)
        metric = row.get(metric_col)
        first = key not in seen
        seen.add(key)

        if first:
            is_first.append(True)
            improved = True
            if pd.notna(metric):
                best_metric[key] = float(metric)
        else:
            is_first.append(False)
            if pd.isna(metric):
                improved = False
            else:
                m = float(metric)
                cur_best = best_metric.get(key)
                if cur_best is None or m < cur_best:
                    improved = True
                    best_metric[key] = m
                else:
                    improved = False
        is_improvement.append(improved)

    work["is_first"] = is_first
    work["is_improvement"] = is_improvement
    return work


def save(fig: plt.Figure, name: str) -> Path:
    p = OUT_DIR / f"{name}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved -> {p.relative_to(REPO_ROOT)}")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — best time_class per shared family (quantum vs serial vs parallel)
# ─────────────────────────────────────────────────────────────────────────────

# Time complexity class legend (approximate mapping, used in annotation)
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


def plot_best_by_family(qdf, serial, parallel):
    best_q = (qdf.dropna(subset=["time_class"])
                 .groupby("family")["time_class"].min()
                 .rename("quantum"))
    best_s = (serial.dropna(subset=["time_class"])
                    .groupby("family")["time_class"].min()
                    .rename("serial"))
    best_p = (parallel.dropna(subset=["time_class"])
                      .groupby("family")["time_class"].min()
                      .rename("parallel"))

    comp = pd.concat([best_q, best_s, best_p], axis=1)
    comp = comp.dropna(subset=["quantum"])
    comp = comp[comp[["serial", "parallel"]].notna().any(axis=1)]

    comp["classical_best"] = comp[["serial", "parallel"]].min(axis=1)
    comp["gap"] = comp["classical_best"] - comp["quantum"]
    comp = comp.sort_values("gap", ascending=False).head(20)
    comp = comp.sort_values("quantum")

    families = comp.index.tolist()
    n = len(families)
    y_pos = np.arange(n)

    bar_h = 0.26
    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.42)), dpi=DPI)

    ax.barh(y_pos + bar_h, comp["quantum"].values,
            height=bar_h, color=C_QUANTUM, alpha=ALPHA, label="Quantum")
    s_vals = comp["serial"].values
    mask_s = ~np.isnan(s_vals)
    if mask_s.any():
        ax.barh(y_pos[mask_s], s_vals[mask_s],
                height=bar_h, color=C_SERIAL, alpha=ALPHA, label="Serial (classical)")
    p_vals = comp["parallel"].values
    mask_p = ~np.isnan(p_vals)
    if mask_p.any():
        ax.barh(y_pos[mask_p] - bar_h, p_vals[mask_p],
                height=bar_h, color=C_PARALLEL, alpha=ALPHA, label="Parallel (classical)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(families, fontsize=10)
    ax.set_xlabel("Best time-complexity class (lower = faster)", fontsize=11)
    ax.set_title(
        "Best known algorithm: quantum vs serial vs parallel\n"
        "(per shared problem family, sorted by quantum–classical gap)",
        fontsize=12, fontweight="bold", pad=12,
    )

    max_x = int(np.nanmax(comp[["quantum", "serial", "parallel"]].values)) + 1
    ax.set_xticks(range(1, max_x + 1))
    ax.set_xticklabels(
        [f"{i}\n({CLASS_LABELS.get(i, '')})" for i in range(1, max_x + 1)],
        fontsize=8.5,
    )

    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.18, linestyle="--")
    fig.tight_layout()
    save(fig, "1_best_by_family")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — runtime distribution for top quantum families
# ─────────────────────────────────────────────────────────────────────────────

def plot_runtime_distribution(qdf, serial, parallel):
    top_q_families = (
        qdf.dropna(subset=["time_class"])
           .groupby("family")["time_class"]
           .agg(["count", "std"])
           .query("count >= 4")
           .sort_values("count", ascending=False)
           .head(10)
           .index.tolist()
    )

    classical = pd.concat([serial, parallel], ignore_index=True)

    n = len(top_q_families)
    # Two-column layout: cleaner and more readable than 5-wide
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(10, nrows * 3.2),
                             dpi=DPI)
    axes = np.array(axes).flatten()

    for idx, fam in enumerate(top_q_families):
        ax = axes[idx]
        q_vals = qdf[qdf["family"] == fam]["time_class"].dropna().values
        c_vals = classical[classical["family"] == fam]["time_class"].dropna().values

        data_list, positions, colors, labels = [], [], [], []
        if len(q_vals) > 0:
            data_list.append(q_vals)
            positions.append(1); colors.append(C_QUANTUM); labels.append("Quantum")
        if len(c_vals) > 0:
            data_list.append(c_vals)
            positions.append(2); colors.append(C_SERIAL); labels.append("Classical")

        if data_list:
            bp = ax.boxplot(data_list, positions=positions[:len(data_list)],
                            widths=0.55, patch_artist=True,
                            medianprops=dict(color="black", linewidth=2),
                            whiskerprops=dict(linewidth=1.2),
                            capprops=dict(linewidth=1.2),
                            flierprops=dict(marker="o", markersize=4, alpha=0.6))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(ALPHA)

        ax.set_xticks(positions[:len(data_list)])
        ax.set_xticklabels(labels[:len(data_list)], fontsize=10)
        title = fam if len(fam) <= 28 else fam.replace(" ", "\n", 2)
        ax.set_title(f"{title}  (n={len(q_vals)})", fontsize=10, pad=4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_locator(mtick.MaxNLocator(integer=False, nbins=5))

        lo = int(np.nanmin([v.min() for v in data_list] + [99])) - 1
        hi = int(np.nanmax([v.max() for v in data_list] + [0])) + 1
        ticks = [t for t in range(max(1, lo), min(9, hi) + 1) if CLASS_LABELS.get(t)]
        if ticks:
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{t}  ({CLASS_LABELS[t]})" for t in ticks], fontsize=8.5)
        ax.grid(True, axis="y", alpha=0.15, linestyle="--")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.text(0.01, 0.5, "Time complexity class\n(lower = faster)",
             va="center", rotation="vertical", fontsize=11)

    handles = [
        mpatches.Patch(facecolor=C_QUANTUM, alpha=ALPHA, label="Quantum"),
        mpatches.Patch(facecolor=C_SERIAL,  alpha=ALPHA, label="Classical (serial + parallel)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Time-complexity distribution — top quantum problem families",
                 y=1.04, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0.04, 0, 1, 0.97])
    save(fig, "2_runtime_distribution")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — histogram of best serial vs best quantum
# ─────────────────────────────────────────────────────────────────────────────

def plot_best_histogram(qdf, serial, parallel):
    best_q = (qdf.dropna(subset=["time_class"])
                 .groupby("family")["time_class"].min()
                 .rename("quantum"))
    best_s = (serial.dropna(subset=["time_class"])
                    .groupby("family")["time_class"].min()
                    .rename("serial"))
    best_p = (parallel.dropna(subset=["time_class"])
                      .groupby("family")["time_class"].min()
                      .rename("parallel"))

    comp = pd.concat([best_q, best_s, best_p], axis=1)
    comp["classical_best"] = comp[["serial", "parallel"]].min(axis=1)

    shared = comp.dropna(subset=["quantum", "classical_best"]).copy()
    shared["gap"] = shared["classical_best"] - shared["quantum"]

    bins = np.arange(1, 10, 0.5)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=DPI)

    # Left: overlaid histograms
    ax = axes[0]
    ax.hist(shared["classical_best"], bins=bins, color=C_SERIAL,
            alpha=0.65, label="Best classical (serial or parallel)", edgecolor="white")
    ax.hist(shared["quantum"], bins=bins, color=C_QUANTUM,
            alpha=0.65, label="Best quantum", edgecolor="white")
    ax.set_xlabel("Time complexity class (lower = faster)", fontsize=11)
    ax.set_ylabel("Number of problem families", fontsize=11)
    ax.set_title("Distribution of best algorithms\nacross shared problem families",
                 fontsize=11, fontweight="bold")
    ax.set_xticks(range(1, 10))
    ax.set_xticklabels(
        [f"{i}\n({CLASS_LABELS.get(i,'')})" for i in range(1, 10)], fontsize=8.5
    )
    ax.legend(frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.18, linestyle="--")

    # Right: scatter — best classical vs best quantum per family
    ax2 = axes[1]
    ax2.scatter(shared["classical_best"], shared["quantum"],
                color=C_QUANTUM, alpha=0.75, s=65, edgecolors="white", linewidths=0.5)
    lo = min(shared["classical_best"].min(), shared["quantum"].min()) - 0.5
    hi = max(shared["classical_best"].max(), shared["quantum"].max()) + 0.5
    ax2.plot([lo, hi], [lo, hi], color="gray", linestyle="--",
             linewidth=1, label="Parity (same class)")
    ax2.set_xlabel("Best classical time-complexity class", fontsize=11)
    ax2.set_ylabel("Best quantum time-complexity class", fontsize=11)
    ax2.set_title("Classical vs quantum best algorithm\nper shared problem family",
                  fontsize=11, fontweight="bold")
    ax2.legend(frameon=False, fontsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True, alpha=0.15, linestyle="--")

    # Annotate top families with largest quantum advantage;
    # use adjust_text-style manual placement to avoid pile-up
    notable = (shared[shared["gap"] >= 1.5]
               .sort_values("gap", ascending=False)
               .head(10))
    # Stagger: alternate left/right and shift y to reduce overlap
    used_y: list[float] = []
    for i, (fam, row) in enumerate(notable.iterrows()):
        x, y = float(row["classical_best"]), float(row["quantum"])
        ha = "left" if x <= 5 else "right"
        xoff = 7 if ha == "left" else -7
        # Nudge vertically if too close to a previous label
        y_text = y
        for prev in used_y:
            if abs(y_text - prev) < 0.45:
                y_text += 0.45
        used_y.append(y_text)
        ax2.annotate(
            fam, xy=(x, y), fontsize=7.5, ha=ha, color="#333333",
            xytext=(xoff, (y_text - y) * 20 + 3),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.6),
        )

    fig.suptitle("Best classical vs best quantum algorithm by problem family",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "3_best_histogram")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — algorithm improvements over time (serial / parallel / quantum)
# ─────────────────────────────────────────────────────────────────────────────

def _cumulative_improvements(df: pd.DataFrame,
                              year_range: range) -> pd.Series:
    """Mark improvements then return cumulative yearly improvement count."""
    annotated = mark_improvements(df)
    impr = annotated[annotated["is_improvement"] & ~annotated["is_first"]]
    counts = (impr["year"].dropna().astype(int)
                          .value_counts()
                          .reindex(year_range, fill_value=0))
    return counts.sort_index().cumsum()


def plot_improvement_over_time(qdf, serial, parallel):
    year_range = range(1940, 2027)

    cum_s = _cumulative_improvements(serial,   year_range)
    cum_p = _cumulative_improvements(parallel, year_range)
    cum_q = _cumulative_improvements(qdf,      year_range)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=DPI)

    ax.plot(cum_s.index, cum_s.values,
            color=C_SERIAL,   linewidth=2.2, label="Serial (classical)", alpha=0.9)
    ax.plot(cum_p.index, cum_p.values,
            color=C_PARALLEL, linewidth=2.2, label="Parallel (classical)", alpha=0.9)
    ax.plot(cum_q.index, cum_q.values,
            color=C_QUANTUM,  linewidth=2.5, label="Quantum", alpha=0.9,
            linestyle="--")

    ax.axvline(1994, color=C_QUANTUM, linewidth=0.9, linestyle=":", alpha=0.55)
    ylo, yhi = ax.get_ylim()
    ax.text(1995, ylo + (yhi - ylo) * 0.42, "Shor's algorithm\n(1994)",
            color=C_QUANTUM, fontsize=9, va="bottom")

    ax.set_xlim(1940, 2026)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Cumulative improvement events", fontsize=11)
    ax.set_title(
        "Cumulative algorithmic improvements over time\n"
        "— serial, parallel, and quantum algorithms —",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.2, linestyle="--")
    fig.tight_layout()
    save(fig, "4_improvement_over_time")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — time-space (qubit) tradeoff for quantum algorithms
# ─────────────────────────────────────────────────────────────────────────────

def plot_time_space_tradeoff(qdf):
    sub = qdf.dropna(subset=["time_class", "space_class"]).copy()

    counts = sub.groupby("family").size()
    top_fams = counts[counts >= 3].sort_values(ascending=False).index.tolist()
    sub = sub[sub["family"].isin(top_fams)].copy()

    palette = plt.colormaps.get_cmap("tab10").resampled(len(top_fams))
    fam_color = {f: palette(i) for i, f in enumerate(top_fams)}

    fig, ax = plt.subplots(figsize=(9, 6), dpi=DPI)

    for fam in top_fams:
        grp = sub[sub["family"] == fam]
        ax.scatter(grp["time_class"], grp["space_class"],
                   color=fam_color[fam], s=75, alpha=0.82,
                   edgecolors="white", linewidths=0.6, label=fam)

    ax.set_xlabel("Circuit-depth / time complexity class\n(lower = faster)", fontsize=11)
    ax.set_ylabel("Qubit / space complexity class\n(lower = fewer qubits)", fontsize=11)
    ax.set_title(
        "Time–space tradeoff for quantum algorithms\n"
        "(top families with ≥ 3 entries having both complexity classes)",
        fontsize=12, fontweight="bold", pad=10,
    )

    max_t = int(sub["time_class"].max()) + 1
    max_s = int(sub["space_class"].max()) + 1
    ax.set_xticks(range(1, max_t + 1))
    ax.set_xticklabels(
        [f"{i}\n({CLASS_LABELS.get(i,'')})" for i in range(1, max_t + 1)],
        fontsize=8.5,
    )
    ax.set_yticks(range(1, max_s + 1))
    ax.set_yticklabels(
        [f"{i}  ({CLASS_LABELS.get(i,'')})" for i in range(1, max_s + 1)],
        fontsize=8.5,
    )

    ax.legend(frameon=False, fontsize=9, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.18, linestyle="--")
    fig.tight_layout()
    save(fig, "5_time_space_tradeoff")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 — per-family improvement trajectory over time
# ─────────────────────────────────────────────────────────────────────────────

# Category display config
_CAT_STYLE = {
    "quantum":  dict(color=C_QUANTUM,  linestyle="--", linewidth=2.4, label="Quantum",            zorder=4),
    "serial":   dict(color=C_SERIAL,   linestyle="-",  linewidth=2.0, label="Serial (classical)",  zorder=3),
    "parallel": dict(color=C_PARALLEL, linestyle="-.", linewidth=1.6, label="Parallel (classical)", zorder=2),
}


def _improvements_for(df: pd.DataFrame, family: str) -> list[tuple[int, float, str]]:
    """Return chronological list of (year, time_class, label) for a family.

    Pass either a full dataframe (df) or a pre-filtered subset. If df is the
    full dataframe, the family column is used to filter; if the subset already
    corresponds to one family, the filter is a no-op.

    Only entries that strictly improve (lower time_class) on the prior best
    are included. The first entry for the family is always included regardless.
    Label is the algorithm name, truncated for readability.
    """
    mask = df["family"].str.lower() == family.lower()
    if not mask.any():
        # Fall back to the pre-filtered subset (caller already filtered)
        sub = df.dropna(subset=["time_class", "year"]).sort_values("year").copy()
    else:
        sub = (df[mask]
                 .dropna(subset=["time_class", "year"])
                 .sort_values("year")
                 .copy())
    results: list[tuple[int, float, str]] = []
    best: float | None = None
    for _, row in sub.iterrows():
        tc = float(row["time_class"])
        yr = int(row["year"])
        alg = str(row.get("algorithm", "") or row.get("variation", "") or "")
        alg = alg.strip()
        for noise in (" Algorithm", " algorithm", "'s Algorithm", "'s algorithm"):
            if alg.endswith(noise):
                alg = alg[: -len(noise)]
        alg = alg[:28]
        if best is None or tc < best:
            results.append((yr, tc, alg))
            best = tc
    return results


def _step_xy(improvements: list[tuple[int, float, str]],
             extend_to: int = 2026) -> tuple[list[int], list[float]]:
    """Convert improvement events into step-function (x, y) arrays.

    The line stays flat at each level until the next improvement, then
    drops vertically to the new value (post-step style).
    """
    if not improvements:
        return [], []
    xs: list[int] = []
    ys: list[float] = []
    for i, (yr, tc, _) in enumerate(improvements):
        if i == 0:
            # Extend leftward a bit so the first entry is visible
            xs.append(yr - 1)
            ys.append(tc)
        else:
            # Horizontal hold at prior level until this year
            xs.append(yr)
            ys.append(ys[-1])
        xs.append(yr)
        ys.append(tc)
    # Extend rightward to current year
    xs.append(extend_to)
    ys.append(ys[-1])
    return xs, ys


def _auto_select_families(qdf: pd.DataFrame, n: int = 6) -> list[str]:
    """Pick the n most interesting families for the timeline plot.

    Prefers families with:
    1. Both quantum and classical algorithms exist (compelling comparison).
    2. The most distinct quantum improvement events.
    Priority list is checked first; remaining slots filled by improvement count.
    """
    # Families known to tell a compelling quantum-vs-classical story
    priority = [
        "Integer Factoring",
        "Multiplication",
        "Sorting",
        "Linear System",
        "Maximum Cardinality Matching",
        "Shortest-Path (Directed Graphs)",
    ]
    counts: dict[str, int] = {}
    for fam, grp in qdf.dropna(subset=["time_class", "year"]).groupby("family"):
        evts = _improvements_for(grp, str(fam))
        counts[str(fam)] = max(len(evts) - 1, 0)

    selected = [f for f in priority if f in counts]
    ranked = sorted(counts, key=lambda f: counts[f], reverse=True)
    for f in ranked:
        if f not in selected:
            selected.append(f)
        if len(selected) >= n:
            break
    return selected[:n]


def plot_family_improvement_timeline(qdf, serial, parallel,
                                     families: list[str] | None = None):
    """Per-family step-function of best known time-class over the years.

    Mirrors the style of Figure 3 in andrewlucasgs/approximation-algorithms:
    each horizontal segment = current best; each drop = a new improvement.
    Quantum steps are annotated with algorithm names.
    """
    if families is None:
        families = _auto_select_families(qdf)

    all_dfs = {
        "quantum":  qdf,
        "serial":   serial,
        "parallel": parallel,
    }

    n = len(families)
    fig, axes = plt.subplots(n, 1,
                             figsize=(11, 4.2 * n),
                             dpi=DPI,
                             constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, family in zip(axes, families):
        any_data = False
        for cat, style in _CAT_STYLE.items():
            df_cat = all_dfs[cat]
            # For serial/parallel, family name matching may need fuzzy lookup
            # (the serial sheet sometimes uses slightly different titles)
            mask = df_cat["family"].str.lower() == family.lower()
            if not mask.any():
                # Try contains match on the first word of family name
                first_word = family.split()[0]
                mask = df_cat["family"].str.lower().str.contains(first_word.lower(), na=False)

            sub = df_cat[mask].copy()
            if sub.empty:
                continue

            evts = _improvements_for(sub, sub["family"].iloc[0])
            if not evts:
                continue
            any_data = True

            xs, ys = _step_xy(evts, extend_to=2026)
            ax.plot(xs, ys, **style)

            # Dot at each improvement point
            ax.scatter([e[0] for e in evts], [e[1] for e in evts],
                       color=style["color"], s=40, zorder=style["zorder"] + 1,
                       edgecolors="white", linewidths=0.6)

            # Annotate quantum improvement steps with algorithm names
            if cat == "quantum":
                # Sort events by time_class descending to assign text offsets
                # without overlapping: alternate above/below each point
                for idx, (yr, tc, alg) in enumerate(evts):
                    if not alg:
                        continue
                    # Alternate text above vs. below the marker to reduce overlap
                    y_sign = 1 if idx % 2 == 0 else -1
                    y_off = y_sign * 14
                    ax.annotate(
                        alg,
                        xy=(yr, tc),
                        xytext=(8, y_off),
                        textcoords="offset points",
                        fontsize=7.5,
                        color=style["color"],
                        alpha=0.92,
                        arrowprops=dict(
                            arrowstyle="->",
                            color=style["color"],
                            lw=0.8,
                            alpha=0.6,
                        ),
                        bbox=dict(
                            boxstyle="round,pad=0.18",
                            fc="white",
                            ec="none",
                            alpha=0.75,
                        ),
                    )

        # Y-axis: time_class 1–8, inverted so lower class (faster) is at top,
        # matching Figure 3 orientation where "up = better"
        all_tc = []
        for cat, df_cat in all_dfs.items():
            mask = df_cat["family"].str.lower() == family.lower()
            if not mask.any():
                first_word = family.split()[0]
                mask = df_cat["family"].str.lower().str.contains(first_word.lower(), na=False)
            vals = df_cat[mask]["time_class"].dropna().tolist()
            all_tc.extend(vals)

        if all_tc:
            lo = max(1, int(min(all_tc)) - 1)
            hi = min(8, int(max(all_tc)) + 1)
        else:
            lo, hi = 1, 8

        ticks = [t for t in range(lo, hi + 1) if t in CLASS_LABELS]
        ax.set_yticks(ticks)
        ax.set_yticklabels(
            [f"{t}  ({CLASS_LABELS[t]})" for t in ticks], fontsize=8
        )
        # Invert so that improvements (lower class = faster) appear as upward steps
        ax.invert_yaxis()

        ax.set_xlim(left=min(
            (qdf[qdf["family"].str.lower() == family.lower()]["year"].min()
             if (qdf["family"].str.lower() == family.lower()).any() else 1990),
            1990,
        ) - 2, right=2027)
        ax.set_xlabel("Year", fontsize=9)
        ax.set_ylabel("Best time-complexity class\n(higher on axis = faster)",
                      fontsize=8.5, labelpad=8)
        ax.set_title(f"Fastest known algorithms over time — {family}",
                     fontsize=11, pad=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.15, linestyle="--")
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    fig.suptitle(
        "Algorithm improvement trajectories by problem family\n"
        "— each step marks a new best-known time complexity —",
        fontsize=12,
        fontweight="bold",
    )
    save(fig, "6_family_improvement_timeline")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading workbook: {XLSX.name}")
    xls = pd.ExcelFile(XLSX)

    print("  parsing sheets...")
    qdf    = load_quantum(xls)
    serial = load_serial(xls)
    para   = load_parallel(xls)

    print(f"  quantum  : {len(qdf):>4} rows, {qdf['family'].nunique()} families")
    print(f"  serial   : {len(serial):>4} rows, {serial['family'].nunique()} families")
    print(f"  parallel : {len(para):>4} rows, {para['family'].nunique()} families")

    print(f"\nSaving plots to {OUT_DIR.relative_to(REPO_ROOT)} …\n")

    print("[1/6] Best time-class per family (quantum vs serial vs parallel)")
    plot_best_by_family(qdf, serial, para)

    print("[2/6] Runtime distribution for top quantum families")
    plot_runtime_distribution(qdf, serial, para)

    print("[3/6] Histogram – best serial vs best quantum")
    plot_best_histogram(qdf, serial, para)

    print("[4/6] Cumulative improvement events over time")
    plot_improvement_over_time(qdf, serial, para)

    print("[5/6] Time–space tradeoff (quantum)")
    plot_time_space_tradeoff(qdf)

    print("[6/6] Per-family algorithm improvement timeline")
    plot_family_improvement_timeline(qdf, serial, para)

    print("\nDone.")


if __name__ == "__main__":
    main()
