"""
Jun 2026 -- Part 2: Rigorously-matched, single-problem comparisons.

Unlike plots_jun2026.py (which compares whole *families*, e.g. all
"Integer Factoring" sub-variations lumped together), this script only
compares algorithms that solve the *exact same problem* with a formula
that is a *pure function of the same single governing parameter*.

Why this needed a rewrite
--------------------------
An earlier version trusted AlgoWiki's own "Time Complexity Class" field
for every row. That produced a nonsensical result for Subset Sum: the
best *parallel* algorithm looked faster than the best *quantum* one.
Digging into the raw formulas showed why:

  * Parallel rows report *span/T_p* -- wall-clock time when spread
    across processors, e.g. "O(2^(n/2) / p)" or (implicitly, with no
    "p" ever written down) "O(log^2 n)" for an algorithm whose total
    work is really O(n^9 log^2 n). Either way this assumes access to as
    many processors as the algorithm can use -- not comparable to a
    single quantum computer or a single CPU. It's exactly the
    "O(k) vs O(n^2) with k != n" mismatch the parallel-processor count
    introduces implicitly. The *fair* single-processor-equivalent
    number is the algorithm's WORK (total operations) column, which is
    what this script uses for every parallel row whenever it's
    available and itself a pure function of the canonical parameter.
  * At least one row has a plain, processor-free formula
    ("O(2^(3n/8))") that is nonetheless miscoded in the source
    spreadsheet as class "4" (should be class 8, exponential -- the
    identical formula is coded "8" for a quantum row). This is a
    spreadsheet data-entry error, not a modeling issue, and it will
    silently corrupt any plot that trusts the field.

So instead of trusting the field, this script:
  1. Extracts every variable *token* actually used in each formula
     (not just the "Preferred Parameter" the curator picked).
  2. For parallel rows, always tries the "Parallel Algorithm Work"
     formula first (total operations, the fair metric); only falls
     back to the "Time Complexity" (span) formula if Work is missing.
  3. Keeps a row only if the formula actually used reduces to the
     problem's single canonical parameter (e.g. just "n", or just "V")
     -- any row with an extra free symbol (p, t, k, sigma(...), c,
     w_min, h, eps, ...) is dropped. This also catches genuinely
     different problem parameterizations hiding under the same family,
     e.g. classic Vertex Cover FPT algorithms are O(f(k)*n) in the
     cover size k, not simply O(g(n)) -- not comparable to an O(n)-only
     formula without an extra assumption this script refuses to make.
  4. Every surviving formula is re-classified from scratch with a small
     auditable regex classifier (exponential / n^k / log factors), so
     the final numbers never depend on a possibly-mistyped category
     column.
  5. Two of the original six candidate problems (Vertex Cover, Convex
     Hull 2D) turned out to have *zero* quantum rows survive this
     filtering -- the only cataloged quantum entries for those problems
     depend on k / h, not n -- so they were dropped and replaced with
     Matrix Chain Multiplication and DFA Minimization, which do survive
     with quantum representation intact.

Additional formatting per user request (Jul 2026):
  * All plots restricted to years >= 1950.
  * Y-axis gridlines standardized to steps of 0.5 across all panels.
  * Each panel is annotated with its governing parameter + per-category
    sample size.

Output: Plots/jun2026/6_*.png, 7_*.png, 8_*.png
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent
XLSX = REPO_ROOT / "data" / "AlgoWiki algorithms (our copy) (3).xlsx"
OUT_DIR = REPO_ROOT / "Plots" / "jun2026"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_QUANTUM  = "#4C72B0"
C_SERIAL   = "#DD8452"
C_PARALLEL = "#55A868"
CAT_COLOR = {"quantum": C_QUANTUM, "serial": C_SERIAL, "parallel": C_PARALLEL}
CAT_LABEL = {"quantum": "Quantum", "serial": "Serial (classical)", "parallel": "Parallel (classical)"}
DPI = 200
MIN_YEAR = 1950

CLASS_LABELS = {
    1: "O(1)", 2: "O(log n)", 3: "O(n)", 4: "O(n log n)",
    5: "O(n\u00b2)", 6: "O(n\u00b3)", 7: "O(n\u2074)", 8: "O(2\u207f)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Load raw sheets
# ─────────────────────────────────────────────────────────────────────────────

def _norm(s: pd.Series) -> pd.Series:
    return (s.fillna("").astype(str)
              .str.replace(r"\s+", " ", regex=True)
              .str.strip().str.title().replace({"": None}))


def _load_sheet(xls: pd.ExcelFile, sheet: str, category: str,
                row_filter=None) -> pd.DataFrame:
    raw = xls.parse(sheet)
    colmap = {
        "Family Name": "family", "Variation": "variation", "Year": "year",
        "Time Complexity Class": "raw_time_class",
        "Preferred Parameter": "pref_param",
        "Parameter definitions": "param_defs",
        "Parallel Algorithm Work": "work_expr",
    }
    tc_col = ("Time Complexity / Circuit Depth (Worst Only)"
              if sheet == "Quantum Algorithms" else "Time Complexity (Worst Only)")
    colmap[tc_col] = "time_expr"

    keep = [c for c in colmap if c in raw.columns]
    df = raw[keep].copy()
    df.columns = [colmap[c] for c in keep]
    if "work_expr" not in df.columns:
        df["work_expr"] = None
    if row_filter is not None:
        df = df[row_filter(raw)]

    df["family"] = _norm(df["family"])
    df["variation"] = _norm(df["variation"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["family", "variation"]).copy()
    df["category"] = category
    return df


def load_all() -> pd.DataFrame:
    xls = pd.ExcelFile(XLSX)
    qdf = _load_sheet(xls, "Quantum Algorithms", "quantum")

    def serial_filter(raw):
        return ((pd.to_numeric(raw["Parallel?"], errors="coerce").fillna(0) == 0)
                & (pd.to_numeric(raw["Quantum?"], errors="coerce").fillna(0) != 1))
    sdf = _load_sheet(xls, "Sheet1", "serial", serial_filter)
    pdf = _load_sheet(xls, "Parallel Algos", "parallel")

    return pd.concat([qdf, sdf, pdf], ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Hand-verified problem definitions
# ─────────────────────────────────────────────────────────────────────────────
PROBLEMS: dict[str, dict] = {
    "Integer Factoring": {
        "keys": [
            ("Integer Factoring", "Integer Factoring"),
            ("Integer Factoring", "First Category Integer Factoring"),
            ("Integer Factoring", "Second Category Integer Factoring"),
        ],
        "param": "n", "param_note": "n = number of bits encoding the input integer",
    },
    "Subset Sum": {
        "keys": [("The Subset-Sum Problem", "Subset Sum")],
        "param": "n", "param_note": "n = number of elements in the input set",
    },
    "Discrete Fourier Transform": {
        "keys": [("Discrete Fourier Transform", "Discrete Fourier Transform")],
        "param": "n", "param_note": "n = length of the input sequence",
    },
    "st-Maximum Flow": {
        "keys": [("Maximum Flow", "St-Maximum Flow")],
        "param": "v", "param_note": "V = number of vertices (formulas that also need E free are excluded)",
    },
    "Matrix Chain Multiplication": {
        "keys": [
            ("Matrix Chain Multiplication", "Matrix Chain Ordering Problem"),
            ("Matrix Chain Multiplication", "Mcop"),
        ],
        "param": "n", "param_note": "n = number of matrices in the chain",
    },
    "DFA Minimization": {
        "keys": [("Dfa Minimization", "Cyclic Nontrivial Sccs Dfa Minimization")],
        "param": "n", "param_note": "n = number of states in the DFA",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Formula parsing: token extraction + from-scratch complexity classification
# ─────────────────────────────────────────────────────────────────────────────
_STOPWORDS = {
    "o", "log", "ln", "exp", "min", "max", "floor", "ceil", "poly", "tilde",
    "theta", "omega", "sqrt", "big", "mathcal", "under", "assumption",
    "about", "numbers", "in", "a", "sequence", "behaving", "randomly",
    "range", "time", "using", "processors", "more", "specifically", "if",
    "and", "of", "for", "the", "expected", "where", "is", "constant", "no",
}
_FUNC_BASES = {"log", "ln", "exp"}


def _mark_euler_e(expr: str) -> str:
    """Euler's constant is often written as a bare 'e' used as the base of
    an exponential, e.g. "e^{sqrt(n log n)}". That's a math constant, not
    a free parameter -- but a bare 'E' elsewhere (e.g. graph edge count,
    "O(V^2*E)") is a genuine extra parameter and must NOT be conflated with
    it. Only the "e" immediately followed by "^" is Euler's constant."""
    return re.sub(r"(?<![A-Za-z_])e(?=\s*\^)", "EULERCONST", expr, flags=re.IGNORECASE)


def _clean_expr(expr) -> str | None:
    if not isinstance(expr, str):
        return None
    e = expr.strip()
    if e in ("", "-", "nan", "#VALUE!"):
        return None
    e = e.replace("$", "")
    # curators sometimes write "Name[params] = actual formula" (e.g. the
    # standard number-theoretic L-notation "L_n[1/3,c] = exp(...)") -- the
    # real formula is whatever follows the LAST "=", so use that.
    if "=" in e:
        e = e.rsplit("=", 1)[1]
    # drop trailing prose after the first comma/semicolon (footnote-style caveats)
    e = re.split(r",\s*(?:under|where|more)", e)[0]
    return e.strip() or None


def _extract_tokens(expr: str) -> set[str]:
    """Return the set of variable-like symbols used in a formula, minus
    known function names / stopwords / bare numbers."""
    # normalize common LaTeX-ish forms so tokens merge sanely -- pad with
    # spaces so e.g. "n\log n" doesn't collapse into the single bogus
    # token "nlog" once the backslash is stripped.
    e = _mark_euler_e(expr)
    for fn in ("log", "sqrt", "exp", "min", "max", "floor", "ceil"):
        e = re.sub(r"\\" + fn, f" {fn} ", e)
    e = e.replace("\\", " ")
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z_]*", e)
    tokens = set()
    for t in raw_tokens:
        tl = t.lower()
        if tl in _STOPWORDS or tl == "eulerconst":
            continue
        tokens.add(tl.rstrip("_"))
    return tokens


def _base_and_exponents(e: str) -> list[tuple[str, str]]:
    """Scan `e` for every 'BASE^EXPONENT' occurrence and return (base,
    exponent) string pairs, properly bounding each side at balanced
    parens (or, if unparenthesized, at the nearest non-alphanumeric
    boundary) rather than greedily scanning across unrelated terms."""
    pairs = []
    i = 0
    while True:
        idx = e.find("^", i)
        if idx == -1:
            break
        # ---- base: token (or balanced-paren group) immediately before ^
        b = idx - 1
        while b >= 0 and e[b] == " ":
            b -= 1
        end_base = b + 1
        if b >= 0 and e[b] == ")":
            depth = 0
            while b >= 0:
                if e[b] == ")":
                    depth += 1
                elif e[b] == "(":
                    depth -= 1
                    if depth == 0:
                        break
                b -= 1
            base = e[b:end_base]
        else:
            sb = b
            while sb >= 0 and re.match(r"[A-Za-z0-9._]", e[sb]):
                sb -= 1
            base = e[sb + 1:end_base]

        # ---- exponent: balanced-paren group, or bare alnum token
        j = idx + 1
        while j < len(e) and e[j] == " ":
            j += 1
        if j < len(e) and e[j] == "(":
            depth, k, start = 0, j, j + 1
            while k < len(e):
                if e[k] == "(":
                    depth += 1
                elif e[k] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            exponent = e[start:k]
            i = k + 1
        else:
            m = re.match(r"[A-Za-z0-9.]*", e[j:])
            exponent = m.group(0) if m else ""
            i = j + max(len(exponent), 1)

        pairs.append((base.strip(), exponent.strip()))
    return pairs


def classify_expression(expr: str) -> float | None:
    """From-scratch complexity-class classifier (1..8 scale, matching the
    AlgoWiki convention: 1=O(1), 2=O(log n)/polylog, 3=O(n), 4=O(n log n),
    5=O(n^2), 6=O(n^3), 7=O(n^4), 8=exponential or worse).
    Only intended for formulas that are already known to be a pure
    function of a single parameter (post token-filtering).
    """
    e = _mark_euler_e(expr)
    for fn in ("log", "sqrt", "exp", "min", "max", "floor", "ceil"):
        e = re.sub(r"\\" + fn, f" {fn} ", e)
    e = e.replace("\\", " ").replace("$", "").lower()
    # LaTeX uses "^{...}" for exponents exactly like "^(...)" -- normalize
    # so the paren-based exponent scanner below catches both forms.
    e = e.replace("{", "(").replace("}", ")")
    prev = None
    while prev != e:
        prev = e
        e = e.replace("((", "(").replace("))", ")")

    pairs = _base_and_exponents(e)

    is_exponential = bool(re.search(r"exp\s*\(", e)) or bool(re.search(r"[a-z]\s*!", e))
    degree = None
    for base, exponent in pairs:
        if any(fn in base for fn in _FUNC_BASES):
            # a power applied to log/ln/exp (e.g. "log^4 n") is a log
            # factor, not a polynomial-degree or exponential signal.
            continue
        if re.search(r"[a-z]", exponent):
            # variable in the exponent (e.g. "2^n", "eulerconst^(sqrt(n))")
            is_exponential = True
        else:
            try:
                deg = eval(exponent) if "/" in exponent else float(exponent)
            except Exception:
                continue
            degree = deg if degree is None else max(degree, deg)

    if is_exponential:
        return 8.0

    has_log = "log" in e or "ln" in e
    if degree is None:
        has_var = re.search(r"[a-z]", e) is not None
        degree = 1.0 if has_var and not e.strip().startswith("log") else 0.0
        if e.strip() in ("1",):
            degree = 0.0

    if degree <= 0:
        return 2.0 if has_log else 1.0
    if degree <= 1:
        return 4.0 if has_log else 3.0
    if degree <= 2:
        return 5.0
    if degree <= 3:
        return 6.0
    if degree <= 4:
        return 7.0
    return 8.0


def build_verified_dataset(all_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    audit = []
    rows_out = []

    for problem, spec in PROBLEMS.items():
        canon = spec["param"].lower()
        mask = pd.Series(False, index=all_df.index)
        for fam, var in spec["keys"]:
            mask |= (all_df["family"] == fam) & (all_df["variation"] == var)
        grp = all_df[mask].copy()
        grp = grp[grp["year"].isna() | (grp["year"] >= MIN_YEAR)]

        kept_rows = []
        excluded = []
        for _, r in grp.iterrows():
            texpr = _clean_expr(r.get("time_expr"))
            wexpr = _clean_expr(r.get("work_expr")) if r["category"] == "parallel" else None

            used_expr, used_via = None, None

            # Parallel rows: total Work is the fair, single-processor-
            # equivalent metric -- always prefer it over Time/span when
            # it's itself a pure function of the canonical parameter.
            if wexpr is not None:
                wtok = _extract_tokens(wexpr) - {canon}
                if not wtok:
                    used_expr, used_via = wexpr, "work"

            if used_expr is None and texpr is not None:
                ttok = _extract_tokens(texpr) - {canon}
                if not ttok:
                    used_expr, used_via = texpr, "time"

            if used_expr is None:
                if texpr is None and wexpr is None:
                    excluded.append((r, "no formula recorded"))
                else:
                    reasons = []
                    if texpr is not None:
                        reasons.append(f"time depends on {sorted(_extract_tokens(texpr) - {canon})}")
                    if wexpr is not None:
                        reasons.append(f"work depends on {sorted(_extract_tokens(wexpr) - {canon})}")
                    excluded.append((r, f"formula(s) depend on extra symbol(s) beyond '{canon}': " + "; ".join(reasons)))
                continue

            cls = classify_expression(used_expr)
            if cls is None:
                excluded.append((r, "could not classify formula"))
                continue

            kept_rows.append({
                "problem": problem, "family": r["family"], "variation": r["variation"],
                "category": r["category"], "year": r["year"],
                "time_class": cls, "used_expr": used_expr, "used_via": used_via,
                "orig_expr": texpr, "raw_time_class": r.get("raw_time_class"),
            })

        kept_df = pd.DataFrame(kept_rows)
        rows_out.append(kept_df)

        cat_counts = (kept_df["category"].value_counts().to_dict()
                     if not kept_df.empty else {})
        n_work = int((kept_df["used_via"] == "work").sum()) if not kept_df.empty else 0
        audit.append({
            "problem": problem, "param": canon,
            "kept": len(kept_df), "dropped": len(excluded),
            "kept_by_category": {c: cat_counts.get(c, 0) for c in ["quantum", "serial", "parallel"]},
            "rescued": n_work,
        })
        for r, reason in excluded:
            audit.append({
                "problem": problem, "excluded_detail": True,
                "category": r["category"], "reason": reason,
                "expr": r.get("time_expr"), "year": r.get("year"),
            })

    long_df = pd.concat(rows_out, ignore_index=True) if rows_out else pd.DataFrame()
    return long_df, audit


def print_audit(audit: list[dict]) -> None:
    print("\nParameter-consistency + re-classification audit")
    print("-" * 78)
    for a in audit:
        if "kept_by_category" in a:
            c = a["kept_by_category"]
            print(f"  {a['problem']:<28} param='{a['param']}'  kept={a['kept']:<3} "
                  f"dropped={a['dropped']:<3} parallel_rows_using_work={a['rescued']}")
            print(f"    {'':28} quantum={c['quantum']}  serial={c['serial']}  parallel={c['parallel']}")
    print("\n  Exclusions (extra / unresolvable parameters, or bad data):")
    for a in audit:
        if a.get("excluded_detail"):
            print(f"    - [{a['problem']}] {a['category']} ({a.get('year')}): {a['reason']}  "
                  f"(expr: {a.get('expr')})")
    print("-" * 78)


def save(fig, name: str) -> Path:
    p = OUT_DIR / f"{name}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved -> {p.relative_to(REPO_ROOT)}")
    return p


def half_step_ticks(lo: float, hi: float) -> list[float]:
    lo2 = np.floor(lo * 2) / 2
    hi2 = np.ceil(hi * 2) / 2
    return list(np.arange(lo2, hi2 + 0.5, 0.5))


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 — best-known algorithm per category, per verified problem
# ─────────────────────────────────────────────────────────────────────────────

def plot_best_by_problem(long_df: pd.DataFrame) -> None:
    problems = list(PROBLEMS.keys())
    best = (long_df.dropna(subset=["time_class"])
                    .groupby(["problem", "category"])["time_class"].min()
                    .unstack("category"))
    best = best.reindex(problems)

    cats = [c for c in ["quantum", "serial", "parallel"] if c in best.columns]
    n = len(problems)
    y_pos = np.arange(n)
    bar_h = 0.8 / len(cats)

    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=DPI)
    for i, cat in enumerate(cats):
        offset = (i - (len(cats) - 1) / 2) * bar_h
        vals = best[cat].values
        mask = ~np.isnan(vals)
        ax.barh(y_pos[mask] + offset, vals[mask], height=bar_h * 0.92,
                color=CAT_COLOR[cat], alpha=0.88, label=CAT_LABEL[cat])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(problems, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Best known time-complexity class (lower = faster)", fontsize=10)
    ax.set_title(
        "Best known algorithm by category \u2014 re-classified from raw formulas\n"
        "(same problem, same single parameter; parallel rows use Work, not T_p, where needed)",
        fontsize=10.8, pad=10
    )
    valid_vals = best.values[~np.isnan(best.values)]
    max_x = int(np.ceil(valid_vals.max())) if len(valid_vals) else 8
    ax.set_xlim(0, max_x + 1.6)
    ax.set_xticks(range(1, max_x + 1))
    ax.set_xticklabels([f"{i}\n({CLASS_LABELS.get(i,'')})" for i in range(1, max_x + 1)],
                       fontsize=8)
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.2, linestyle="--")

    counts = (long_df.dropna(subset=["time_class"])
                     .groupby(["problem", "category"]).size().unstack(fill_value=0)
                     .reindex(problems, fill_value=0))
    for i, problem in enumerate(problems):
        note = PROBLEMS[problem]["param_note"]
        n_str = ", ".join(f"{c[:1].upper()}{c[1:3]}={int(counts.loc[problem, c])}"
                          for c in cats if c in counts.columns)
        ax.text(max_x + 0.15, i, f"{note}\n(n: {n_str})", fontsize=6.3, va="center", color="gray")

    fig.text(0.01, -0.06,
             "Note: a category's bar is omitted for a problem if no surviving formula in that\n"
             "category reduces to a pure function of the stated parameter alone.",
             fontsize=7.5, color="gray", style="italic")

    fig.tight_layout()
    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=3,
              frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.11))
    save(fig, "6_verified_best_by_problem")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 7 — runtime distribution per verified problem, faceted by category
# ─────────────────────────────────────────────────────────────────────────────

def plot_distribution_by_problem(long_df: pd.DataFrame) -> None:
    problems = list(PROBLEMS.keys())
    ncols = 3
    nrows = int(np.ceil(len(problems) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 4.3), dpi=DPI)
    axes = np.array(axes).flatten()

    for idx, problem in enumerate(problems):
        ax = axes[idx]
        sub = long_df[long_df["problem"] == problem]
        cats = [c for c in ["quantum", "serial", "parallel"] if (sub["category"] == c).any()]

        data, positions, colors, labels = [], [], [], []
        for i, cat in enumerate(cats):
            vals = sub[sub["category"] == cat]["time_class"].dropna().values
            if len(vals) == 0:
                continue
            data.append(vals)
            positions.append(i + 1)
            colors.append(CAT_COLOR[cat])
            labels.append(f"{CAT_LABEL[cat]}\n(n={len(vals)})")

        if data:
            bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True,
                            medianprops=dict(color="black", linewidth=2),
                            whiskerprops=dict(linewidth=1.1), capprops=dict(linewidth=1.1),
                            flierprops=dict(marker="o", markersize=4, alpha=0.6))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color); patch.set_alpha(0.85)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(f"{problem}\n({PROBLEMS[problem]['param_note']})", fontsize=7.6, pad=4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        all_vals = np.concatenate(data) if data else np.array([3])
        ticks = half_step_ticks(all_vals.min(), all_vals.max())
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:g}" for t in ticks], fontsize=6.5)
        ax.grid(True, axis="y", alpha=0.2, linestyle="--")

    for ax in axes[len(problems):]:
        ax.set_visible(False)

    fig.text(0.005, 0.5, "Time complexity class (lower = faster)",
             va="center", rotation="vertical", fontsize=9)
    fig.suptitle(
        "Runtime distribution for verified single-problem comparisons\n"
        "(re-classified from raw formulas; parallel rescued via Work where the reported time was T_p)",
        fontsize=11.5, y=1.03
    )
    fig.tight_layout(rect=[0.02, 0, 1, 0.95])
    save(fig, "7_verified_runtime_distribution")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 8 — chronological improvement timeline, per verified problem
# ─────────────────────────────────────────────────────────────────────────────

def plot_timeline_by_problem(long_df: pd.DataFrame) -> None:
    problems = list(PROBLEMS.keys())
    ncols = 3
    nrows = int(np.ceil(len(problems) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 4.0), dpi=DPI)
    axes = np.array(axes).flatten()

    for idx, problem in enumerate(problems):
        ax = axes[idx]
        sub = long_df[(long_df["problem"] == problem)].dropna(subset=["year", "time_class"])
        sub = sub[sub["year"] >= MIN_YEAR].sort_values("year")

        for cat in ["serial", "parallel", "quantum"]:
            csub = sub[sub["category"] == cat]
            if csub.empty:
                continue
            marker = "D" if cat == "quantum" else ("o" if cat == "serial" else "s")
            ax.scatter(csub["year"], csub["time_class"], color=CAT_COLOR[cat],
                      label=CAT_LABEL[cat], s=48, alpha=0.85,
                      edgecolors="white", linewidths=0.6, marker=marker, zorder=3)
            running = csub.sort_values("year")
            best_so_far = running["time_class"].cummin()
            ax.plot(running["year"], best_so_far, color=CAT_COLOR[cat],
                    linewidth=1.4, alpha=0.55, drawstyle="steps-post", zorder=2)

        param_note = PROBLEMS[problem]["param_note"]
        cat_counts = sub["category"].value_counts().to_dict()
        n_str = ", ".join(f"{c[:1].upper()}{c[1:3]}={cat_counts.get(c,0)}"
                          for c in ["quantum", "serial", "parallel"])
        ax.set_title(f"{problem}\n{param_note}\n(n: {n_str})", fontsize=7.3, pad=4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.set_xlim(MIN_YEAR, 2027)
        ax.set_xlabel("Year", fontsize=7.5)
        if idx % ncols == 0:
            ax.set_ylabel("Time class", fontsize=7.5)

        if not sub.empty:
            ticks = half_step_ticks(sub["time_class"].min(), sub["time_class"].max())
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{t:g}" for t in ticks], fontsize=6.5)

    for ax in axes[len(problems):]:
        ax.set_visible(False)

    handles = [mpatches.Patch(facecolor=CAT_COLOR[c], alpha=0.85, label=CAT_LABEL[c])
               for c in ["quantum", "serial", "parallel"]]
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.suptitle(
        "Algorithmic progress over time \u2014 verified single-problem comparisons (1950\u2013present)\n"
        "(lines trace the running-best time-complexity class within each category; "
        "parallel points use Work when the raw time was processor-normalized)",
        fontsize=11.5, y=0.99
    )
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
              fontsize=10, bbox_to_anchor=(0.5, 0.925))
    save(fig, "8_verified_timeline")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading workbook: {XLSX.name}")
    all_df = load_all()
    print(f"  total rows across 3 sheets: {len(all_df)}")

    long_df, audit = build_verified_dataset(all_df)
    print_audit(audit)

    print(f"\nSaving verified-problem plots to {OUT_DIR.relative_to(REPO_ROOT)} ...\n")
    print("[1/3] Best known algorithm per category, per problem")
    plot_best_by_problem(long_df)
    print("[2/3] Runtime distribution per problem")
    plot_distribution_by_problem(long_df)
    print("[3/3] Chronological timeline per problem")
    plot_timeline_by_problem(long_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
