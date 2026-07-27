"""Check the parsed runtime dataset before anything is plotted from it.

Three independent checks:

1. **Known answers.** A hand-written table of formulas whose value at a given
   problem size can be worked out on paper, plus published operation-count
   estimates for the factoring algorithms.
2. **Agreement with the curators.** The workbook carries a "Param: Time
   Class" column -- an ordinal severity score the curators assigned to each
   parameter by eye, never read by :mod:`src.dataset`. If the parser is
   reading the formulas correctly, ranking a family's algorithms by parsed
   runtime should reproduce ranking them by that score.
3. **Parameter consistency.** Every row inside a family must be a function of
   the same quantity, and no family may mix, say, vertices with edges.

Run: ``python validate_dataset.py``
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.advantage import problems_with, runtime_table  # noqa: E402
from src.dataset import build_dataset, write_audit  # noqa: E402
from src.formula import parse_formula  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
AUDIT_PATH = REPO_ROOT / "Plots" / "jun2026" / "runtime_dataset_audit.csv"

# (cell text, symbol, size, expected log10, tolerance, what it checks)
KNOWN_ANSWERS = [
    (r"$O(n)$", "n", 1e6, 6.0, 0.0, "linear"),
    (r"$O(n^2)$", "n", 1e6, 12.0, 0.0, "quadratic"),
    (r"$O(n \log n)$", "n", 1e6, 7.2999, 0.001, "n log2 n = 1.99e7"),
    (r"$O(n \log^2 n)$", "n", 1e6, 8.599, 0.01, "n (log2 n)^2"),
    (r"$O(\sqrt{n})$", "n", 1e6, 3.0, 0.0, "Grover"),
    (r"$O(2^{n/2})$", "n", 100, 15.051, 0.001, "meet in the middle, 2^50"),
    (r"$\tilde{O}(n^{2.5})$", "n", 1e4, 10.0, 0.0, "soft-O reads its argument"),
    (r"$O(n^{2.373})$", "n", 1e4, 9.492, 0.001, "matrix multiplication"),
    (r"$O(E \log V)$", "V", 1e3, 6.697, 0.01, "E->V(V-1)/2 worst case"),
    (r"O(n^5/2 + m)", "n", 1e3, 7.507, 0.01, "bare-integer divisor is an exponent"),
    (r"$O(n^3/2^{\sqrt{\log n}})$", "n", 1e3, 8.052, 0.01, "a real division, left alone"),
    (r"$O(n \log^* n)$", "n", 1e6, 6.699, 0.01, "log* 1e6 = 5"),
    (r"$O(3^n)$", "n", 20, 9.542, 0.001, "3^20"),
    (r"$O(\log^2 n)$", "n", 1e6, 2.599, 0.01, "(log2 1e6)^2 = 397"),
    (r"$O(\exp((1+o(1))(64n/9)^{1/3}(\log n)^{2/3})$", "n", 1024, 30.6, 0.1,
     "number field sieve, natural logs under the exponential"),
]

# Published operation-count estimates for factoring a 1024-bit modulus.
# Ranges are wide on purpose: these are order-of-magnitude sanity bounds, not
# targets to tune against.
FACTORING_BOUNDS = [
    ("General number field sieve", 1e26, 1e33),
    ("Quadratic sieve", 1e29, 1e38),
    ("Shor's Algorithm", 1e6, 1e11),
]


def check_known_answers() -> int:
    print("1. FORMULAS WITH KNOWN ANSWERS")
    failures = 0
    for raw, symbol, size, expected, tol, note in KNOWN_ANSWERS:
        try:
            parsed = parse_formula(raw)
            expr = parsed.expr
            if parsed.symbols - {symbol}:
                # Worst-case reduction lives in src.dataset; redo it here for
                # the multi-parameter cases in the table.
                import sympy
                v = sympy.Symbol(symbol)
                for edges in ("E", "m"):
                    expr = expr.subs(sympy.Symbol(edges), v * (v - 1) / 2)
            from src.formula import log10_runtime
            got = log10_runtime(expr, symbol, size)
        except Exception as exc:                       # noqa: BLE001
            print(f"   FAIL  {raw:<26} -> {type(exc).__name__}: {exc}")
            failures += 1
            continue
        ok = abs(got - expected) <= max(tol, 1e-9)
        print(f"   {'ok  ' if ok else 'FAIL'}  {raw:<26} log10={got:>8.3f} "
              f"expected {expected:>8.3f}   {note}")
        failures += not ok
    print(f"   -> {len(KNOWN_ANSWERS) - failures}/{len(KNOWN_ANSWERS)} passed\n")
    return failures


def check_factoring(ds) -> int:
    print("2. FACTORING A 1024-BIT MODULUS vs PUBLISHED ESTIMATES")
    rows = {r.name: r for r in ds.rows if r.family == "integer factoring"}
    failures = 0
    for name, low, high in FACTORING_BOUNDS:
        row = next((r for n, r in rows.items() if n and n.startswith(name)), None)
        if row is None:
            print(f"   FAIL  {name}: not in the retained dataset")
            failures += 1
            continue
        ops = row.log10_at(1024)
        ok = math_in(ops, low, high)
        import math
        print(f"   {'ok  ' if ok else 'FAIL'}  {name:<30} 1e{ops:<7.2f} "
              f"expected 1e{math.log10(low):.0f}-1e{math.log10(high):.0f}")
        failures += not ok
    print()
    return failures


def math_in(log10_value: float, low: float, high: float) -> bool:
    import math
    return math.log10(low) <= log10_value <= math.log10(high)


def check_curator_agreement(ds) -> int:
    print("3. AGREEMENT WITH THE WORKBOOK'S OWN COMPLEXITY CLASS")
    df = ds.frame()
    df["log10_at_1e6"] = [r.log10_at(1e6) for r in ds.rows]
    # Only serial and quantum rows can be checked this way. On the parallel
    # sheet "Param: Time Class" scores the algorithm's *span*, while the
    # column being parsed is its total work -- Valiant-Skyum-Berkowitz is
    # class 2 there (log^2 n time) against n^9 log^2 n operations.
    scored = df[df["curator_class"].notna() & df["category"].isin(["serial", "quantum"])].copy()
    eligible = df[df["category"].isin(["serial", "quantum"])]
    print(f"   {len(scored)} of {len(eligible)} serial/quantum rows carry a curator class")

    overall = scored["log10_at_1e6"].corr(scored["curator_class"], method="spearman")
    print(f"   rank correlation, all rows            rho = {overall:+.3f}")

    within = []
    for family, group in scored.groupby("family"):
        if group["curator_class"].nunique() < 2 or len(group) < 4:
            continue
        rho = group["log10_at_1e6"].corr(group["curator_class"], method="spearman")
        if pd.notna(rho):
            within.append((family, rho, len(group)))
    if within:
        med = pd.Series([r for _, r, _ in within]).median()
        print(f"   median rank correlation within family rho = {med:+.3f} "
              f"over {len(within)} families")
        worst = sorted(within, key=lambda t: t[1])[:5]
        print("   weakest families:")
        for family, rho, n in worst:
            print(f"      {family[:44]:<46} rho={rho:+.3f}  (n={n})")

    print("\n   largest single disagreements (parsed rank vs curator rank):")
    scored["parsed_rank"] = scored.groupby("family")["log10_at_1e6"].rank(pct=True)
    scored["curator_rank"] = scored.groupby("family")["curator_class"].rank(pct=True)
    scored["gap"] = (scored["parsed_rank"] - scored["curator_rank"]).abs()
    for _, r in scored.nlargest(8, "gap").iterrows():
        print(f"      {r['family_label'][:26]:<28} {str(r['name'])[:24]:<26} "
              f"{r['formula'][:24]:<26} class={r['curator_class']:<5} gap={r['gap']:.2f}")
    print()
    return 0


def check_parameter_consistency(ds) -> int:
    print("4. PARAMETER CONSISTENCY")
    failures = 0
    df = ds.frame()
    mixed = df.groupby("family")["semantic"].nunique()
    bad = mixed[mixed > 1]
    if len(bad):
        print(f"   FAIL  {len(bad)} families mix two quantities: {list(bad.index)}")
        failures += len(bad)
    else:
        print(f"   ok    all {df['family'].nunique()} families measured in one quantity")

    for row in ds.rows:
        free = {str(s) for s in row.expr.free_symbols}
        if free != {row.symbol}:
            print(f"   FAIL  {row.family_label} / {row.name}: leftover symbols {free}")
            failures += 1
    if not failures:
        print(f"   ok    all {len(ds.rows)} retained runtimes are a function of "
              "exactly one variable")

    table = runtime_table(ds, 1e6)
    both = problems_with(table, "quantum", "serial")
    print(f"\n   {len(both)} problems carry both a quantum and a classical serial "
          "entry; the symbol each category uses:")
    for problem in both:
        sub = table[table["problem"] == problem]
        rows = [r for r in ds.rows if problem in
                {f"{r.family}||{v}" for v in r.variations}]
        params = {}
        for r in rows:
            params.setdefault(r.category, set()).add(r.symbol)
        detail = "  ".join(f"{c}:{'/'.join(sorted(v))}"
                           for c, v in sorted(params.items()))
        family = sub["family"].iloc[0]
        print(f"      {sub['problem_label'].iloc[0][:48]:<50} "
              f"[{ds.parameter_label(family)}]  {detail}")
    print()
    return failures


def check_problem_matching(ds) -> int:
    """Report the families that lose their cross-category comparison, and why."""
    print("5. PROBLEM MATCHING ACROSS SHEETS")
    table = runtime_table(ds, 1e6)
    matched = set(problems_with(table, "quantum", "serial"))
    matched_families = {p.split("||")[0] for p in matched}

    has_quantum = set(table[table["category"] == "quantum"]["family"])
    has_serial = set(table[table["category"] == "serial"]["family"])
    unmatched = sorted((has_quantum & has_serial) - matched_families)

    print(f"   {len(matched)} problems matched across the quantum and serial sheets")
    if unmatched:
        print(f"   {len(unmatched)} families have entries on both sheets but no "
              "variation in common, so they are excluded:")
        for family in unmatched:
            sub = table[table["family"] == family]
            for category in ("quantum", "serial"):
                names = sorted(set(sub[sub["category"] == category]["variation"]))
                print(f"      {family[:34]:<36} {category:<8} {names}")
    print()
    return 0


def main() -> int:
    ds = build_dataset()
    print()
    failures = 0
    failures += check_known_answers()
    failures += check_factoring(ds)
    failures += check_curator_agreement(ds)
    failures += check_parameter_consistency(ds)
    failures += check_problem_matching(ds)

    write_audit(ds, AUDIT_PATH)
    print(f"wrote the full keep/drop record to {AUDIT_PATH.relative_to(REPO_ROOT)}")
    print("FAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
