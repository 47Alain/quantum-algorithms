"""Assemble a parameter-consistent, numerically evaluable algorithm dataset.

This is the single place where the workbook is read for the advantage
plots.  It produces one row per catalogued algorithm that survives a
deliberately strict filter, plus an audit trail explaining every row that
did not.

The filter, in order:

1. **One metric per category.**  Serial rows use "Time Complexity (Worst
   Only)".  Quantum rows use "Time Complexity / Circuit Depth", and a row
   whose cell is labelled with a *gate count* ("Toffoli Gates:", "T
   Count:") is dropped rather than compared against a depth.  Parallel
   rows use "Parallel Algorithm Work" -- total operations -- because the
   reported parallel *time* is a span that assumes as many processors as
   the algorithm can use, which is not comparable to a single machine.

2. **The formula must be a determinate function of one variable.**
   Anything with a second free symbol is dropped.  That is what removes,
   for example, the adjacency-list matching algorithms written in both
   ``n`` and ``m`` while keeping the adjacency-matrix ones written in
   ``n`` alone.

3. **That one variable must mean the same thing family-wide.**  Each
   family gets a single canonical *semantic* parameter (see
   :mod:`src.params`), and a row is kept only if its variable carries
   that meaning.  This is what lets Minimum Spanning Tree compare quantum
   ``n`` against serial ``V`` -- both are "number of vertices" -- while
   refusing to compare Subset Sum's ``$O(2^{n/2})$`` against ``$O(nt)$``,
   whose ``t`` is a target sum rather than an input size.

Rule 2 on its own is too blunt for graph and geometry problems, where
almost every runtime is naturally written in two variables: serial
Minimum Spanning Tree is ``$O(E \\log V)$``, and dropping it would leave
the classical side of that family represented by a single 1957 entry.
``_WORST_CASE_BOUNDS`` recovers these by substituting the secondary
parameter's worst-case value in terms of the canonical one -- a simple
graph on V vertices has at most V(V-1)/2 edges, and a convex hull of n
points has at most n hull points.  This is sound precisely because the
columns being read are the workbook's *worst-case* columns, so the
worst-case value of the secondary parameter is the right one to use.  A
substitution is applied only after checking that the formula really does
grow with that parameter; the audit records every row it touched.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .formula import FormulaError, ParsedFormula, log10_runtime, parse_formula
from .params import NON_SIZE_SEMANTICS, SEMANTIC_LABELS, semantics_for_row
from .problems import GENERAL, display_name, normalize_family, variation_keys

REPO_ROOT = Path(__file__).resolve().parent.parent
XLSX_PATH = REPO_ROOT / "data" / "AlgoWiki algorithms (our copy) (3).xlsx"

CATEGORIES = ("serial", "parallel", "quantum")
CATEGORY_LABELS = {
    "serial": "Classical serial",
    "parallel": "Classical parallel",
    "quantum": "Quantum",
}
CATEGORY_METRIC = {
    "serial": "sequential running time",
    "parallel": "total work (operations)",
    "quantum": "circuit depth / running time",
}

MIN_YEAR = 1950
MAX_YEAR = 2026

# Formulas in more variables than this are hopeless to reduce and only slow
# the search down.
_MAX_SYMBOLS = 3

# (secondary semantic, canonical semantic) -> worst-case value of the secondary
# in terms of the canonical. Only relationships that are true by definition
# belong here; anything merely plausible is left out so that the row is dropped
# instead of being silently reinterpreted.
_WORST_CASE_BOUNDS: dict[tuple[str, str], "callable"] = {
    # A simple graph on V vertices has at most V(V-1)/2 edges.
    ("edges", "vertices"): lambda v: v * (v - 1) / 2,
    # Every input point can lie on the hull.
    ("hull_size", "points"): lambda n: n,
    ("hull_size", "segments"): lambda n: n,
    ("hull_size", "elements"): lambda n: n,
}

# A few definition cells label a quantity in a way that contradicts how the
# rest of the same family describes it. These rename the label only; no
# formula is touched, and each one is a case where the workbook's own rows
# disagree with each other.
_FAMILY_SEMANTIC_FIXES: dict[str, dict[str, str]] = {
    # Most Convex Hull rows carry a copy-pasted "n: number of line segments"
    # definition. The input to a hull algorithm is a point set, which is how
    # the quantum row and two of the parallel rows describe the same n.
    "convex hull": {"segments": "points"},
}


@dataclass
class AlgorithmRow:
    family: str
    family_label: str
    variation: str | None
    variations: frozenset[str]   # normalized variation keys this row solves
    category: str
    year: int
    name: str | None
    symbol: str
    semantic: str
    expr: object                 # SymPy expression in `symbol` alone
    source_formula: str
    cleaned_formula: str
    notes: tuple[str, ...] = ()
    curator_class: float | None = None   # the workbook's own complexity class

    def log10_at(self, size: float) -> float:
        """log10 of the operation count at the given problem size."""
        from .formula import log10_runtime
        return log10_runtime(self.expr, self.symbol, size)


@dataclass
class Dataset:
    rows: list[AlgorithmRow]
    family_parameter: dict[str, str]      # family -> canonical semantic tag
    audit: pd.DataFrame                   # every considered row, kept or not

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"family": r.family, "family_label": r.family_label,
             "variation": r.variation, "variations": "; ".join(sorted(r.variations)),
             "category": r.category, "year": r.year,
             "name": r.name, "symbol": r.symbol, "semantic": r.semantic,
             "formula": r.cleaned_formula, "source_formula": r.source_formula,
             "expr": str(r.expr), "curator_class": r.curator_class,
             "notes": "; ".join(r.notes)}
            for r in self.rows
        ])

    def families_with(self, *categories: str) -> list[str]:
        present = defaultdict(set)
        for r in self.rows:
            present[r.family].add(r.category)
        wanted = set(categories)
        return sorted(f for f, cats in present.items() if wanted <= cats)

    def parameter_label(self, family: str) -> str:
        tag = self.family_parameter.get(family, "")
        return SEMANTIC_LABELS.get(tag, tag.replace("raw:", ""))


# ---------------------------------------------------------------------------
# Sheet reading
# ---------------------------------------------------------------------------

_SHEET_SPEC = {
    "quantum": {
        "sheet": "Quantum Algorithms",
        "formula_col": "Time Complexity / Circuit Depth (Worst Only)",
    },
    "serial": {
        "sheet": "Sheet1",
        "formula_col": "Time Complexity (Worst Only)",
    },
    "parallel": {
        "sheet": "Parallel Algos",
        "formula_col": "Parallel Algorithm Work",
    },
}


def _first_present(df: pd.DataFrame, *names: str):
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([None] * len(df), index=df.index)


def _flag(df: pd.DataFrame, column: str, value: int = 1) -> pd.Series:
    """Rows whose yes/no column equals ``value``.

    A missing column or a blank, "0?" or "1?" cell never matches, so an
    exclusion built on this can only fire on an explicit answer.
    """
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df[column], errors="coerce") == value


def _clean_text(value) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def _strip_list_literal(value) -> str | None:
    """The Title and Authors columns hold stringified Python lists."""
    text = _clean_text(value)
    if not text:
        return None
    parts = re.findall(r"['\"]([^'\"]+)['\"]", text)
    if parts:
        return ", ".join(p.strip() for p in parts if p.strip()) or None
    return text.strip("[]") or None


def _resolve_names(raw: pd.DataFrame) -> pd.Series:
    """Best available label for each algorithm.

    "Algorithm Name" is blank for 87 of the 234 quantum rows, but the paper is
    still identified by Title and Authors -- the O((log n)^2) factoring entry
    is Brenner et al., *Factoring an integer with three oscillators and a
    qubit*. Falling back keeps those rows attributable in the audit instead of
    looking like unsourced stubs.
    """
    name = _first_present(raw, "Algorithm Name").map(_clean_text)
    title = _first_present(raw, "Title").map(_strip_list_literal)
    authors = _first_present(raw, "Authors").map(_strip_list_literal)

    def pick(n, t, a):
        if n:
            return n
        if t:
            return t if len(t) <= 70 else t[:67] + "..."
        if a:
            return a.split(",")[0] + " et al." if "," in a else a
        return None

    return pd.Series([pick(n, t, a) for n, t, a in zip(name, title, authors)],
                     index=raw.index)


def _load_category(xls: pd.ExcelFile, category: str) -> pd.DataFrame:
    spec = _SHEET_SPEC[category]
    raw = xls.parse(spec["sheet"])

    out = pd.DataFrame({
        "family": raw["Family Name"].map(normalize_family),
        "variation": _first_present(raw, "Variation").map(_clean_text),
        "year": pd.to_numeric(_first_present(raw, "Year"), errors="coerce"),
        "name": _resolve_names(raw),
        "formula": raw[spec["formula_col"]],
        "param_defs": _first_present(raw, "Parameter definitions"),
        "curator_class": _first_present(raw, "Param: Time Class"),
    })
    out["category"] = category

    # An approximation buys its speed with a weaker answer, so its runtime is
    # not comparable to an exact algorithm's -- racing classical Max Cut's
    # poly-time approximation against an exact quantum 2^(n/2) is meaningless.
    #
    # The test is whether the algorithm *answers* exactly, which is not the
    # same as whether its running time was proved rigorously. The workbook
    # flags the number field sieve "heuristic-based" because its runtime
    # analysis assumes smooth numbers behave randomly, but it returns true
    # factors and is the classical state of the art; excluding it would gut
    # the comparison it exists to anchor.
    approximate = _flag(raw, "Approximate?", 1)
    inexact = _flag(raw, "Exact algorithm?", 0)
    out["excluded"] = None
    out.loc[inexact, "excluded"] = "does not solve the problem exactly"
    out.loc[approximate, "excluded"] = (
        "approximation algorithm, not comparable to an exact runtime")

    if category == "serial":
        # Sheet1 also holds rows flagged as parallel or quantum; those belong
        # to their own sheets and would otherwise be double counted.
        is_parallel = _flag(raw, "Parallel?")
        is_quantum = _flag(raw, "Quantum?")
        out = out[~(is_parallel | is_quantum)]
        # A few entries slip through unflagged but announce themselves in the
        # name; "Bitonic Merge Sort Parallel Implementation, O(log^2 n)" is a
        # span, and reading it as a sequential running time would make the
        # fastest classical sorter look polylogarithmic.
        looks_parallel = out["name"].fillna("").str.contains(
            r"parallel implementation", case=False, regex=True)
        out.loc[looks_parallel & out["excluded"].isna(), "excluded"] = (
            "a parallel implementation listed on the sequential sheet; its "
            "runtime is a span, not a sequential running time")

    return out


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def _resolve_symbol_semantics(
    candidate: pd.DataFrame,
) -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, dict[str, str]]]:
    """Consensus ``{symbol: semantic}`` maps for rows that define nothing.

    Roughly half the quantum rows leave "Parameter definitions" blank, and a
    blank cell has to borrow its meaning from somewhere. It borrows from the
    row's *own sheet* first, because the sheets were curated separately and do
    not always agree: every Integer Factoring row on the quantum and serial
    sheets defines ``n`` as the bit length, while every one on the parallel
    sheet defines it as the integer itself. Reading a blank parallel cell as
    bits would evaluate the number field sieve at N=1024 rather than N=2^1024.

    The family-wide map is only consulted when a sheet is silent about a
    symbol, and only where the whole family agrees on it.
    """
    sheet_votes: dict[tuple[str, str], dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter))
    family_votes: dict[str, dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter))
    for row in candidate.itertuples():
        for symbol, tag in row.semantics.items():
            sheet_votes[(row.family, row.category)][symbol][tag] += 1
            family_votes[row.family][symbol][tag] += 1

    def settle(votes, unanimous_only: bool):
        out = {}
        for key, symbols in votes.items():
            resolved = {}
            for symbol, counter in symbols.items():
                (top, top_n), = counter.most_common(1)
                if len(counter) == 1 or (
                        not unanimous_only and top_n >= 0.75 * sum(counter.values())):
                    resolved[symbol] = top
            out[key] = resolved
        return out

    return settle(sheet_votes, False), settle(family_votes, True)


def _reduce_to_parameter(parsed: ParsedFormula, tags: dict[str, str],
                         canonical: str) -> tuple[str, object, list[str]] | None:
    """Try to express ``parsed`` as a function of the canonical quantity alone.

    Returns ``(symbol, expression, notes)`` or None if the formula cannot be
    reduced. Secondary parameters are replaced by their worst-case value in
    terms of the canonical one, but only when the runtime actually increases
    with them, so the substitution can never flatter an algorithm.
    """
    import sympy

    primary = [s for s in parsed.symbols if tags.get(s) == canonical]
    if len(primary) != 1:
        return None
    symbol = primary[0]
    others = [s for s in parsed.symbols if s != symbol]
    if not others:
        return symbol, parsed.expr, []

    expr = parsed.expr
    notes = []
    for other in others:
        tag = tags.get(other)
        bound = _WORST_CASE_BOUNDS.get((tag, canonical)) if tag else None
        if bound is None:
            return None
        sym = sympy.Symbol(symbol)
        substituted = expr.subs(sympy.Symbol(other), bound(sym))
        # Confirm the runtime grows with the parameter being bounded; if it
        # shrinks, the worst case is at the small end and this rewrite would
        # be wrong.
        small = _safe_log10(expr.subs(sympy.Symbol(other), sympy.Integer(2)), symbol, 1e4)
        large = _safe_log10(substituted, symbol, 1e4)
        if small is None or large is None or large < small - 1e-9:
            return None
        expr = substituted
        notes.append(f"substituted worst-case {_describe(tag)} for '{other}'")
    return symbol, expr, notes


def _safe_log10(expr, symbol: str, size: float) -> float | None:
    from .formula import log10_runtime
    try:
        return log10_runtime(expr, symbol, size)
    except FormulaError:
        return None


def _choose_family_parameter(group: pd.DataFrame) -> str | None:
    """Pick the one semantic quantity a family will be measured in.

    Each candidate quantity is scored by how many rows would actually survive
    if it were chosen, preferring the choice that covers the most *categories*
    -- a parameter present in only one category cannot support any
    cross-category comparison.
    """
    candidates = Counter()
    for row in group.itertuples():
        for tag in row.tags.values():
            if tag and tag not in NON_SIZE_SEMANTICS:
                candidates[tag] += 1
    if not candidates:
        return None

    best, best_score = None, None
    for tag in candidates:
        kept_categories, kept_rows = set(), 0
        for row in group.itertuples():
            if _reduce_to_parameter(row.parsed, row.tags, tag) is not None:
                kept_categories.add(row.category)
                kept_rows += 1
        score = (len(kept_categories), kept_rows)
        if kept_rows and (best_score is None or score > best_score):
            best, best_score = tag, score
    return best


def build_dataset(xlsx_path: Path | None = None, verbose: bool = True) -> Dataset:
    xls = pd.ExcelFile(xlsx_path or XLSX_PATH)
    frames = [_load_category(xls, c) for c in CATEGORIES]
    raw = pd.concat(frames, ignore_index=True)

    records = []
    for row in raw.itertuples():
        rec = {
            "family": row.family, "variation": row.variation,
            "category": row.category, "year": row.year, "name": row.name,
            "source_formula": row.formula, "semantics": {}, "symbol": None,
            "semantic": None, "parsed": None, "kept": False, "reason": "",
            "curator_class": row.curator_class,
        }

        if not row.family:
            rec["reason"] = "no problem family recorded"
            records.append(rec)
            continue
        if row.excluded:
            rec["reason"] = row.excluded
            records.append(rec)
            continue
        if pd.isna(row.year) or not (MIN_YEAR <= row.year <= MAX_YEAR):
            rec["reason"] = "year missing or outside 1950-2026"
            records.append(rec)
            continue
        rec["year"] = int(row.year)
        rec["semantics"] = semantics_for_row(row.param_defs)

        try:
            parsed = parse_formula(row.formula)
        except FormulaError as exc:
            rec["reason"] = f"formula not usable: {exc}"
            records.append(rec)
            continue

        if row.category == "quantum" and parsed.metric == "work":
            rec["reason"] = ("cell reports a gate count, not circuit depth; "
                             "not comparable to a running time")
            records.append(rec)
            continue

        if not parsed.symbols:
            rec["reason"] = "formula is a constant with no problem size"
            records.append(rec)
            continue
        if len(parsed.symbols) > _MAX_SYMBOLS:
            rec["reason"] = ("formula depends on too many parameters "
                             f"({', '.join(sorted(parsed.symbols))})")
            records.append(rec)
            continue

        rec["parsed"] = parsed
        records.append(rec)

    candidate = pd.DataFrame(records)
    candidate["tags"] = [{} for _ in range(len(candidate))]

    # What each symbol means, per row: the row's own definitions first, then
    # its sheet's consensus, then the family's, for rows that left the
    # definitions blank.
    sheet_consensus, family_consensus = _resolve_symbol_semantics(
        candidate[candidate["family"].notna()])
    for idx, row in candidate.iterrows():
        if row["parsed"] is None:
            continue
        sheet_map = sheet_consensus.get((row["family"], row["category"]), {})
        family_map = family_consensus.get(row["family"], {})
        fixes = _FAMILY_SEMANTIC_FIXES.get(row["family"], {})
        tags = {}
        for s in row["parsed"].symbols:
            tag = row["semantics"].get(s) or sheet_map.get(s) or family_map.get(s)
            tags[s] = fixes.get(tag, tag)
        candidate.at[idx, "tags"] = tags

    # One canonical parameter per family, chosen to maximise what survives.
    family_parameter: dict[str, str] = {}
    usable = candidate[candidate["parsed"].notna() & candidate["family"].notna()]
    for family, group in usable.groupby("family"):
        chosen = _choose_family_parameter(group)
        if chosen:
            family_parameter[family] = chosen

    rows: list[AlgorithmRow] = []
    for idx, row in candidate.iterrows():
        parsed = row["parsed"]
        if parsed is None:
            continue
        canonical = family_parameter.get(row["family"])
        tags = row["tags"]
        if canonical is None:
            candidate.at[idx, "reason"] = "family has no usable input-size parameter"
            continue

        reduced = _reduce_to_parameter(parsed, tags, canonical)
        if reduced is None:
            candidate.at[idx, "reason"] = _explain_rejection(parsed, tags, canonical)
            continue
        symbol, expr, notes = reduced

        try:
            log10_runtime(expr, symbol, 1e6)
        except FormulaError as exc:
            candidate.at[idx, "reason"] = f"formula could not be evaluated: {exc}"
            continue

        candidate.at[idx, "kept"] = True
        candidate.at[idx, "symbol"] = symbol
        candidate.at[idx, "semantic"] = canonical
        candidate.at[idx, "reason"] = "; ".join(["kept", *notes]) if notes else "kept"
        rows.append(AlgorithmRow(
            family=row["family"], family_label=display_name(row["family"]),
            variation=row["variation"],
            variations=variation_keys(row["variation"], row["family"]),
            category=row["category"],
            year=int(row["year"]), name=row["name"], symbol=symbol,
            semantic=canonical, expr=expr, source_formula=str(row["source_formula"]),
            cleaned_formula=parsed.cleaned, notes=tuple(parsed.notes) + tuple(notes),
            curator_class=curator_class_for(row["curator_class"], symbol),
        ))

    audit = candidate.drop(columns=["parsed", "semantics", "tags", "curator_class"]).copy()
    dataset = Dataset(rows=rows, family_parameter=family_parameter, audit=audit)

    if verbose:
        _print_summary(dataset, raw)
    return dataset


_CLASS_ENTRY = re.compile(r"([^\s:,;]+)\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)")


def curator_class_for(cell, symbol: str) -> float | None:
    """The workbook's own complexity class for one parameter of a row.

    "Param: Time Class" reads like ``n: 8,\\nt: 4``: an ordinal severity score
    the curators assigned per parameter, independent of the formula text. It
    is never used to build the dataset -- only to check the parser against, in
    :mod:`validate_dataset`.
    """
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return None
    text = str(cell).strip()
    if not text:
        return None
    entries = {m.group(1).strip("$\\ "): float(m.group(2))
               for m in _CLASS_ENTRY.finditer(text)}
    if symbol in entries:
        return entries[symbol]
    if not entries:
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _describe(tag: str | None) -> str:
    if not tag:
        return "an undefined parameter"
    return SEMANTIC_LABELS.get(tag, tag.replace("raw:", "").replace("_", " "))


def _explain_rejection(parsed: ParsedFormula, tags: dict[str, str],
                       canonical: str) -> str:
    """A specific, human-checkable reason a row failed the parameter gate."""
    primary = [s for s in parsed.symbols if tags.get(s) == canonical]
    if len(primary) > 1:
        return (f"formula uses {len(primary)} different symbols that all mean "
                f"{_describe(canonical)}")
    if not primary:
        unknown = [s for s in parsed.symbols if not tags.get(s)]
        if unknown:
            return (f"cannot tell what {', '.join(sorted(unknown))} means "
                    "(no definition given and the family is inconsistent)")
        measured = sorted({_describe(tags[s]) for s in parsed.symbols})
        return (f"measured in {', '.join(measured)}, but this family is "
                f"compared in {_describe(canonical)}")
    blockers = []
    for s in parsed.symbols:
        if s in primary:
            continue
        tag = tags.get(s)
        if (tag, canonical) not in _WORST_CASE_BOUNDS:
            blockers.append(f"'{s}' ({_describe(tag)})")
    if blockers:
        return ("also depends on " + ", ".join(sorted(blockers))
                + f", which has no worst-case bound in terms of {_describe(canonical)}")
    return "formula does not increase with the parameter being bounded"


def _print_summary(dataset: Dataset, raw: pd.DataFrame) -> None:
    frame = dataset.frame()
    print(f"  read {len(raw)} catalogued algorithms from 3 sheets")
    print(f"  retained {len(frame)} with a parameter-consistent, evaluable runtime")
    by_cat = frame["category"].value_counts()
    for cat in CATEGORIES:
        print(f"      {CATEGORY_LABELS[cat]:<20} {int(by_cat.get(cat, 0)):>5}"
              f"   (metric: {CATEGORY_METRIC[cat]})")
    print(f"  families represented: {frame['family'].nunique()}"
          f"   with quantum + serial: {len(dataset.families_with('quantum', 'serial'))}")


def write_audit(dataset: Dataset, path: Path) -> None:
    """Write the full keep/drop record so every figure can be traced back."""
    audit = dataset.audit.copy()
    audit["family_label"] = audit["family"].map(
        lambda f: display_name(f) if isinstance(f, str) else "")
    audit["canonical_parameter"] = audit["family"].map(
        lambda f: _describe(dataset.family_parameter.get(f, "")) if isinstance(f, str) else "")
    audit["variation_keys"] = [
        "; ".join(sorted(variation_keys(v, f))) if isinstance(f, str) else ""
        for v, f in zip(audit["variation"], audit["family"])]
    columns = ["family_label", "variation", "variation_keys", "category", "year",
               "name", "source_formula", "symbol", "semantic",
               "canonical_parameter", "kept", "reason"]
    audit = audit[[c for c in columns if c in audit.columns]]
    audit = audit.sort_values(["family_label", "category", "year"], na_position="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(path, index=False, encoding="utf-8")
