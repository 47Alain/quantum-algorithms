"""Decide when two complexity formulas are talking about the same quantity.

Comparing runtimes across sheets is only meaningful if the variable inside
them means the same thing, and in this workbook the *symbol* is not a
reliable guide:

* Minimum Spanning Tree quantum rows are written in ``n`` ("number of
  vertices in the graph") while the serial rows for the same problem use
  ``V`` ("number of vertices").  Different symbol, identical quantity --
  these must be compared.
* Subset Sum serial rows include ``$O(nt)$`` where ``t`` is the *target
  sum*.  That is a pseudo-polynomial runtime living on a completely
  different axis from ``$O(2^{n/2})$``; treating ``t`` as interchangeable
  with ``n`` would make the classical algorithm look exponentially better
  than it is.  These must never be compared.

So the unit of identity here is the curator's own prose definition, not
the letter.  ``parse_definitions`` turns the "Parameter definitions" cell
into ``{symbol: meaning}``, and ``semantic_of`` collapses the prose onto a
canonical tag.  Anything the rule table does not recognize falls back to
the normalized definition text itself, so unfamiliar parameters stay
distinct from one another instead of silently merging.
"""
from __future__ import annotations

import math
import re

# Ordered rules: the first pattern that matches the definition text wins, so
# more specific phrases must come before more general ones.
_SEMANTIC_RULES: list[tuple[str, str]] = [
    # --- quantities that are emphatically NOT the input size ---
    (r"number of processors|processor count|\bprocessors\b", "processors"),
    (r"target sum|target value", "target_sum"),
    (r"condition number", "condition_number"),
    (r"\bsparsity\b|maximum number of nonzero|nonzero entries", "sparsity"),
    (r"approximation (factor|error)|error parameter|success probability|"
     r"failure probability|accuracy", "error_tolerance"),
    (r"maximum edge (capacity|weight)|max(imum)? (absolute )?(value of )?"
     r"(edge )?(cost|weight|capacity)", "max_edge_weight"),
    (r"size of (the )?(vertex cover|solution|cover|clique|set cover)|"
     r"solution size|size of the answer", "solution_size"),
    (r"branching factor", "branching_factor"),
    (r"depth of the solution|solution depth", "solution_depth"),
    (r"size of (the )?alphabet|alphabet size", "alphabet_size"),
    (r"number of (output |resulting )?(points on the convex hull|hull points)",
     "hull_size"),
    (r"treewidth", "treewidth"),
    (r"\bpathwidth\b", "pathwidth"),
    (r"\bcutwidth\b", "cutwidth"),
    (r"number of (distinct )?(dimensions|dimension)\b|dimensionality", "dimensions"),

    # --- genuine input-size measures ---
    (r"number of vertices|number of nodes|\bvertices\b|\bnodes\b", "vertices"),
    (r"number of edges|\bedges\b", "edges"),
    (r"number of bits|bit length|length of .* in bits|bits (needed |encoding|"
     r"to represent)|bits of|qubits of", "input_bits"),
    (r"dimension of (the )?(non-hermitian |square |input )?matrix|"
     r"matrix dimension|dimension of grid|order of the matrix", "matrix_dim"),
    (r"number of matrices", "matrices"),
    (r"number of states", "states"),
    (r"number of cities", "cities"),
    (r"number of (clauses|constraints)", "clauses"),
    (r"number of variables", "variables"),
    (r"number of (input )?points", "points"),
    (r"number of line segments|number of segments", "segments"),
    (r"number of sequences|number of strings", "sequences"),
    # Two-string problems name their operands separately, and "longer" is not
    # interchangeable with "shorter", so they keep distinct tags.
    (r"length of (the )?long(er|est)", "string_length_longer"),
    (r"length of (the )?short(er|est)", "string_length_shorter"),
    (r"(length|size) of (the )?(searchable )?text|text length", "text_length"),
    (r"(length|size) of (the )?pattern|pattern length", "pattern_length"),
    (r"integer (to be|being) factor(ed|ized)|number to (be )?factor", "integer_value"),
    (r"number of processes\b", "processes"),
    (r"length of (the |one of the )?(two )?(input )?(strings?|"
     r"sequence|cycle string)|string length|size of (the )?(input )?string",
     "string_length"),
    (r"(length|size|dimension) of (the )?(input )?(array|data|list|vector)",
     "elements"),
    (r"number of (input )?(elements|integers|items|keys|numbers|observations|"
     r"records)|size of (the )?(list|input|array|set|sample)|elements to be "
     r"sorted|length of the (input )?(data set|list|array)|number of "
     r"(elements|integers) in the set", "elements"),
]

_COMPILED_RULES = [(re.compile(p, re.IGNORECASE), tag) for p, tag in _SEMANTIC_RULES]

# Human-readable axis labels for the tags a plot is likely to show.
SEMANTIC_LABELS = {
    "vertices": "vertices in the graph",
    "edges": "edges in the graph",
    "elements": "elements in the input",
    "input_bits": "bits of input",
    "matrix_dim": "matrix dimension",
    "points": "input points",
    "string_length": "input string length",
    "states": "states",
    "matrices": "matrices in the chain",
    "dimensions": "dimensions",
    "sequences": "input sequences",
    "segments": "line segments",
    "text_length": "text length",
    "pattern_length": "pattern length",
    "string_length_longer": "length of the longer string",
    "string_length_shorter": "length of the shorter string",
    "integer_value": "the integer being factored",
    "processes": "processes",
    "variables": "variables",
    "clauses": "clauses",
    "cities": "cities",
}

# Tags that describe something other than "how big is the input". A runtime
# expressed purely in one of these is not a function of problem size and can
# never be placed on the same axis as one that is.
NON_SIZE_SEMANTICS = frozenset({
    "processors", "target_sum", "condition_number", "sparsity",
    "error_tolerance", "max_edge_weight", "solution_size", "branching_factor",
    "solution_depth", "alphabet_size", "hull_size", "treewidth", "pathwidth",
    "cutwidth",
})

_STOP_WORDS = {"the", "a", "an", "of", "in", "for", "to", "is", "are", "and",
               "input", "given", "its", "each", "this", "that"}


def _normalize_prose(text: str) -> str:
    t = re.sub(r"\$|\\[A-Za-z]+|[{}]", " ", text.lower())
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    words = [w for w in t.split() if w not in _STOP_WORDS]
    return " ".join(words)


def semantic_of(definition: str | None) -> str | None:
    """Canonical tag for one parameter's prose definition.

    Falls back to the normalized definition text when no rule matches, so
    unrecognized parameters remain distinguishable rather than colliding.
    """
    if not definition:
        return None
    text = definition.strip()
    if not text:
        return None
    for pattern, tag in _COMPILED_RULES:
        if pattern.search(text):
            return tag
    normalized = _normalize_prose(text)
    return f"raw:{normalized}" if normalized else None


# A definition entry starts with a symbol followed by ":" / ";" / "-". Symbols
# may be wrapped in $...$, carry a LaTeX backslash, or a {subscript}.
_ENTRY_START = re.compile(
    r"(?:^|[\n;/,]|\s{2,}|(?<=[a-z])\s(?=\$))\s*"
    r"\$?\s*(\\?[A-Za-z][A-Za-z0-9]*(?:_\{?[A-Za-z0-9]+\}?)?|\\[A-Za-z]+)\s*\$?\s*"
    r"[:;]\s*"
)


def _clean_symbol(raw: str) -> str:
    s = raw.strip().lstrip("\\").replace("$", "")
    s = re.sub(r"_\{([A-Za-z0-9]+)\}", r"_\1", s)
    return s.strip()


def parse_definitions(cell) -> dict[str, str]:
    """Turn a "Parameter definitions" cell into ``{symbol: prose}``.

    Entries are separated inconsistently -- newlines, slashes, semicolons, or
    just a run of spaces -- so the split is driven by where the next
    ``symbol:`` marker begins rather than by any one delimiter.
    """
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return {}
    text = str(cell).strip()
    if not text or text.lower() == "nan":
        return {}
    text = re.sub(r"\$\s*([A-Za-z][A-Za-z0-9]*)\s*\$\s*;", r"$\1$:", text)

    matches = list(_ENTRY_START.finditer(text))
    if not matches:
        return {}

    out: dict[str, str] = {}
    for i, m in enumerate(matches):
        symbol = _clean_symbol(m.group(1))
        if not symbol:
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        prose = text[m.end():end].strip().strip("/;, ").strip()
        if prose and symbol not in out:
            out[symbol] = prose
    return out


def semantics_for_row(cell) -> dict[str, str]:
    """``{symbol: semantic tag}`` for one row's parameter-definition cell."""
    out = {}
    for symbol, prose in parse_definitions(cell).items():
        tag = semantic_of(prose)
        if tag:
            out[symbol] = tag
    return out
