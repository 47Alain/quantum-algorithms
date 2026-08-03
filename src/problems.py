"""Reconcile problem names across the workbook's three algorithm sheets.

The Quantum, Sheet1 (serial) and Parallel Algos sheets were curated at
different times and do not agree on spelling: the quantum sheet says
"DFT" where the serial sheet says "Discrete Fourier Transform", and
"Minumum Spanning Tree (MST)" appears alongside "Minimum Spanning Tree
(MST)".  Comparing categories requires a single key per problem.

The family alone is too coarse to compare across sheets.  Graph Coloring's
quantum entries solve the **chromatic number**, while every classical entry
in the family solves **3-** or **4-coloring**; reading them as one problem
makes quantum look 10^254200 times slower than classical.  Maximum Cut's
classical side includes a planar-graphs-only result and a flagged
approximation.  So the comparison unit is ``(family, variation)``.

Variations need real work before they match.  A cell can list several
("APSP on Dense Undirected Unweighted Graphs; APSP on Sparse Undirected
Unweighted Graphs"), so each is split into a set and two rows are comparable
when their sets intersect.  Sheets abbreviate differently -- "SCCs" against
"Strongly Connected Components", "MCOP" against "Matrix Chain Ordering
Problem" -- so an alias table folds the synonyms together.  Quantum rows
often add a graph-representation annotation, "Undirected, General MST
(Adjacency Matrix Model)", that names the input encoding rather than a
different problem, and it is stripped.

Where the sheets genuinely disagree about what problem was solved, nothing
matches and the family drops out of every cross-category figure.  That is
the intended behaviour: Graph Coloring and Longest Path leave the comparison
rather than contributing a fabricated speedup.
"""
from __future__ import annotations

import re

# Left: the name as spelled in some sheet (after normalization).
# Right: the canonical family key.
_FAMILY_ALIASES: dict[str, str] = {
    "dft": "discrete fourier transform",
    "minumum spanning tree (mst)": "minimum spanning tree (mst)",
    "all-pairs shortest paths": "all-pairs shortest paths (apsp)",
    "shortest-path (directed graphs)": "shortest path (directed graphs)",
    "graph isomorphism": "graph isomorphism problem",
    "nearest neighbor": "nearest neighbor search",
    "nash equilbrium": "nash equilibria",
    "nash equilibrium": "nash equilibria",
    "the traveling salesman problem": "the traveling-salesman problem",
    "maximum cardinality matching (mcm)": "maximum cardinality matching",
    "minimum spanning tree": "minimum spanning tree (mst)",
    "boolean satisfiability (sat)": "boolean satisfiability",
    "max cut": "maximum cut",
    "set cover": "the set-covering problem",
    "subset sum": "the subset-sum problem",
    "vertex cover": "the vertex cover problem",
    "lowest common ancestor (lca)": "lowest common ancestor",
    "strongly connected components (scc)": "strongly connected components",
}

# Display names for the canonical keys, used in figure labels.
_DISPLAY_OVERRIDES = {
    "the subset-sum problem": "Subset Sum",
    "the vertex cover problem": "Vertex Cover",
    "the set-covering problem": "Set Cover",
    "the traveling-salesman problem": "Traveling Salesman",
    "minimum spanning tree (mst)": "Minimum Spanning Tree",
    "all-pairs shortest paths (apsp)": "All-Pairs Shortest Paths",
    "shortest path (directed graphs)": "Shortest Path (Directed)",
    "graph isomorphism problem": "Graph Isomorphism",
    "nearest neighbor search": "Nearest Neighbor Search",
    "discrete fourier transform": "Discrete Fourier Transform",
    "maximum cardinality matching": "Maximum Cardinality Matching",
    "strongly connected components": "Strongly Connected Components",
    "factorization of polynomials over finite fields": "Polynomial Factorization (Finite Fields)",
    "eigenvalues (iterative methods)": "Eigenvalues (Iterative)",
    "matrix chain multiplication": "Matrix Chain Multiplication",
    "optimal binary search trees": "Optimal Binary Search Trees",
    "sdd systems solvers": "SDD / Laplacian Solvers",
    "dfa minimization": "DFA Minimization",
    "lu decomposition": "LU Decomposition",
}


# The key used when a row does not name a variation, or names the family
# itself: "the problem in the family's own general form".
GENERAL = "(general)"

# Parenthetical qualifiers that describe how the input is encoded rather than
# which problem is being solved.
_MODEL_ANNOTATION = re.compile(
    r"\s*\((?:adjacency (?:list|matrix) model|quantum walks?|nns|mst|apsp|scc|"
    r"sccs|cc|mcm|dft)\)\s*", re.IGNORECASE)

# Synonyms across sheets. Applied before the "is this just the family name?"
# test, so an alias may resolve all the way to GENERAL.
_VARIATION_ALIASES: dict[str, str] = {
    "sccs": "strongly connected components",
    "cc": "connected components in an undirected graph",
    "connected components": "connected components in an undirected graph",
    "transitive closure": "transitive closure of a symmetric boolean matrix",
    "mcop": "matrix chain ordering problem",
    "mcsp": "matrix chain scheduling problem",
    "bipartite graphs mcm": "bipartite graph mcm",
    "general graphs mcm": "general graph mcm",
    # "First/Second Category" describes the algorithm's dependence on the size
    # of the factor versus the size of the modulus, not a different problem.
    "first category integer factoring": GENERAL,
    "second category integer factoring": GENERAL,
    # The catalogued quantum sorting results (Farhi et al.; Hoyer, Neerbek &
    # Shi) are comparison-model bounds, so they belong with comparison sorting;
    # non-comparison sorting stays a separate problem.
    "comparison sorting": GENERAL,
    "general weights": "general weights",
    "nonnegative weighted digraph": "nonnegative weights",
    "nonnegative weights, undirected": "nonnegative weights, undirected",
    "undirected, nonnegative weights": "nonnegative weights, undirected",
}


def _normalize_variation_token(token: str, family: str) -> str | None:
    text = _MODEL_ANNOTATION.sub(" ", token)
    text = re.sub(r"\s+", " ", text).strip().strip(".,;:").lower()
    if not text or text in {"nan", "-", "?"}:
        return None
    text = _VARIATION_ALIASES.get(text, text)
    if text == GENERAL:
        return GENERAL
    # A variation that merely restates the family is the family's general form.
    if text in {family, display_name(family).lower()}:
        return GENERAL
    stripped = re.sub(r"^the\s+|\s+problem$", "", family)
    if text == stripped:
        return GENERAL
    return text


def variation_keys(variation, family: str) -> frozenset[str]:
    """The set of problem variations one row's "Variation" cell claims.

    Returns ``{GENERAL}`` for a blank cell, since a row that names no
    variation is understood to solve the family's general problem.
    """
    if variation is None or not isinstance(variation, str):
        return frozenset({GENERAL})
    tokens = [t for t in re.split(r"[;/]", variation) if t.strip()]
    keys = {k for k in (_normalize_variation_token(t, family) for t in tokens) if k}
    return frozenset(keys) if keys else frozenset({GENERAL})


def variation_label(key: str) -> str:
    """Readable name for a variation key."""
    if key == GENERAL:
        return "general form"
    return key[:1].upper() + key[1:]


def normalize_family(name) -> str | None:
    """Canonical key for a "Family Name" cell, or None if the cell is empty."""
    if name is None or not isinstance(name, str):
        return None
    key = re.sub(r"\s+", " ", name).strip().lower()
    if not key or key == "nan":
        return None
    key = key.replace("minumum", "minimum").replace("equilbrium", "equilibrium")
    return _FAMILY_ALIASES.get(key, key)


def display_name(key: str) -> str:
    """Title-cased label for a canonical family key."""
    if key in _DISPLAY_OVERRIDES:
        return _DISPLAY_OVERRIDES[key]
    words = key.split()
    small = {"of", "in", "the", "a", "an", "and", "for", "over", "to", "with"}
    titled = [w.upper() if w in {"lu", "dfa", "mst", "apsp", "sdd", "dft", "lcs"}
              else (w if i and w in small else w.capitalize())
              for i, w in enumerate(words)]
    return " ".join(titled)
