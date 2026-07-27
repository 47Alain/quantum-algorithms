"""Turn AlgoWiki complexity strings into numbers.

The workbook records runtimes as free-text formulas in a mix of LaTeX and
ASCII: ``$O(n \\log n)$``, ``O(2^(0.241*n))``, ``\\tilde{O}(n^{2.5})``,
``Toffoli Depth: O(n^1.143)``.  Every downstream plot needs an actual
number -- "how many operations at n = 10^6" -- so this module does three
things:

1. ``clean_formula``  -- strip bound notation, LaTeX and metric labels, and
   reject anything whose meaning is not a single determinate runtime
   (conditional formulas, ``poly(n)``, lower bounds, ...).
2. ``parse_formula``  -- hand the cleaned string to SymPy and report the
   free symbols that survive.
3. ``log10_runtime`` -- evaluate the parsed expression at a given problem
   size, in log10 space.

Everything is evaluated in log10 space on purpose.  A single Subset Sum
entry is ``O(2^(n/2))``; at n = 10^9 that is a number with 150 million
digits, so any attempt to materialize it as a float overflows instantly.
Carrying log10 throughout keeps exponential and polynomial algorithms on
one comparable scale, and every plot in this repo consumes ratios of
runtimes, which are differences of log10 values.

Conventions
-----------
* ``log`` means log base 2 (the standard convention in algorithm
  analysis) and ``ln`` means natural log -- except underneath an
  ``exp(...)``, where a bare ``log`` is read as natural.  Sub-exponential
  runtimes are quoted from the L-notation literature, whose inner logs are
  natural, and there the base is not a constant factor.
* Soft-O forms (``\\tilde{O}``, ``Õ``, ``O^*``) are read as their
  argument.  They suppress polylog / polynomial factors that the curators
  chose not to record, so the argument is the best available estimate.
* Big-Omega is a *lower* bound, not a runtime, and is rejected.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# Metric labels that quantum rows prefix onto a formula. The workbook's
# quantum time column is "Time Complexity / Circuit Depth", so depth-like
# labels are genuine time measurements; count/size-like labels are total
# gate counts (work) and must not be compared against a depth. We keep the
# label so the caller can enforce that separation.
_DEPTH_LABELS = {"circuit depth", "depth", "toffoli depth", "toffli depth",
                 "t depth", "tofoli depth", "time", "span"}
_WORK_LABELS = {"toffoli gates", "toffolli gates", "t count", "toffoli count",
                "cnot gates", "gate count", "circuit size", "work", "gates"}
_LABEL_RE = re.compile(
    r"^\s*(" + "|".join(sorted(map(re.escape, _DEPTH_LABELS | _WORK_LABELS), key=len, reverse=True)) + r")\s*:\s*",
    re.IGNORECASE,
)

# Text that means "this cell is empty" rather than "this runtime is X".
_NULL_TOKENS = {"", "-", "--", "nan", "none", "n/a", "na", "?", "xxx", "xxxx",
                "#value!", "tbd", "unknown"}

# Constructs that make a cell impossible to evaluate as one number.
_REJECT_PATTERNS: list[tuple[str, str]] = [
    (r"\bomega\s*\(", "Big-Omega is a lower bound, not a runtime"),
    (r"\bpoly\s*\(", "poly(...) has no determinate exponent"),
    (r"polylog", "polylog(...) has no determinate exponent"),
    (r"\bif\b", "formula is conditional on a case split"),
    (r"\botherwise\b", "formula is conditional on a case split"),
    (r"\bamortized\b", "amortized qualifier"),
]

# LaTeX / unicode symbol names mapped to plain tokens so that, e.g., every
# spelling of epsilon collapses to one free symbol instead of several.
_GREEK = {
    "epsilon": "epsilon", "varepsilon": "epsilon", "eps": "epsilon",
    "\u03b5": "epsilon", "\u03f5": "epsilon",
    "kappa": "kappa", "\u03ba": "kappa",
    "omega": "omega", "\u03c9": "omega",
    "sigma": "sigma", "\u03c3": "sigma",
    "mu": "mu", "\u03bc": "mu",
    "delta": "delta", "\u03b4": "delta",
    "alpha": "alpha", "\u03b1": "alpha",
    "lambda": "lambda_", "\u03bb": "lambda_",
    "gamma": "gamma", "\u03b3": "gamma",
    "tau": "tau", "\u03c4": "tau",
    "rho": "rho", "\u03c1": "rho",
    "theta": "theta_", "\u03b8": "theta_",
    "phi": "phi", "\u03c6": "phi",
    "beta": "beta", "\u03b2": "beta",
    "zeta": "zeta", "eta": "eta", "\u03b7": "eta",
    "pi": "pi_", "\u03c0": "pi_",
    "chi": "chi", "psi": "psi", "xi": "xi", "nu": "nu",
}


class FormulaError(ValueError):
    """Raised when a cell cannot be reduced to one determinate runtime."""


@dataclass
class ParsedFormula:
    """A complexity cell that survived parsing."""

    source: str                       # original cell text
    cleaned: str                      # normalized string handed to SymPy
    expr: sympy.Expr                  # parsed expression
    symbols: frozenset[str]           # free variable names
    metric: str = "time"              # "time" (depth/span) or "work" (gate count)
    notes: tuple[str, ...] = field(default_factory=tuple)  # approximations applied

    def log10_at(self, symbol: str, value: float) -> float:
        return log10_runtime(self.expr, symbol, value)


# ---------------------------------------------------------------------------
# Step 1 -- cleaning
# ---------------------------------------------------------------------------

def _strip_latex(text: str) -> str:
    t = text.replace("\u00a0", " ")
    # Some cells were pasted with their escapes doubled ("\\log"); left alone,
    # the command is consumed and a stray backslash breaks tokenization.
    t = re.sub(r"\\{2,}", r"\\", t)
    # Soft-O written in LaTeX. Normalize to the ASCII "O~" spelling first, so
    # that the generic "\command" sweep below cannot eat the accent and leave
    # a bare "(O)(...)" behind.
    t = re.sub(r"\\(?:tilde|widetilde|hat|widehat)\s*\{?\s*O\s*\}?", " O~", t)
    t = t.replace("\u02dc", "~").replace("\u223c", "~")
    # \text{...}, \mathrm{...}, \mathcal{...} wrappers contribute no math
    t = re.sub(r"\\(?:text|mathrm|mathcal|mathbf|operatorname)\s*\{([^{}]*)\}", r" \1 ", t)
    t = re.sub(r"\\left|\\right|\\!|\\,|\\;|\\ ", " ", t)
    t = re.sub(r"\\cdot|\\times|\\ast", "*", t)
    t = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", t)
    t = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", t)
    t = re.sub(r"\\sqrt\s*\(", "sqrt(", t)
    t = re.sub(r"\\sqrt\s*([A-Za-z0-9])", r"sqrt(\1)", t)
    # Iterated log and inverse Ackermann, both of which stay below ~5 for any
    # representable input. Rename before the generic sweep eats the backslash.
    t = re.sub(r"\\?log\s*\^?\s*\*", " logstar ", t, flags=re.IGNORECASE)
    t = re.sub(r"\\alpha\s*\(", " ackinv(", t)
    # \log_2, \log_{2} -> log (base handled by convention, see module docstring)
    t = re.sub(r"\\?log\s*_\s*\{?\s*2\s*\}?", " log ", t, flags=re.IGNORECASE)
    t = re.sub(r"\\?log\s*_\s*\{?\s*e\s*\}?", " ln ", t, flags=re.IGNORECASE)
    # Math functions must become bare names before the generic sweep, or
    # "\log n" degrades into "n" and silently changes the complexity.
    for fn in ("log", "ln", "exp", "min", "max", "sqrt", "lg", "ceil", "floor"):
        t = re.sub(r"\\" + fn + r"\b", f" {fn} ", t)
    for name, repl in _GREEK.items():
        if len(name) == 1:
            t = t.replace(name, f" {repl} ")
        else:
            t = re.sub(r"\\" + name + r"\b", f" {repl} ", t)
    t = re.sub(r"\\[A-Za-z]+", " ", t)          # any remaining LaTeX command
    return t


def _normalize_subscripts(text: str) -> str:
    """``n_{max}`` is one variable, but brace-to-paren rewriting would turn it
    into the function call ``n_(max)``. Fold subscripts into the name."""
    text = re.sub(r"([A-Za-z])\s*_\s*\{([A-Za-z0-9]+)\}", r"\1_\2", text)
    return re.sub(r"([A-Za-z])\s*_\s*([A-Za-z0-9]+)", r"\1_\2", text)


_LOGSTAR_SENTINEL = "\x01"


def _separate_run_together_names(text: str) -> str:
    """``O(logn)`` and ``O(nlogn)`` mean ``log n`` and ``n log n``; without a
    space they parse as one nonsense variable."""
    # "logstar" must be shielded, or the very first rule splits it into
    # "log star" and invents a free variable named star.
    text = text.replace("logstar", _LOGSTAR_SENTINEL)
    for _ in range(4):
        new = re.sub(r"(?<![A-Za-z_])(log|ln)(?=[A-Za-z])(?!og\b)", r"\1 ", text)
        new = re.sub(r"([A-Za-z0-9])(log|ln)(?![A-Za-z_])", r"\1 \2 ", new)
        if new == text:
            break
        text = new
    return text.replace(_LOGSTAR_SENTINEL, "logstar")


_KNOWN_FUNCTIONS = ("log", "ln", "sqrt", "exp", "min", "max", "Min", "Max",
                    "logstar", "ackinv", "ceil", "ceiling", "floor")


def _atom_end(text: str, i: int) -> int:
    """Index just past the atom starting at ``i``: a function call, a name, or
    a number, plus any trailing ``^exponent``."""
    call = re.match(r"(?:logstar|log|ln|sqrt|exp)\s*\(", text[i:])
    if call:
        close = _matching_paren(text, i + call.end() - 1)
        if close == -1:
            return i
        end = close + 1
    else:
        atom = re.match(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+(?:\.[0-9]+)?", text[i:])
        if not atom:
            return i
        end = i + atom.end()
    exponent = re.match(r"\s*\^\s*(?:\([^()]*\)|[0-9]+(?:\.[0-9]+)?)", text[end:])
    return end + exponent.end() if exponent else end


def _parenthesize_bare_log_args(text: str) -> str:
    """``log n`` and ``log log n`` need explicit arguments.

    Left implicit, ``E log V (log log V)^3`` is ambiguous to the parser.
    Matches are resolved right to left so the inner ``log`` already carries
    its parentheses by the time the outer one is rewritten.
    """
    for _ in range(12):
        matches = list(re.finditer(r"(?<![A-Za-z_])(logstar|log|ln)(?![A-Za-z_0-9])", text))
        for m in reversed(matches):
            j = m.end()
            while j < len(text) and text[j] == " ":
                j += 1
            if j >= len(text) or text[j] == "(":
                continue
            end = _atom_end(text, j)
            if end <= j:
                continue
            text = text[:m.end()] + "(" + text[j:end] + ")" + text[end:]
            break
        else:
            return text
    return text


def _make_multiplication_explicit(text: str) -> str:
    """``E log(V) (log(log(V)))^3`` multiplies by the bracketed group, but
    SymPy's implicit-application rule reads ``V (...)`` as calling V.
    Inserting the ``*`` for every non-function name removes the ambiguity."""
    guard = r"(?!(?:" + "|".join(_KNOWN_FUNCTIONS) + r")\s*\()"
    text = re.sub(r"(?<![A-Za-z_])" + guard + r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", r"\1*(", text)
    return re.sub(r"\)\s*\(", ")*(", text)


def _resolve_little_o_terms(text: str) -> tuple[str, list[str]]:
    """``(1 + o(1))`` and ``exponent + o(1)`` are asymptotic slack, not values.

    Dropping the ``o(1)`` gives the leading-order runtime, which is what the
    curators intended these cells to convey.
    """
    if not re.search(r"\bo\s*\(\s*1\s*\)", text):
        return text, []
    out = re.sub(r"[+\-]\s*o\s*\(\s*1\s*\)", "", text)
    out = re.sub(r"\bo\s*\(\s*1\s*\)\s*[+\-]", "", out)
    out = re.sub(r"\bo\s*\(\s*1\s*\)", "0", out)
    return out, ["dropped o(1) slack terms, keeping the leading-order runtime"]


def _strip_bound_notation(text: str) -> tuple[str, list[str]]:
    """Peel off O(...) / Theta(...) / soft-O wrappers, returning the argument."""
    notes: list[str] = []
    t = text.strip()
    for _ in range(4):                          # nested wrappers, e.g. O(O(n))
        t = t.strip()
        m = re.match(r"^(O~|~O|O\^\*|O\*|\u00d5|\u00f5|tilde\s*O|O)\s*\(", t, re.IGNORECASE)
        if m and _matching_paren(t, m.end() - 1) == len(t) - 1:
            if m.group(1).lower() not in ("o",):
                notes.append("soft-O: suppressed polylog/polynomial factors ignored")
            t = t[m.end():-1]
            continue
        m = re.match(r"^(theta_?|Theta)\s*\(", t, re.IGNORECASE)
        if m and _matching_paren(t, m.end() - 1) == len(t) - 1:
            t = t[m.end():-1]
            continue
        m = re.match(r"^o\s*\(", t)
        if m and _matching_paren(t, m.end() - 1) == len(t) - 1:
            notes.append("little-o read as its argument (upper bound)")
            t = t[m.end():-1]
            continue
        break
    return t.strip(), notes


def _matching_paren(text: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _rewrite_log_powers(text: str) -> str:
    """``log^2 n`` and ``log^{2}{n}`` mean ``(log n)^2``.

    Written naively this parses as ``log ** 2 * n``, which is meaningless,
    so the exponent has to be moved outside an explicit function call.
    """
    # The exponent may be braced, parenthesized or bare: log^2 n, log^{3/2}(n),
    # and (after brace rewriting) log^(3/2)(n) all occur in the workbook.
    pattern = re.compile(
        r"\b(log|ln|sqrt)\s*\^\s*[({]?\s*([0-9]+(?:\.[0-9]+)?(?:\s*/\s*[0-9]+)?)\s*[)}]?"
    )
    while True:
        m = pattern.search(text)
        if not m:
            return text
        fn, exponent = m.group(1), m.group(2).replace(" ", "")
        rest = text[m.end():]
        stripped = rest.lstrip()
        pad = len(rest) - len(stripped)
        if stripped.startswith("(") or stripped.startswith("{"):
            close = "(" if stripped.startswith("(") else "{"
            end = _matching_paren(stripped.replace("{", "(").replace("}", ")"), 0)
            if end == -1:
                return text[:m.start()] + f"{fn}(n)**({exponent})" + rest
            arg = stripped[1:end]
            tail = stripped[end + 1:]
        else:
            am = re.match(r"[A-Za-z0-9_.]+", stripped)
            if not am:
                return text[:m.start()] + f"({fn}(n))**({exponent})" + rest
            arg, tail = am.group(0), stripped[am.end():]
        text = text[:m.start()] + f"({fn}({arg}))**({exponent})" + " " * pad + tail


def _fold_constant_division_into_exponent(text: str) -> tuple[str, list[str]]:
    """Inside big-O, ``n^5/2`` is ``n^(5/2)``, not half of ``n^5``.

    Only fires when the divisor is a bare integer literal, because dividing
    by a constant inside asymptotic notation is a no-op -- so the curator
    cannot have meant it -- while ``n^3/2^{\\sqrt{\\log n}}``, ``4^n/n^{3/2}``
    and ``n^2/p`` all divide by something that grows and are left alone.
    """
    pattern = re.compile(r"(?<![A-Za-z0-9_)])([A-Za-z][A-Za-z0-9_]*)"
                         r"\^\s*([0-9]+)\s*/\s*([0-9]+)(?![0-9.^])")
    out, n = pattern.subn(r"\1^(\2/\3)", text)
    if n:
        return out, [f"read the exponent in '{text.strip()}' as a fraction "
                     "(a constant divisor is meaningless inside big-O)"]
    return text, []


def clean_formula(raw) -> tuple[str, str, list[str]]:
    """Normalize one complexity cell.

    Returns ``(cleaned_expression, metric, notes)`` where metric is
    ``"time"`` or ``"work"``.  Raises :class:`FormulaError` if the cell has
    no single determinate runtime.
    """
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        raise FormulaError("no formula recorded")
    text = str(raw).strip()
    if text.lower() in _NULL_TOKENS:
        raise FormulaError("no formula recorded")

    notes: list[str] = []

    # "O(poly(n))\nmore specifically: O(n^2 log n)" -- the curator's refinement
    # after the colon is the usable formula, so prefer it.
    m = re.search(r"more specifically\s*:?\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        text = m.group(1).strip()
        notes.append("used the curator's 'more specifically' refinement")

    metric = "time"
    lm = _LABEL_RE.match(text)
    if lm:
        label = lm.group(1).strip().lower()
        metric = "work" if label in _WORK_LABELS else "time"
        text = text[lm.end():].strip()

    # Many cells wrap the formula in LaTeX math delimiters and then trail off
    # into prose ("$O(n \\log n)$ with binary tree", "$O(\\log n)$? (originally
    # this had $O(n)$)"). The first delimited group is the formula.
    math_groups = re.findall(r"\$([^$]+)\$", text)
    if math_groups:
        text = math_groups[0].strip()
    else:
        text = text.split("\n")[0].strip()

    # Named-notation definitions ("L_n[1/3,c] = exp(...)"): the formula is the
    # right-hand side.
    if "=" in text and not re.search(r"[<>!]=", text):
        text = text.rsplit("=", 1)[1].strip()

    # Trailing prose after a comma ("O(n^2), under the assumption that ...")
    text = re.split(r",\s*(?:under|where|assuming|for|with|if|when|and|approximately)\b",
                    text, flags=re.IGNORECASE)[0].strip()
    text = text.rstrip("?").strip()

    lowered = text.lower()
    for pattern, reason in _REJECT_PATTERNS:
        if re.search(pattern, lowered):
            raise FormulaError(reason)

    text = _strip_latex(text)
    # Re-check now that "\text{poly}(n)" has collapsed to "poly (n)".
    for pattern, reason in _REJECT_PATTERNS:
        if re.search(pattern, text.lower()):
            raise FormulaError(reason)
    text = _normalize_subscripts(text)
    text = text.replace("{", "(").replace("}", ")")
    text, o1_notes = _resolve_little_o_terms(text)
    notes.extend(o1_notes)
    text = _separate_run_together_names(text)
    text = re.sub(r"\s+", " ", text).strip().rstrip("*+-/ ").strip()

    # Unbalanced parentheses are a common data-entry slip. Repair them *before*
    # peeling the O(...) wrapper, otherwise the wrapper looks unclosed, survives
    # into SymPy, and silently becomes a symbolic Order object.
    opens, closes = text.count("("), text.count(")")
    if opens > closes:
        text += ")" * (opens - closes)
        notes.append("closed unbalanced parentheses in the source cell")
    elif closes > opens:
        raise FormulaError("unbalanced parentheses in the source cell")

    before_wrapper = text
    text, bound_notes = _strip_bound_notation(text)
    notes.extend(bound_notes)
    if text != before_wrapper:
        text, fold_notes = _fold_constant_division_into_exponent(text)
        notes.extend(fold_notes)

    # A second label can hide behind the wrapper, e.g. "O(n), CNOT Gates: O(n^2)"
    text = re.split(r",\s*[A-Za-z][A-Za-z ]{2,}:", text)[0].strip()

    if re.search(r"\bO\s*\(", text):
        raise FormulaError("nested asymptotic notation has no determinate value")

    # These three run last, and in this order: exponents first so that
    # "log^2 n" is already a power, then bare arguments, and only then the
    # explicit "*" -- which must not see an O( wrapper or it would mangle it.
    text = _rewrite_log_powers(text)
    text = _parenthesize_bare_log_args(text)
    text = _make_multiplication_explicit(text)
    text = re.sub(r"\s+", " ", text).strip().rstrip("*+-/ ").strip()

    if not text or text.lower() in _NULL_TOKENS:
        raise FormulaError("no formula recorded")
    if "!" in text:
        raise FormulaError("factorial runtimes are not evaluated")

    opens, closes = text.count("("), text.count(")")
    if opens > closes:
        text += ")" * (opens - closes)
    elif closes > opens:
        raise FormulaError("unbalanced parentheses in the source cell")

    return text, metric, notes


# ---------------------------------------------------------------------------
# Step 2 -- parsing
# ---------------------------------------------------------------------------

LOG2 = sympy.Function("log2_")
LOGSTAR = sympy.Function("logstar_")
ACKINV = sympy.Function("ackinv_")

_LOCALS = {
    "sqrt": sympy.sqrt,
    "log": LOG2,
    "lg": LOG2,
    "ln": sympy.log,
    "exp": sympy.exp,
    "logstar": LOGSTAR,
    "ackinv": ACKINV,
    "ceil": sympy.ceiling,
    "ceiling": sympy.ceiling,
    "floor": sympy.floor,
    "min": sympy.Min,
    "max": sympy.Max,
    "Min": sympy.Min,
    "Max": sympy.Max,
    "e": sympy.E,
    "I": sympy.Symbol("I_"),     # keep "I" a plain variable, not sqrt(-1)
    "E": sympy.Symbol("E"),      # "E" means edge count here, not Euler's number
    "N": sympy.Symbol("N"),
    "S": sympy.Symbol("S"),
    "Q": sympy.Symbol("Q"),
    "beta": sympy.Symbol("beta"),
    "gamma": sympy.Symbol("gamma"),
    "zeta": sympy.Symbol("zeta"),
}


def _naturalize_logs_in_exponents(expr: sympy.Expr) -> sympy.Expr:
    """Inside ``exp(...)``, read a bare ``log`` as the natural log.

    Sub-exponential runtimes are quoted from the L-notation literature, where
    ``L_N[a,c] = exp(c (ln N)^a (ln ln N)^(1-a))``.  The curators write those
    inner logs as ``log``, and everywhere else in the workbook ``log`` is base
    2 -- but under an exponential the base is not a constant factor.  Reading
    the number field sieve's ``exp((64n/9)^(1/3)(log n)^(2/3))`` in base 2
    inflates it by eight orders of magnitude at n = 1024 bits.
    """
    return expr.replace(
        lambda e: isinstance(e, sympy.exp),
        lambda e: sympy.exp(e.args[0].replace(LOG2, lambda a: sympy.log(a))),
    )


def parse_formula(raw) -> ParsedFormula:
    """Clean and parse one complexity cell into a :class:`ParsedFormula`."""
    cleaned, metric, notes = clean_formula(raw)
    try:
        expr = parse_expr(cleaned, local_dict=dict(_LOCALS),
                          transformations=_TRANSFORMS, evaluate=True)
    except Exception as exc:                     # noqa: BLE001 - report and skip
        raise FormulaError(f"could not parse {cleaned!r} ({type(exc).__name__})") from exc

    # A comma-separated cell parses to a tuple; that is two runtimes, not one.
    if not isinstance(expr, sympy.Expr):
        raise FormulaError(f"{cleaned!r} did not parse to a single expression")
    if expr.has(sympy.zoo) or expr.has(sympy.nan):
        raise FormulaError(f"degenerate expression from {cleaned!r}")

    naturalized = _naturalize_logs_in_exponents(expr)
    if naturalized != expr:
        notes.append("read log as natural log inside the exponential "
                     "(L-notation convention)")
        expr = naturalized

    symbols = frozenset(str(s) for s in expr.free_symbols)
    return ParsedFormula(source=str(raw), cleaned=cleaned, expr=expr,
                         symbols=symbols, metric=metric, notes=tuple(notes))


# ---------------------------------------------------------------------------
# Step 3 -- numeric evaluation, in log10 space
# ---------------------------------------------------------------------------

_LOG10_2 = math.log10(2.0)
_MAX_EXP = 1e12          # refuse absurd exponents rather than return inf

# Inverse Ackermann is at most 4 for every input that fits in the universe,
# so it is treated as that constant rather than left unevaluated.
_ACKERMANN_INV = 4


def _iterated_log(x: float) -> float:
    """log*(x): how many times log2 must be applied to bring x down to 1.
    Reaches only 5 at x = 2^65536, so it is effectively a small constant."""
    count = 0
    while x > 1.0 and count < 10:
        x = math.log2(x)
        count += 1
    return float(count)


def _plain_value(expr: sympy.Expr, symbol: str, value: float) -> float:
    """Direct float evaluation, for sub-expressions known to stay small
    (exponents, log arguments). Raises on overflow."""
    # Substitute the problem size before touching log2_, so the substitution
    # for that undefined function lands on concrete numbers and SymPy never
    # tries to series-expand anything.
    result = expr.subs(sympy.Symbol(symbol), sympy.Float(value))
    result = result.replace(LOG2, lambda arg: sympy.log(arg) / sympy.log(2))
    result = result.replace(LOGSTAR, lambda arg: sympy.Float(_iterated_log(float(sympy.N(arg)))))
    result = result.replace(ACKINV, lambda arg: sympy.Integer(_ACKERMANN_INV))
    try:
        evaluated = sympy.N(result, 30)
        if evaluated.free_symbols or not evaluated.is_real:
            raise FormulaError(f"{expr} did not reduce to a real number")
        out = float(evaluated)
    except FormulaError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise FormulaError(f"could not evaluate {expr}") from exc
    if not math.isfinite(out):
        raise FormulaError(f"non-finite value from {expr}")
    return out


def log10_runtime(expr: sympy.Expr, symbol: str, value: float) -> float:
    """log10 of ``expr`` evaluated at ``symbol = value``.

    Recurses over the expression tree so that exponentials never have to be
    materialized: log10(2**(n/2)) is computed as (n/2)*log10(2), not by
    building a 150-million-digit integer.
    """
    if value <= 0:
        raise FormulaError("problem size must be positive")
    sym = sympy.Symbol(symbol)

    def rec(e: sympy.Expr) -> float:
        if e.is_Number:
            v = float(e)
            if v <= 0:
                raise FormulaError("non-positive constant factor")
            return math.log10(v)
        if e == sym:
            return math.log10(value)
        if e.is_Symbol:
            raise FormulaError(f"free symbol {e} has no value")

        if isinstance(e, sympy.Mul):
            return sum(rec(a) for a in e.args)

        if isinstance(e, sympy.Pow):
            base, exponent = e.args
            # Exponents stay small even when the runtime does not, so they are
            # safe to evaluate directly. They are not always plain numbers:
            # "n^log_2(3)" yields the constant log(3)/log(2).
            exp_val = _plain_value(exponent, symbol, value)
            if abs(exp_val) > _MAX_EXP:
                raise FormulaError("exponent too large to evaluate")
            return exp_val * rec(base)

        if isinstance(e, sympy.Add):
            # log10 of a sum, via log-sum-exp on the positive terms. Negative
            # terms only shrink the total, and every complexity formula here
            # is dominated by its positive part, so dropping them is safe and
            # keeps the result an upper bound.
            parts = []
            for a in e.args:
                try:
                    parts.append(rec(a))
                except FormulaError:
                    continue
            if not parts:
                raise FormulaError(f"could not evaluate sum {e}")
            top = max(parts)
            return top + math.log10(sum(10.0 ** (p - top) for p in parts))

        if isinstance(e, sympy.Min):
            return min(rec(a) for a in e.args)
        if isinstance(e, sympy.Max):
            return max(rec(a) for a in e.args)

        if isinstance(e, LOG2):
            inner = _plain_value(e.args[0], symbol, value)
            if inner <= 1:
                return 0.0                     # log2 of a tiny argument -> O(1)
            return math.log10(math.log2(inner))
        if isinstance(e, LOGSTAR):
            return math.log10(max(_iterated_log(_plain_value(e.args[0], symbol, value)), 1.0))
        if isinstance(e, ACKINV):
            return math.log10(_ACKERMANN_INV)
        if isinstance(e, (sympy.ceiling, sympy.floor)):
            return rec(e.args[0])
        if isinstance(e, sympy.log):
            inner = _plain_value(e.args[0], symbol, value)
            if inner <= 1:
                return 0.0
            return math.log10(math.log(inner))
        if isinstance(e, sympy.exp):
            return _plain_value(e.args[0], symbol, value) * math.log10(math.e)

        raise FormulaError(f"unsupported construct {type(e).__name__} in {e}")

    out = rec(sympy.sympify(expr))
    if not math.isfinite(out):
        raise FormulaError("non-finite runtime")
    return out
