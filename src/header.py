"""Global configuration for the quantum-algorithms analysis pipeline.

Modeled after the `header.py` used in dtontici/parallel-algorithms: every plot
script imports from here, and styling / paths / dataset version live in one
spot so the rest of the code stays focused on analysis.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PLOTS_DIR = REPO_ROOT / "Plots"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Source spreadsheet (the AlgoWiki workbook that lives in data/)
XLSX_PATH = DATA_DIR / "AlgoWiki algorithms (our copy) (3).xlsx"

# Sheets we care about
QUANTUM_SHEET = "Quantum Algorithms"
PROBLEMS_SHEET = "Problems"

# Cached, cleaned outputs (populated by src/data_loader.py)
VERSION = "_v1"
QUANTUM_JSON = PROCESSED_DIR / f"quantum_algos{VERSION}.json"
QUANTUM_CSV = PROCESSED_DIR / f"quantum_algos{VERSION}.csv"
PROBLEMS_JSON = PROCESSED_DIR / f"problems{VERSION}.json"
PROBLEMS_CSV = PROCESSED_DIR / f"problems{VERSION}.csv"

# ---------------------------------------------------------------------------
# Year window + decade buckets
# ---------------------------------------------------------------------------
MIN_YEAR = 1980
CUR_YEAR = 2026  # update annually

# Mirrors `g_decades` in parallel-algorithms/src/thesis_plots/decade_progress.py.
# Each entry is the *upper bound* of that decade and a short label for the
# x-axis tick. Years <= "max" fall into that bucket.
DECADES = [
    {"max": 1990, "label": "80s"},
    {"max": 2000, "label": "90s"},
    {"max": 2010, "label": "00s"},
    {"max": 2020, "label": "10s"},
    {"max": 2030, "label": "20s"},
]

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
# We support two named styles, each writing into its own subfolder so the two
# can be compared side by side:
#   * "parallel" - matches dtontici/parallel-algorithms: compact 6.55 x 3.5
#                  figure at 200 dpi, pastel SEQ_PAR_COLORS palette, clean
#                  white background, horizontal multi-line ylabel, single
#                  axis (no cumulative).
#   * "classic"  - the original look from the first iteration of this repo:
#                  larger 10 x 5 figure at 150 dpi, distinct matplotlib
#                  primary colors, faint horizontal grid, cumulative line
#                  drawn on a twinned y-axis, vertical y-label.
# ---------------------------------------------------------------------------
EDGE_COLOR = "white"

# Each style writes into its own subfolder under Plots/. Classic filenames
# still carry a "_classic" suffix so the two sets stay distinguishable by name.
SAVE_LOC = PLOTS_DIR

# Pastel palette from parallel-algorithms' SEQ_PAR_COLORS for the parallel style
SEQ_PAR_COLORS = ["#F5C8AF", "#58D68D"]
PROCESSOR_COLORS = ["#3cb44b", "#ffe119", "#a9a9a9"]

STYLES = {
    "parallel": {
        "figsize": (6.55, 3.5),
        "dpi": 200,
        "grid_alpha": 0.0,
        "save_subdir": PLOTS_DIR / "parallel",
        "name_suffix": "",
        "show_cumulative": False,
        "horizontal_ylabel": True,
        "color_problems": "#F5C8AF",       # peach
        "color_improvements": "#58D68D",   # green
        "color_families": "#58D68D",       # green
    },
    "classic": {
        "figsize": (10, 5),
        "dpi": 150,
        "grid_alpha": 0.25,
        "save_subdir": PLOTS_DIR / "classic",
        "name_suffix": "_classic",
        "show_cumulative": True,
        "horizontal_ylabel": False,
        "color_problems": "#1f77b4",       # blue
        "color_improvements": "#2ca02c",   # green
        "color_families": "#d62728",       # red
    },
}

# Backwards-compatible exports (parallel style is the default for any code
# that still imports the bare constants directly).
DEFAULT_STYLE = "parallel"
DPI = STYLES[DEFAULT_STYLE]["dpi"]
FIGSIZE = STYLES[DEFAULT_STYLE]["figsize"]
GRID_ALPHA = STYLES[DEFAULT_STYLE]["grid_alpha"]
COLOR_PROBLEMS = STYLES[DEFAULT_STYLE]["color_problems"]
COLOR_IMPROVEMENTS = STYLES[DEFAULT_STYLE]["color_improvements"]
COLOR_FAMILIES = STYLES[DEFAULT_STYLE]["color_families"]
COLOR_CUMULATIVE = "#7f7f7f"
