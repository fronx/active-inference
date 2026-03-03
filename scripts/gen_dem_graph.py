#!/usr/bin/env python3
"""
Generate a high-level DOT/Graphviz dataflow diagram for SPM DEM scripts like DEM_morphogenesis.m.

This is intentionally *semantic* rather than a full MATLAB AST renderer:
- It extracts a few key struct assignments (G(1).g, G(2).a, M(1).g, M(1).pE, M(2).v, etc.)
- It extracts function handle targets (e.g., observe/expect)
- It extracts local function definitions and a simple intra-file call list (best-effort)
- It emits a stable DOT diagram with clusters G and M and a black-box spm_ADEM node.

Usage:
  python scripts/gen_dem_graph.py path/to/DEM_morphogenesis.m -o dem_morphogenesis.dot
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class Assign:
    struct_name: str   # "G" or "M"
    level: int         # 1,2,...
    field: str         # "g", "a", "v", "pE", ...
    rhs: str           # right-hand side text (trimmed)


_ASSIGN_RE = re.compile(
    r"""(?mx)                       # multi-line, verbose
    ^\s*
    (?P<struct>[GM])                # G or M
    \(\s*(?P<level>\d+)\s*\)        # (1) or (2)
    \.\s*(?P<field>[A-Za-z]\w*)     # .g .v .a .pE ...
    \s*=\s*
    (?P<rhs>[^;]+)                  # everything up to semicolon
    \s*;
    """,
)

# Detect function handle RHS like: @(...) observe(...)
_FH_TARGET_RE = re.compile(
    r"""(?x)
    @\s*\([^)]*\)\s*                # @(args)
    (?P<target>[A-Za-z]\w*)\s*\(    # targetFunc(
    """
)

# Local function definition lines: function out = name(args)
_FUNC_DEF_RE = re.compile(
    r"""(?mx)
    ^\s*function\b
    [^\n=]*=\s*                     # allow outputs like "sense =" or "observed ="
    (?P<name>[A-Za-z]\w*)\s*\(      # function name(
    """
)

# Also handle "function name(args)" (no explicit output)
_FUNC_DEF_NOOUT_RE = re.compile(
    r"""(?mx)
    ^\s*function\b
    \s+(?P<name>[A-Za-z]\w*)\s*\(
    """
)

# Very lightweight call detection: name( ... ) not preceded by '.' and not keywords.
_CALL_RE = re.compile(r"(?m)(?<!\.)\\b([A-Za-z]\w*)\s*\(")
_KEYWORDS = {
    "if", "for", "while", "switch", "case", "otherwise", "return", "function",
    "end", "global", "clear", "rng", "subplot", "plot", "imagesc", "axis",
    "title", "xlabel", "ylabel", "set", "hold", "cla", "clf", "mkdir",
    "exist", "fprintf", "linspace", "ndgrid", "find", "sort", "max", "min",
    "sqrt", "exp", "kron", "eye", "ones", "zeros", "double", "full",
}


def strip_comments(matlab: str) -> str:
    """
    Remove MATLAB '%' comments (best-effort).
    Does not attempt to respect '%' inside strings.
    """
    out_lines = []
    for line in matlab.splitlines():
        # remove anything after %
        if "%" in line:
            line = line.split("%", 1)[0]
        out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def extract_assignments(matlab: str) -> List[Assign]:
    assigns: List[Assign] = []
    for m in _ASSIGN_RE.finditer(matlab):
        assigns.append(
            Assign(
                struct_name=m.group("struct"),
                level=int(m.group("level")),
                field=m.group("field"),
                rhs=" ".join(m.group("rhs").strip().split()),
            )
        )
    return assigns


def index_assignments(assigns: List[Assign]) -> Dict[Tuple[str, int, str], Assign]:
    return {(a.struct_name, a.level, a.field): a for a in assigns}


def extract_function_handle_target(rhs: str) -> Optional[str]:
    m = _FH_TARGET_RE.search(rhs)
    return m.group("target") if m else None


def extract_local_functions(matlab: str) -> Set[str]:
    names = set(m.group("name") for m in _FUNC_DEF_RE.finditer(matlab))
    names |= set(m.group("name") for m in _FUNC_DEF_NOOUT_RE.finditer(matlab))
    return names


def extract_calls(matlab: str, local_funcs: Set[str]) -> Set[Tuple[str, str]]:
    """
    Best-effort intra-file call edges between local functions.
    We do not build a proper call graph by function scope; we just find call tokens.
    For your use-case, it's mainly to detect that observe/expect call morphogenesis.
    """
    calls: Set[Tuple[str, str]] = set()
    tokens = _CALL_RE.findall(matlab)
    for name in tokens:
        if name in _KEYWORDS:
            continue
        # Track only local-to-local edges (e.g., observe -> morphogenesis) is hard without scope.
        # Here, we just record "GLOBAL" -> name for debugging, plus local callers if we can.
        # For now keep only callee list in footer; not used for DOT edges.
        # (You can extend this later with scope parsing if needed.)
        pass
    return calls


def generate_dot(
    src_path: Path,
    assigns_by_key: Dict[Tuple[str, int, str], Assign],
    fh_targets: Dict[Tuple[str, int, str], str],
    include_detection_footer: bool = True,
) -> str:
    """
    Emit the same semantic DOT structure as the manual version.
    Uses extracted info for labels where possible, but remains stable if extraction fails.
    """
    # Pull out key extracted items (best-effort)
    g1g = assigns_by_key.get(("G", 1, "g"))
    g2a = assigns_by_key.get(("G", 2, "a"))
    m1g = assigns_by_key.get(("M", 1, "g"))
    m1pE = assigns_by_key.get(("M", 1, "pE"))
    m2v = assigns_by_key.get(("M", 2, "v"))

    g1g_target = fh_targets.get(("G", 1, "g"))
    m1g_target = fh_targets.get(("M", 1, "g"))

    def rhs_excerpt(a: Optional[Assign], max_len: int = 60) -> str:
        if not a:
            return "N/A"
        s = a.rhs
        return s if len(s) <= max_len else (s[: max_len - 3] + "...")

    # Stable labels with optional detected RHS excerpts
    a_g_label = "a_G\\nG(2).a\\n(endogenous cause / action vector)"
    if g2a:
        a_g_label += f"\\n= {rhs_excerpt(g2a)}"

    g_g_label = "g_G\\nG(1).g\\n(observation mapping)"
    if g1g_target:
        g_g_label += f"\\n@(...) {g1g_target}(...)"
    elif g1g:
        g_g_label += f"\\n= {rhs_excerpt(g1g)}"

    v_m_label = "v_M\\nM(2).v\\n(hidden causes)"
    if m2v:
        v_m_label += f"\\n= {rhs_excerpt(m2v)}"

    p_label = "P\\nM(1).pE\\n(priors / prototypes)"
    if m1pE:
        p_label += f"\\n= {rhs_excerpt(m1pE)}"

    g_m_label = "g_M\\nM(1).g\\n(model mapping"
    if m1g_target:
        g_m_label += f"\\n@(...) {m1g_target}(...)"
    elif m1g:
        g_m_label += f"\\n= {rhs_excerpt(m1g)}"
    g_m_label += ")"

    dot = f"""digraph DEM_Morphogenesis_Dataflow {{
  rankdir=LR;
  graph [fontsize=12, fontname="Helvetica", labelloc="t",
         label="{src_path.name}: high-level active inference dataflow (spm_ADEM as black box)"];
  node  [shape=box, fontsize=11, fontname="Helvetica", style="rounded"];
  edge  [fontsize=10, fontname="Helvetica"];

  spm_ADEM [shape=box, style="rounded,filled", fillcolor="#f2f2f2",
            label="spm_ADEM(DEM)\\n(active inference / variational loop)\\nupdates qU.v, qU.a, etc."];

  subgraph cluster_G {{
    label="Generative Process (G)";
    style="rounded";
    color="#4e79a7";

    a_G [label="{a_g_label}"];
    g_G [label="{g_g_label}"];
    y   [label="y\\nobservations (process)\\nCellState {{position, signal, sense}}"];

    a_G -> g_G [label="action a(t)"];
    g_G -> y   [label="y(t)"];
  }}

  subgraph cluster_M {{
    label="Generative Model (M)";
    style="rounded";
    color="#59a14f";

    v_M [label="{v_m_label}"];
    P   [label="{p_label}"];
    g_M [label="{g_m_label}"];
    yhat [label="ŷ\\npredicted observations (model)\\nCellState {{position, signal, sense}}"];

    v_M -> g_M  [label="beliefs over causes"];
    P   -> g_M  [label="priors / prototypes"];
    g_M -> yhat [label="ŷ(t)"];
  }}

  y    -> spm_ADEM [label="sensory evidence y(t)"];
  yhat -> spm_ADEM [label="predictions ŷ(t)\\n→ prediction errors"];

  spm_ADEM -> v_M [label="inference update\\n(qU.v / posterior causes)"];
  spm_ADEM -> a_G [label="action update\\n(qU.a / posterior action)"];

  {{ rank=same; a_G; v_M; }}
  {{ rank=same; g_G; g_M; }}
  {{ rank=same; y; yhat; }}
}}
"""

    if include_detection_footer:
        dot += "\n/*\nDETECTED_ASSIGNMENTS (best-effort):\n"
        for key in sorted(assigns_by_key.keys()):
            a = assigns_by_key[key]
            dot += f"  {a.struct_name}({a.level}).{a.field} = {a.rhs}\n"
        dot += "\nDETECTED_FUNCTION_HANDLES:\n"
        for key in sorted(fh_targets.keys()):
            dot += f"  {key[0]}({key[1]}).{key[2]} -> {fh_targets[key]}\n"
        dot += "*/\n"

    return dot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("matlab_file", type=Path, help="Path to DEM_morphogenesis.m (or similar)")
    ap.add_argument("-o", "--out", type=Path, default=None, help="Output DOT path (default: stdout)")
    ap.add_argument("--no-footer", action="store_true", help="Do not include detected-info footer comments")
    args = ap.parse_args()

    src_path: Path = args.matlab_file
    raw = src_path.read_text(encoding="utf-8", errors="ignore")
    txt = strip_comments(raw)

    assigns = extract_assignments(txt)
    assigns_by_key = index_assignments(assigns)

    fh_targets: Dict[Tuple[str, int, str], str] = {}
    for a in assigns:
        if a.field == "g":  # only care about function handles for g fields here
            tgt = extract_function_handle_target(a.rhs)
            if tgt:
                fh_targets[(a.struct_name, a.level, a.field)] = tgt

    dot = generate_dot(
        src_path=src_path,
        assigns_by_key=assigns_by_key,
        fh_targets=fh_targets,
        include_detection_footer=not args.no_footer,
    )

    if args.out:
        args.out.write_text(dot, encoding="utf-8")
    else:
        print(dot)


if __name__ == "__main__":
    main()
