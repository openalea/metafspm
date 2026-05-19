from __future__ import annotations

import ast
import inspect
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import openalea.metafspm
import pyan


# mamba install -c conda-forge pyan3 graphviz python-graphviz

# ============================================================================
# Configuration
# ============================================================================

PACKAGE_NAME = openalea.metafspm.__name__
PACKAGE_DIR = Path(openalea.metafspm.__path__[0]).resolve()

# Output directory for GitHub Pages / docs
OUTDIR = Path("./output/")

# Optional subdirectories to ignore during docstring extraction
# Example: {"adrien", "legacy", "sandbox"}
SKIP_DIR_NAMES: set[str] = {"unused"}

# Optional file names to ignore
SKIP_FILE_NAMES: set[str] = set()

# Node labels to drop from the callgraph (applied to both node declarations
# and any edge that references them).  Use this for noisy dunder methods or
# ubiquitous helpers that clutter the graph.
SKIP_NODE_NAMES: set[str] = {"__iter__", "__next__", "__init__", "__repr__", "__str__"}

# Which pyan edge types to include.  Running both simultaneously causes graphviz
# to hang — the script works around this by calling pyan twice and merging edges.
DRAW_DEFINES: bool = True   # "X is defined inside Y" structural edges
DRAW_USES: bool = True      # "X calls / uses Y" call edges

# Truncate long docstrings for tooltip readability
MAX_DOC_CHARS = 900

# ---- Layout ----------------------------------------------------------------
# Engine — pick one depending on graph size:
#   "fdp"    force-directed with cluster support — required because pyan always
#            emits subgraph clusters; sfdp ignores clusters and hangs  (default)
#   "neato"  spring-model, good for ~100 nodes
#   "circo"  circular, useful for cycle-heavy graphs
#   "sfdp"   DO NOT USE — hangs on any graph that contains subgraph clusters
LAYOUT_ENGINE: str = "fdp"

# Edge drawing style:
#   "spline"   classic cubic splines, smooth and readable  (recommended)
#   "polyline" bent straight lines, faster than spline
#   "line"     straight lines, fastest to render
#   "none"     raw lines ignoring node boundaries (very fast, ugly)
# Note: "curved" hangs with fdp on clustered graphs — avoid it.
# Note: "ortho" (right-angle bends) only works with the "dot" engine.
SPLINES: str = "spline"

# Overlap removal after layout:
#   "prism"    iterative removal, best quality for large graphs  (recommended)
#   "false"    basic removal
#   "scale"    scale the graph up until nodes stop overlapping
#   "compress" try to reduce bounding box after removing overlaps
OVERLAP: str = "prism"

# Merge parallel edges that share the same endpoints — reduces visual clutter.
CONCENTRATE: bool = True

# Ideal spring length for fdp/sfdp.  Increase to spread nodes further apart.
SPRING_K: float = 1.5


# ============================================================================
# Helpers
# ============================================================================

def ensure_graphviz_available() -> None:
    """Raise a clear error if Graphviz 'dot' is not available."""
    dot = shutil.which("dot")
    if dot is None:
        raise RuntimeError(
            "Graphviz executable 'dot' was not found on PATH.\n"
            "Install Graphviz and ensure 'dot -V' works in your shell.\n"
            "For conda/mamba, try:\n"
            "  mamba install -c conda-forge graphviz python-graphviz"
        )


def pyan_node_id_from_dotted(dotted_name: str) -> str:
    """
    Convert a dotted Python name to the node identifier style typically used by pyan.

    Example:
        granap.organ_class.Organ.generate_cells
        -> granap__organ_class__Organ__generate_cells
    """
    return "__".join(dotted_name.split("."))


def module_name_from_file(py_file: Path) -> str:
    """
    Convert a Python file path under PACKAGE_DIR to its dotted module name.

    Example:
        src/granap/organ_class.py -> granap.organ_class
        src/granap/subpkg/__init__.py -> granap.subpkg
    """
    rel = py_file.relative_to(PACKAGE_DIR).with_suffix("")
    parts = [PACKAGE_NAME] + list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def clean_docstring(doc: str | None, max_chars: int = MAX_DOC_CHARS) -> str:
    """Normalize and shorten a docstring for tooltip display."""
    if not doc:
        return ""

    doc = inspect.cleandoc(doc).strip()
    if not doc:
        return ""

    # Prefer first paragraph for readability
    first_para = doc.split("\n\n", 1)[0].strip()
    text = first_para or doc

    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"

    return text


def should_skip_file(py_file: Path) -> bool:
    """Return True if the file should be ignored during docstring extraction."""
    if py_file.name in SKIP_FILE_NAMES:
        return True
    if any(part in SKIP_DIR_NAMES for part in py_file.parts):
        return True
    return False


# ============================================================================
# AST docstring extraction
# ============================================================================

class DocCollector(ast.NodeVisitor):
    """
    Collect docstrings from classes and functions/methods.

    We store them under keys matching pyan-style node identifiers, so the
    generated SVG nodes can be matched in the browser for hover tooltips.
    """

    def __init__(self, module_name: str, out_map: dict[str, dict[str, str]]) -> None:
        self.module_name = module_name
        self.stack: list[str] = []
        self.out_map = out_map

    def _store(self, dotted_name: str, kind: str, doc: str | None) -> None:
        cleaned = clean_docstring(doc)
        if not cleaned:
            return

        self.out_map[pyan_node_id_from_dotted(dotted_name)] = {
            "name": dotted_name,
            "kind": kind,
            "doc": cleaned,
        }

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        dotted = ".".join([self.module_name, *self.stack, node.name])
        self._store(dotted, "class", ast.get_docstring(node))

        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        dotted = ".".join([self.module_name, *self.stack, node.name])
        kind = "method" if self.stack else "function"
        self._store(dotted, kind, ast.get_docstring(node))

        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        dotted = ".".join([self.module_name, *self.stack, node.name])
        kind = "method" if self.stack else "function"
        self._store(dotted, kind, ast.get_docstring(node))

        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()


def collect_docstrings(package_dir: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    """
    Parse all Python files under the package and collect docstrings.

    Returns
    -------
    docs_map:
        Mapping from pyan node id -> {name, kind, doc}
    skipped:
        Human-readable messages for files that could not be parsed
        (syntax error, indentation error, etc.)
    """
    docs: dict[str, dict[str, str]] = {}
    skipped: list[str] = []

    for py_file in sorted(package_dir.rglob("*.py")):
        if should_skip_file(py_file):
            continue

        try:
            source = py_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            source = py_file.read_text(encoding="latin1")

        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError as e:
            skipped.append(f"{py_file} (line {e.lineno}): {e.msg}")
            continue

        module_name = module_name_from_file(py_file)

        # Module docstring
        module_doc = clean_docstring(ast.get_docstring(tree))
        if module_doc:
            docs[pyan_node_id_from_dotted(module_name)] = {
                "name": module_name,
                "kind": "module",
                "doc": module_doc,
            }

        collector = DocCollector(module_name, docs)
        collector.visit(tree)

    return docs, skipped


# ============================================================================
# Graph generation
# ============================================================================

# DOT node IDs use double-underscore separators (pyan convention)
_INTERNAL_PREFIX = PACKAGE_NAME.replace(".", "__")
_EDGE_RE = re.compile(r"^\s*(\w+)\s*->\s*(\w+)")


def filter_dot_external_nodes(dot_source: str) -> str:
    """Drop external edges, banned nodes, and layout attributes that override the engine.

    pyan embeds ``layout=dot`` and ``clusterrank="local"`` in the graph header,
    which forces graphviz back to the slow hierarchical engine even when sfdp is
    called.  We strip those attributes, drop edges to/from external packages, and
    remove any node whose label is listed in SKIP_NODE_NAMES.
    """
    # Remove layout=dot and clusterrank so the chosen engine (sfdp) is not overridden.
    dot_source = re.sub(r',?\s*layout\s*=\s*\w+', '', dot_source)
    dot_source = re.sub(r',?\s*clusterrank\s*=\s*"[^"]*"', '', dot_source)

    # Node declaration:  "some__id" [label="short_name", ...]
    node_decl_re = re.compile(r'^\s*"([^"]+)"\s*\[.*\blabel\s*=\s*"([^"]*)"')
    # Edge line:  "some__id" -> "other__id" [...]
    edge_re = re.compile(r'^\s*"([^"]+)"\s*->\s*"([^"]+)"')

    # Pass 1: collect node IDs whose label is in SKIP_NODE_NAMES.
    banned_ids: set[str] = set()
    for line in dot_source.splitlines():
        m = node_decl_re.match(line)
        if m and m.group(2) in SKIP_NODE_NAMES:
            banned_ids.add(m.group(1))

    # Pass 2: drop banned node declarations and any edge touching a banned or external node.
    out = []
    for line in dot_source.splitlines(keepends=True):
        m_decl = node_decl_re.match(line)
        if m_decl:
            if m_decl.group(1) in banned_ids:
                continue
            out.append(line)
            continue

        m_edge = edge_re.match(line)
        if m_edge:
            src, dst = m_edge.group(1), m_edge.group(2)
            if (src.startswith(_INTERNAL_PREFIX) and dst.startswith(_INTERNAL_PREFIX)
                    and src not in banned_ids and dst not in banned_ids):
                out.append(line)
            continue

        out.append(line)
    return "".join(out)


_EDGE_LINE_RE = re.compile(r'^\s*"[^"]+"\s*->\s*"[^"]+"')


def _pyan_call(*, draw_defines: bool, draw_uses: bool) -> str:
    return pyan.create_callgraph(
        filenames=[str(f) for f in PACKAGE_DIR.rglob("*.py")],
        format="dot",
        colored=True,
        nested_groups=True,
        draw_defines=draw_defines,
        draw_uses=draw_uses,
        # root must point to src/ so pyan resolves openalea.metafspm.* correctly
        root=str(PACKAGE_DIR.parent.parent),
    )


def build_dot_source() -> str:
    """Generate filtered DOT callgraph from pyan.

    When both DRAW_DEFINES and DRAW_USES are requested pyan is called twice and
    the edge sets are merged.  Running both in a single pyan call produces a
    combined graph that causes graphviz to hang.
    """
    if DRAW_DEFINES and DRAW_USES:
        # Use defines as the base (it carries all node/cluster declarations).
        base = filter_dot_external_nodes(_pyan_call(draw_defines=True, draw_uses=False))
        uses = filter_dot_external_nodes(_pyan_call(draw_defines=False, draw_uses=True))
        # Extract only edge lines from the uses graph and inject before closing '}'.
        extra_edges = [l for l in uses.splitlines(keepends=True) if _EDGE_LINE_RE.match(l)]
        return base.rstrip().rstrip("}").rstrip() + "\n" + "".join(extra_edges) + "\n}\n"

    return filter_dot_external_nodes(
        _pyan_call(draw_defines=DRAW_DEFINES, draw_uses=DRAW_USES)
    )


GRAPHVIZ_TIMEOUT = 120  # seconds before we give up and raise TimeoutExpired


def render_svg_from_dot(dot_source: str) -> str:
    """Render DOT to SVG using the configured Graphviz engine and layout options."""
    cmd = [
        LAYOUT_ENGINE, "-Tsvg",
        f"-Gsplines={SPLINES}",
        f"-Goverlap={OVERLAP}",
        f"-Gconcentrate={'true' if CONCENTRATE else 'false'}",
        f"-GK={SPRING_K}",
    ]
    result = subprocess.run(
        cmd,
        input=dot_source,
        text=True,
        capture_output=True,
        check=True,
        timeout=GRAPHVIZ_TIMEOUT,
    )
    return result.stdout


# ============================================================================
# HTML generation
# ============================================================================

def build_html(svg: str, docs_map: dict[str, dict[str, str]]) -> str:
    """Return a self-contained interactive HTML page."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{PACKAGE_NAME} call graph</title>
  <style>
    html, body {{
      margin: 0;
      height: 100%;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #ffffff;
    }}

    #topbar {{
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      z-index: 20;
      background: rgba(255,255,255,0.96);
      border-bottom: 1px solid #ddd;
      padding: 8px 12px;
      font-size: 14px;
      line-height: 1.4;
    }}

    #graph-wrap {{
      position: absolute;
      top: 52px;
      left: 0;
      right: 0;
      bottom: 0;
      overflow: auto;
      background: #fafafa;
    }}

    #graph {{
      display: inline-block;
      transform-origin: 0 0;
    }}

    .hint {{
      color: #666;
      margin-left: 10px;
    }}

    .dim {{
      opacity: 0.12;
      transition: opacity 120ms ease-in-out;
    }}

    .focus {{
      opacity: 1 !important;
    }}

    .node {{
      cursor: pointer;
    }}

    #tooltip {{
      position: fixed;
      z-index: 1000;
      max-width: 560px;
      min-width: 280px;
      display: none;
      background: rgba(30, 30, 30, 0.97);
      color: white;
      border-radius: 8px;
      padding: 10px 12px;
      box-shadow: 0 8px 30px rgba(0,0,0,0.28);
      font-size: 13px;
      line-height: 1.45;
      pointer-events: none;
      white-space: pre-wrap;
    }}

    #tooltip .name {{
      font-weight: 700;
      margin-bottom: 4px;
    }}

    #tooltip .kind {{
      font-size: 11px;
      opacity: 0.75;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 8px;
    }}

    #tooltip .doc {{
      opacity: 0.96;
    }}
  </style>
</head>
<body>
  <div id="topbar">
    <strong>{PACKAGE_NAME} call graph</strong>
    <span class="hint">Click a node to highlight dependencies · Hover to see docstrings · Shift+wheel to zoom · Esc to reset</span>
  </div>

  <div id="graph-wrap">
    <div id="graph"></div>
  </div>

  <div id="tooltip">
    <div class="name"></div>
    <div class="kind"></div>
    <div class="doc"></div>
  </div>

  <script>
    const svgText = {json.dumps(svg)};
    const docsMap = {json.dumps(docs_map)};

    const graphDiv = document.getElementById("graph");
    const wrap = document.getElementById("graph-wrap");
    const tooltip = document.getElementById("tooltip");
    const tooltipName = tooltip.querySelector(".name");
    const tooltipKind = tooltip.querySelector(".kind");
    const tooltipDoc = tooltip.querySelector(".doc");

    graphDiv.innerHTML = svgText;

    const svgEl = graphDiv.querySelector("svg");
    svgEl.style.display = "block";

    // Reorder SVG groups so edges render behind nodes.
    // SVG draws in document order; we move all edge <g> elements before the
    // first node <g> so arrowheads don't overdraw node labels.
    (function reorderLayers() {{
      const graphG = svgEl.querySelector("g.graph") || svgEl.firstElementChild;
      if (!graphG) return;
      const firstNode = graphG.querySelector(":scope > g.node");
      if (!firstNode) return;
      graphG.querySelectorAll(":scope > g.edge").forEach(e => graphG.insertBefore(e, firstNode));
    }})();

    let scale = 1.0;

    const nodeEls = Array.from(graphDiv.querySelectorAll("g.node"));
    const edgeEls = Array.from(graphDiv.querySelectorAll("g.edge"));

    function getTitleText(el) {{
      const t = el.querySelector("title");
      return t ? t.textContent.trim() : null;
    }}

    const nodesByKey = new Map();
    for (const node of nodeEls) {{
      const key = getTitleText(node);
      if (key) {{
        nodesByKey.set(key, node);
      }}
    }}

    const outgoing = new Map();
    const incoming = new Map();
    const edges = [];

    function addMapSet(map, key, value) {{
      if (!map.has(key)) {{
        map.set(key, new Set());
      }}
      map.get(key).add(value);
    }}

    for (const edge of edgeEls) {{
      const text = getTitleText(edge);
      if (!text || !text.includes("->")) {{
        continue;
      }}

      const parts = text.split("->").map(s => s.trim());
      if (parts.length !== 2) {{
        continue;
      }}

      const [src, dst] = parts;
      const edgeKey = `${{src}}->${{dst}}`;

      edges.push({{
        el: edge,
        src: src,
        dst: dst,
        key: edgeKey
      }});

      addMapSet(outgoing, src, dst);
      addMapSet(incoming, dst, src);
    }}

    function walk(start, adjacency) {{
      const seen = new Set();
      const stack = [start];

      while (stack.length) {{
        const cur = stack.pop();
        const nexts = adjacency.get(cur);
        if (!nexts) continue;

        for (const nxt of nexts) {{
          if (!seen.has(nxt)) {{
            seen.add(nxt);
            stack.push(nxt);
          }}
        }}
      }}

      return seen;
    }}

    function resetHighlight() {{
      for (const n of nodeEls) {{
        n.classList.remove("dim", "focus");
      }}
      for (const e of edgeEls) {{
        e.classList.remove("dim", "focus");
      }}
    }}

    function highlightFrom(nodeKey) {{
      const upstream = walk(nodeKey, incoming);
      const downstream = walk(nodeKey, outgoing);
      const keepNodes = new Set([nodeKey, ...upstream, ...downstream]);
      const keepEdges = new Set();

      for (const e of edges) {{
        if (keepNodes.has(e.src) && keepNodes.has(e.dst)) {{
          keepEdges.add(e.key);
        }}
      }}

      for (const [key, el] of nodesByKey.entries()) {{
        const on = keepNodes.has(key);
        el.classList.toggle("dim", !on);
        el.classList.toggle("focus", on);
      }}

      for (const e of edges) {{
        const on = keepEdges.has(e.key);
        e.el.classList.toggle("dim", !on);
        e.el.classList.toggle("focus", on);
      }}
    }}

    function moveTooltip(evt) {{
      const margin = 16;
      const x = evt.clientX + margin;
      const y = evt.clientY + margin;

      const maxX = Math.max(8, window.innerWidth - tooltip.offsetWidth - 8);
      const maxY = Math.max(8, window.innerHeight - tooltip.offsetHeight - 8);

      tooltip.style.left = Math.min(x, maxX) + "px";
      tooltip.style.top = Math.min(y, maxY) + "px";
    }}

    function showTooltip(evt, key) {{
      const info = docsMap[key];
      if (!info) {{
        hideTooltip();
        return;
      }}

      tooltipName.textContent = info.name;
      tooltipKind.textContent = info.kind;
      tooltipDoc.textContent = info.doc;
      tooltip.style.display = "block";
      moveTooltip(evt);
    }}

    function hideTooltip() {{
      tooltip.style.display = "none";
    }}

    for (const [key, el] of nodesByKey.entries()) {{
      el.addEventListener("click", (evt) => {{
        evt.stopPropagation();
        highlightFrom(key);
      }});

      el.addEventListener("mouseenter", (evt) => {{
        showTooltip(evt, key);
      }});

      el.addEventListener("mousemove", moveTooltip);
      el.addEventListener("mouseleave", hideTooltip);
    }}

    graphDiv.addEventListener("click", () => {{
      resetHighlight();
    }});

    document.addEventListener("keydown", (evt) => {{
      if (evt.key === "Escape") {{
        resetHighlight();
        hideTooltip();
      }}
    }});

    wrap.addEventListener("wheel", (evt) => {{
      if (!evt.shiftKey) return;

      evt.preventDefault();
      const factor = evt.deltaY < 0 ? 1.1 : (1 / 1.1);
      scale = Math.max(0.2, Math.min(8, scale * factor));
      graphDiv.style.transform = `scale(${{scale}})`;
    }}, {{ passive: false }});
  </script>
</body>
</html>
"""


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    ensure_graphviz_available()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print(f"Package dir: {PACKAGE_DIR}")
    print("Generating DOT with pyan...")
    dot_source = build_dot_source()

    print("Collecting docstrings...")
    docs_map, skipped = collect_docstrings(PACKAGE_DIR)

    if skipped:
        print("\nSkipped files during docstring extraction:")
        for msg in skipped:
            print(" -", msg)

    # dot_path = OUTDIR / "callgraph.dot"
    # dot_path.write_text(dot_source, encoding="utf-8")
    # print(f"Wrote DOT ({len(dot_source):,} bytes, {dot_source.count('->'):,} edges): {dot_path.resolve()}")

    print("Rendering SVG with Graphviz...")
    svg = render_svg_from_dot(dot_source)

    preview_path = OUTDIR / "preview.svg"
    html_path = OUTDIR / "index.html"

    preview_path.write_text(svg, encoding="utf-8")
    html_path.write_text(build_html(svg, docs_map), encoding="utf-8")

    print(f"\nWrote: {preview_path.resolve()}")
    print(f"Wrote: {html_path.resolve()}")
    print(f"Collected docstrings for {len(docs_map)} nodes")


if __name__ == "__main__":
    main()
