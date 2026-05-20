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
OUTDIR = Path("../")

# Optional subdirectories to ignore during docstring extraction
# Example: {"adrien", "legacy", "sandbox"}
SKIP_DIR_NAMES: set[str] = {"unused"}

# Optional file names to ignore
SKIP_FILE_NAMES: set[str] = set()

# Node labels to drop from the callgraph (applied to both node declarations
# and any edge that references them).  Use this for noisy dunder methods or
# ubiquitous helpers that clutter the graph.
SKIP_NODE_NAMES: set[str] = {"__iter__", "__next__", "__repr__", "__str__"}

# Font size (pt) for all node and cluster labels in the graph.
# pyan defaults to 14; increase for high-DPI / large-display use.
FONT_SIZE: int = 14

# Which pyan edge types to include.  Running both simultaneously causes graphviz
# to hang — the script works around this by calling pyan twice and merging edges.
DRAW_DEFINES: bool = True   # "X is defined inside Y" structural edges
DRAW_USES: bool = True      # "X calls / uses Y" call edges

# Truncate long docstrings for tooltip readability
MAX_DOC_CHARS = 1800

# ---- Layout ----------------------------------------------------------------
# Engine — pick one depending on graph size:
#   "fdp"    force-directed with cluster support — required because pyan always
#            emits subgraph clusters; sfdp ignores clusters and hangs  (default)
#   "neato"  spring-model, good for ~100 nodes
#   "circo"  circular, useful for cycle-heavy graphs
#   "sfdp"   DO NOT USE — hangs on any graph that contains subgraph clusters
LAYOUT_ENGINE: str = "fdp"

# Edge drawing style:
#   "line"     straight lines — only mode that doesn't hang fdp on large graphs (default)
#   "spline"   cubic splines — hangs on the full merged graph, avoid
#   "polyline" bent straight lines — also hangs on large graphs, avoid
#   "none"     no routing, lines through nodes (fast but ugly)
# Note: "curved" and "spline"/"polyline" all hang with fdp once the graph is large.
# Note: "ortho" only works with the "dot" engine.
SPLINES: str = "line"

# Overlap removal after layout:
#   "prism"    iterative overlap removal — correct, avoids node collision (default)
#   "compress" shrinks bounding box only — causes overlaps with compound=true, avoid
#   "false"    basic removal
#   "scale"    scale the graph up — makes things larger, avoid
OVERLAP: str = "prism"

# Merge parallel edges that share the same endpoints — reduces visual clutter.
CONCENTRATE: bool = True

# Ideal spring length for fdp.  compound=true (used for cluster-boundary edges)
# inflates spacing, so this needs to be much smaller than without it.
# 0.3 is compact; raise toward 1.0 if clusters overlap too much.
SPRING_K: float = 0.3

# Extra padding (points) around each node during overlap removal.
# Default Graphviz value is +4; lower values give tighter packing.
# Use "+1" for compact output; increase if node labels clip each other.
SEP: str = "+1"

# Maximum canvas size in inches (width,height).  Graphviz scales the layout
# to fit within these dimensions.  Use "" to disable.
# Tip: compound=true inflates natural spacing; size reins it back in.
GRAPH_SIZE: str = "20,15"


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

    # Rounded-rectangle nodes: inject global shape default, fontsize, and promote
    # per-node style="filled" → style="rounded,filled".  Cluster graph attrs use
    # style="filled,rounded" (already has rounded) so they are unaffected.
    dot_source = re.sub(
        r'(digraph\s+\w+\s*\{)',
        rf'\1\n    node [shape=box, fontsize={FONT_SIZE}];'
        rf'\n    graph [fontsize={FONT_SIZE}];',
        dot_source,
    )
    dot_source = dot_source.replace('style="filled"', 'style="rounded,filled"')

    # Replace any explicit per-element fontsize values (e.g. from future pyan versions).
    dot_source = re.sub(r'\bfontsize\s*=\s*\d+(?:\.\d+)?', f'fontsize={FONT_SIZE}', dot_source)

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
_SUBGRAPH_RE = re.compile(r'subgraph\s+"cluster_([^"]+)"')
_NODE_DECL_RE = re.compile(r'^\s*"([^"]+)"\s*\[')
_EDGE_FULL_RE = re.compile(r'^\s*"([^"]+)"\s*->\s*"([^"]+)"\s*(\[.*?\])', re.DOTALL)


def collapse_class_nodes(dot_source: str,
                          inh_pairs: set[tuple[str, str]] | None = None) -> str:
    """Remove nodes that duplicate their cluster box (applies to all nesting levels).

    With nested_groups=True pyan emits both a subgraph cluster AND a standalone
    node for every module and class.  This function:
      - collects all cluster IDs and builds a proxy-node map (first leaf node
        inside each cluster, used as the edge anchor for compound edges)
      - removes standalone node declarations whose ID matches a cluster
      - drops containment edges (dst.startswith(src+"__")) — already shown by nesting
      - redirects all other edges touching a removed node via lhead/ltail so that
        arrows terminate at the cluster boundary
    Applied to the fully-merged DOT so that both defines and uses edges are handled.

    inh_pairs: if provided, bare subclass nodes (classes with no cluster of their
    own that inherit from another class) are excluded from proxy selection.  pyan
    declares these class nodes inside their module cluster before the module's real
    method nodes, so without this exclusion they would be chosen as module proxies,
    then removed later by apply_inheritance_edges, leaving undeclared proxy references.
    """
    # Pass 1: collect ALL cluster IDs before selecting proxies so that class nodes
    # declared inside their module cluster (before their own subgraph appears) are
    # correctly recognised as cluster IDs and skipped during proxy selection.
    cluster_ids: set[str] = set()
    for line in dot_source.splitlines():
        m = _SUBGRAPH_RE.search(line)
        if m:
            cluster_ids.add(m.group(1))

    if not cluster_ids:
        return dot_source

    def _is_cluster_ancestor(nid: str) -> bool:
        prefix = nid + "__"
        return any(cid.startswith(prefix) for cid in cluster_ids)

    # Bare subclass nodes: in inh_pairs but have no cluster of their own.
    # pyan declares them inside their module cluster before any method nodes,
    # so they would be selected as module proxies if not excluded here.
    bare_sub_ids: set[str] = set()
    if inh_pairs:
        bare_sub_ids = {sub for sub, _par in inh_pairs if sub not in cluster_ids}

    # Pass 2: collect proxy nodes — first leaf node inside each cluster, skipping
    # cluster IDs themselves, bare subclass nodes, and namespace ancestors.
    proxy: dict[str, str] = {}
    stack: list[str] = []
    for line in dot_source.splitlines():
        m = _SUBGRAPH_RE.search(line)
        if m:
            stack.append(m.group(1))
            continue
        if line.strip() == '}' and stack:
            stack.pop()
            continue
        m = _NODE_DECL_RE.match(line)
        if m and stack:
            nid = m.group(1)
            if nid not in cluster_ids and nid not in bare_sub_ids:
                for cid in stack:
                    proxy.setdefault(cid, nid)

    # Also drop namespace/package ancestor nodes that have no cluster of their own
    # but ARE a prefix of existing clusters (e.g. openalea__metafspm when only
    # openalea__metafspm__coupling__* clusters exist).  These float outside all boxes.

    # Namespace/package ancestor nodes that have no cluster of their own —
    # drop their declarations AND any edges referencing them (not redirected).
    ancestor_ids: set[str] = {
        nid for nid in (
            _NODE_DECL_RE.match(l).group(1)
            for l in dot_source.splitlines()
            if _NODE_DECL_RE.match(l)
        )
        if nid not in cluster_ids and _is_cluster_ancestor(nid)
    }
    # All node IDs whose declarations should be suppressed
    drop_ids: set[str] = cluster_ids | ancestor_ids

    dot_source = re.sub(
        r'(digraph\s+\w+\s*\{)',
        r'\1\n    graph [compound=true];',
        dot_source, count=1,
    )

    out = []
    for line in dot_source.splitlines(keepends=True):
        m_decl = _NODE_DECL_RE.match(line)
        if m_decl and m_decl.group(1) in drop_ids:
            continue  # drop redundant / ancestor node declaration

        m_edge = _EDGE_FULL_RE.match(line)
        if m_edge:
            src, dst, attrs = m_edge.group(1), m_edge.group(2), m_edge.group(3)
            # Drop containment edges and edges referencing ancestor-only nodes
            if dst.startswith(src + "__"):
                continue
            if src in ancestor_ids or dst in ancestor_ids:
                continue
            src_c = src in cluster_ids
            dst_c = dst in cluster_ids
            if src_c or dst_c:
                new_src = proxy.get(src) if src_c else src
                new_dst = proxy.get(dst) if dst_c else dst
                # Skip if no proxy available (empty cluster)
                if (src_c and not new_src) or (dst_c and not new_dst):
                    continue
                extra = ""
                if src_c:
                    extra += f', ltail="cluster_{src}"'
                if dst_c:
                    extra += f', lhead="cluster_{dst}"'
                indent = re.match(r'^\s*', line).group(0)
                new_attrs = attrs[:-1].rstrip() + extra + "]"
                out.append(f'{indent}"{new_src}" -> "{new_dst}" {new_attrs}\n')
                continue

        out.append(line)
    return "".join(out)


def _pyan_call(*, draw_defines: bool, draw_uses: bool) -> str:
    return pyan.create_callgraph(
        filenames=[str(f) for f in PACKAGE_DIR.rglob("*.py") if not should_skip_file(f)],
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
    files = [f for f in PACKAGE_DIR.rglob("*.py") if not should_skip_file(f)]
    inh_pairs = collect_inheritance_pairs(files)

    if DRAW_DEFINES and DRAW_USES:
        base = filter_dot_external_nodes(_pyan_call(draw_defines=True, draw_uses=False))
        uses = filter_dot_external_nodes(_pyan_call(draw_defines=False, draw_uses=True))
        def _tag_uses(line: str) -> str:
            return re.sub(r'\](\s*;)', r', class="uses"]\1', line)
        extra_edges = [_tag_uses(l) for l in uses.splitlines(keepends=True) if _EDGE_LINE_RE.match(l)]
        merged = base.rstrip().rstrip("}").rstrip() + "\n" + "".join(extra_edges) + "\n}\n"
        nested = nest_inheritance_clusters(merged, inh_pairs)
        return drop_compound_inheritance_edges(collapse_class_nodes(nested, inh_pairs), inh_pairs)

    filtered = filter_dot_external_nodes(_pyan_call(draw_defines=DRAW_DEFINES, draw_uses=DRAW_USES))
    nested = nest_inheritance_clusters(filtered, inh_pairs)
    return drop_compound_inheritance_edges(collapse_class_nodes(nested, inh_pairs), inh_pairs)


_COMPOUND_EDGE_RE = re.compile(
    r'"([^"]+)"\s*->\s*"([^"]+)"\s*\[([^\]]+)\]', re.DOTALL
)


def _cluster_of(node_id: str, cluster_set: set[str]) -> str | None:
    """Return the cluster from cluster_set that node_id belongs to, or None.

    Matches either the cluster ID itself (class-level pyan node) or any node
    whose ID starts with <cluster_id>__ (method node inside that cluster).
    Handles dunder method names like __init__ correctly since we match the
    cluster ID as a prefix, not by splitting on __.
    """
    if node_id in cluster_set:
        return node_id
    return next((c for c in cluster_set if node_id.startswith(c + '__')), None)


def _find_cluster_bounds(dot: str, cluster_id: str) -> tuple[int, int]:
    """Return (start, end) char positions of the subgraph block for cluster_id.

    start is the index of 's' in 'subgraph "cluster_<id>"';
    end is one past the closing '}' (exclusive).
    Returns (-1, -1) if not found.
    """
    marker = f'subgraph "cluster_{cluster_id}"'
    pos = dot.find(marker)
    if pos == -1:
        return -1, -1
    brace = dot.index('{', pos)
    depth, i = 1, brace + 1
    while depth > 0 and i < len(dot):
        if dot[i] == '{':
            depth += 1
        elif dot[i] == '}':
            depth -= 1
        i += 1
    return pos, i


def build_proxy_to_cluster(dot_source: str) -> dict[str, str]:
    """Return {proxy_node_id: cluster_id} for every compound edge in the DOT.

    Graphviz compound edges use a real node as the visual anchor (proxy) while
    lhead/ltail attributes make the arrow terminate at the cluster boundary.
    The SVG edge title contains the proxy IDs, not the cluster names, so the
    JS highlight logic needs this map to resolve proxy → cluster.
    """
    result: dict[str, str] = {}
    for m in _COMPOUND_EDGE_RE.finditer(dot_source):
        src_node, dst_node, attrs = m.group(1), m.group(2), m.group(3)
        lt = re.search(r'ltail\s*=\s*"cluster_([^"]+)"', attrs)
        lh = re.search(r'lhead\s*=\s*"cluster_([^"]+)"', attrs)
        if lt:
            result[src_node] = lt.group(1)
        if lh:
            result[dst_node] = lh.group(1)
    return result


def collect_inheritance_pairs(files: list[Path]) -> set[tuple[str, str]]:
    """Return {(subclass_cluster_id, parent_cluster_id)} from class base declarations.

    Uses AST to find class X(Base) relationships and maps them to pyan-style
    cluster IDs so we can tag the corresponding DOT edges.
    """
    # Pass 1: register every class name → list of pyan cluster IDs (may be ambiguous)
    class_registry: dict[str, list[str]] = {}
    parsed: list[tuple[str, ast.Module]] = []

    for f in files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        module_id = pyan_node_id_from_dotted(module_name_from_file(f))
        parsed.append((module_id, tree))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_registry.setdefault(node.name, []).append(
                    f"{module_id}__{node.name}"
                )

    # Pass 2: for each class, add (subclass, parent) for each base
    pairs: set[tuple[str, str]] = set()
    for module_id, tree in parsed:
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            sub_id = f"{module_id}__{node.name}"
            for base in node.bases:
                if isinstance(base, ast.Name):
                    name = base.id
                elif isinstance(base, ast.Attribute):
                    name = base.attr
                else:
                    continue
                for parent_id in class_registry.get(name, []):
                    if parent_id != sub_id:
                        pairs.add((sub_id, parent_id))
    return pairs


def build_cluster_hierarchy(
    dot_source: str,
    inh_pairs: set[tuple[str, str]],
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Return (children, parent) maps for clusters that are actually nested.

    children[par_id] = [sub_id, ...]   (direct children only)
    parent[sub_id]   = par_id          (direct parent)
    """
    cluster_ids: set[str] = set(
        re.findall(r'subgraph\s+"cluster_([^"]+)"', dot_source)
    )
    children: dict[str, list[str]] = {}
    parent: dict[str, str] = {}
    for sub, par in inh_pairs:
        if sub in cluster_ids and par in cluster_ids:
            children.setdefault(par, []).append(sub)
            parent[sub] = par
    return children, parent


def nest_inheritance_clusters(dot_source: str,
                               inheritance_pairs: set[tuple[str, str]]) -> str:
    """Represent inheritance visually by nesting the subclass cluster inside
    the parent class cluster ("class box inside class box").

    For each (sub, par) pair where both classes have their own cluster the
    entire subgraph block for sub is extracted from its current position and
    re-inserted just before the closing brace of the par cluster.

    Also removes:
    - Bare subclass nodes (classes that inherit but have no methods/cluster).
    - Any pre-collapse uses edges that cross an inheritance boundary (they
      become visually redundant once sub is nested inside par).
    """
    if not inheritance_pairs:
        return dot_source

    cluster_ids: set[str] = set(
        re.findall(r'subgraph\s+"cluster_([^"]+)"', dot_source)
    )
    bare_sub_ids: set[str] = {
        sub for sub, _ in inheritance_pairs if sub not in cluster_ids
    }
    inh_set: set[tuple[str, str]] = {
        (sub, par) for sub, par in inheritance_pairs
        if sub in cluster_ids and par in cluster_ids
    }
    sub_ids = {sub for sub, _ in inh_set}
    par_ids = {par for _, par in inh_set}

    # ── Move each sub cluster inside its par cluster ──────────────────────
    for sub_id, par_id in sorted(inh_set):
        sub_s, sub_e = _find_cluster_bounds(dot_source, sub_id)
        if sub_s == -1:
            continue
        par_s, par_e = _find_cluster_bounds(dot_source, par_id)
        if par_s == -1:
            continue
        if par_s <= sub_s < sub_e <= par_e:
            continue  # already nested

        # Include leading whitespace of the sub line
        nl = dot_source.rfind('\n', 0, sub_s)
        ws_start = nl + 1 if nl != -1 else 0
        sub_block = dot_source[ws_start:sub_e]

        # Consume trailing semicolon and single newline after closing brace
        tail = sub_e
        while tail < len(dot_source) and dot_source[tail] in ' \t;':
            tail += 1
        if tail < len(dot_source) and dot_source[tail] == '\n':
            tail += 1

        dot_source = dot_source[:ws_start] + dot_source[tail:]

        # Re-find parent after removal (offsets may have changed)
        par_s2, par_e2 = _find_cluster_bounds(dot_source, par_id)
        if par_s2 == -1:
            dot_source = dot_source[:ws_start] + sub_block + '\n' + dot_source[ws_start:]
            continue

        par_close = par_e2 - 1  # index of par's closing '}'

        # Compute indentation for the re-inserted block
        par_nl = dot_source.rfind('\n', 0, par_s2)
        par_col = par_s2 - (par_nl + 1) if par_nl != -1 else par_s2
        extra = ' ' * (par_col + 4)

        # Re-indent sub block to match the new nesting depth
        lines = sub_block.rstrip('\n').splitlines()
        cur_ind = len(lines[0]) - len(lines[0].lstrip()) if lines else 0
        reindented = '\n'.join(
            extra + ln[cur_ind:] if ln[:cur_ind].strip() == '' else ln
            for ln in lines
        ) + '\n'

        dot_source = dot_source[:par_close] + reindented + dot_source[par_close:]

    # ── Remove bare subclass nodes and inheritance-crossing edges ─────────
    node_re = re.compile(r'^\s*"([^"]+)"\s*\[')
    edge_re = re.compile(r'^\s*"([^"]+)"\s*->\s*"([^"]+)"')

    out: list[str] = []
    for line in dot_source.splitlines(keepends=True):
        m_n = node_re.match(line)
        if m_n and m_n.group(1) in bare_sub_ids:
            continue

        m_e = edge_re.match(line)
        if m_e:
            src_raw, dst_raw = m_e.group(1), m_e.group(2)
            if src_raw in bare_sub_ids or dst_raw in bare_sub_ids:
                continue
            # Drop edges crossing an inheritance boundary (sub → par direction)
            src_c = _cluster_of(src_raw, sub_ids)
            dst_c = _cluster_of(dst_raw, par_ids)
            if src_c and dst_c and (src_c, dst_c) in inh_set:
                continue

        out.append(line)

    return ''.join(out)


def drop_compound_inheritance_edges(dot_source: str,
                                     inheritance_pairs: set[tuple[str, str]]) -> str:
    """Drop compound and semi-compound edges crossing inheritance boundaries.

    Called after collapse_class_nodes which converts class-level pyan edges
    into compound edges with lhead/ltail.  Those are now redundant: the
    sub→par relationship is already expressed by visual nesting.
    """
    if not inheritance_pairs:
        return dot_source

    cluster_ids: set[str] = set(
        re.findall(r'subgraph\s+"cluster_([^"]+)"', dot_source)
    )
    inh_set: set[tuple[str, str]] = {
        (sub, par) for sub, par in inheritance_pairs
        if sub in cluster_ids and par in cluster_ids
    }
    sub_ids = {sub for sub, _ in inh_set}
    par_ids = {par for _, par in inh_set}

    edge_re = re.compile(r'^\s*"([^"]+)"\s*->\s*"([^"]+)"')
    out: list[str] = []
    for line in dot_source.splitlines(keepends=True):
        m = edge_re.match(line)
        if m:
            src_raw, dst_raw = m.group(1), m.group(2)
            lt_m = re.search(r'ltail\s*=\s*"cluster_([^"]+)"', line)
            lh_m = re.search(r'lhead\s*=\s*"cluster_([^"]+)"', line)
            # For src: prefer explicit ltail; fall back to prefix match
            src_c = lt_m.group(1) if lt_m else _cluster_of(src_raw, sub_ids)
            # For dst: prefer explicit lhead; fall back to prefix match
            dst_c = lh_m.group(1) if lh_m else _cluster_of(dst_raw, par_ids)
            if src_c and dst_c and (src_c, dst_c) in inh_set:
                continue
        out.append(line)
    return ''.join(out)


GRAPHVIZ_TIMEOUT = 30  # seconds before we give up and raise TimeoutExpired


def render_svg_from_dot(dot_source: str) -> str:
    """Render DOT to SVG using the configured Graphviz engine and layout options."""
    cmd = [
        LAYOUT_ENGINE, "-Tsvg",
        f"-Gsplines={SPLINES}",
        f"-Goverlap={OVERLAP}",
        f"-Gsep={SEP}",
        f"-Gconcentrate={'true' if CONCENTRATE else 'false'}",
        f"-GK={SPRING_K}",
        *([ f"-Gsize={GRAPH_SIZE}" ] if GRAPH_SIZE else []),
    ]
    result = subprocess.run(
        cmd,
        input=dot_source,
        text=True,
        capture_output=True,
        check=True,
        timeout=GRAPHVIZ_TIMEOUT,
    )
    svg = result.stdout
    # Graphviz embeds font sizes as presentation attributes (font-size="14.00").
    # Replace the default 14pt value with FONT_SIZE so cluster labels also scale.
    svg = re.sub(r'font-size="14\.00"', f'font-size="{FONT_SIZE:.2f}"', svg)
    return svg


# ============================================================================
# HTML generation
# ============================================================================

def build_html(svg: str, docs_map: dict[str, dict[str, str]],
               proxy_to_cluster: dict[str, str] | None = None,
               cluster_children: dict[str, list[str]] | None = None,
               cluster_parent: dict[str, str] | None = None) -> str:
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

    g.cluster {{
      cursor: pointer;
    }}

    #mode-buttons {{
      display: inline-flex;
      gap: 4px;
      margin-left: 14px;
      vertical-align: middle;
    }}
    .mode-btn {{
      padding: 2px 10px;
      font-size: 12px;
      border: 1px solid #bbb;
      border-radius: 4px;
      background: #f0f0f0;
      cursor: pointer;
      user-select: none;
      line-height: 1.6;
    }}
    .mode-btn.active {{
      background: #0066cc;
      color: #fff;
      border-color: #004fa3;
    }}
    .mode-btn:hover:not(.active) {{
      background: #e0e0e0;
    }}

    /* defines edges (structural containment) */
    g.edge:not(.uses) path {{ stroke: black; }}
    g.edge:not(.uses) polygon {{ fill: black; stroke: black; }}

    /* uses edges (calls/references) are shown faint so structure is readable */
    g.edge.uses path, g.edge.uses polygon {{
      opacity: 0.18;
    }}

    #tooltip {{
      position: fixed;
      z-index: 1000;
      max-width: 560px;
      min-width: 280px;
      max-height: 40vh;
      overflow-y: auto;
      display: none;
      background: rgba(30, 30, 30, 0.97);
      color: white;
      border-radius: 8px;
      padding: 10px 12px;
      box-shadow: 0 8px 30px rgba(0,0,0,0.28);
      font-size: 13px;
      line-height: 1.45;
      pointer-events: auto;
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
    <span id="mode-buttons">
      <button class="mode-btn active" data-mode="select" title="Click nodes to highlight dependencies">&#9011; Select</button>
      <button class="mode-btn" data-mode="pan" title="Drag to pan">&#10021; Pan</button>
      <button class="mode-btn" data-mode="boxzoom" title="Drag a rectangle to zoom in">&#8853; Box Zoom</button>
    </span>
    <span class="hint">Hover for docstrings &middot; Shift+wheel to zoom &middot; Esc to reset</span>
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
    // Maps proxy-node IDs (used as edge anchors in compound edges) to their
    // cluster IDs, so click-highlight can resolve node-level edge titles back
    // to the cluster they visually represent.
    const proxyToCluster = {json.dumps(proxy_to_cluster or {})};
    // Inheritance nesting maps — used so that clicking a parent class also
    // highlights its nested child classes, and vice versa.
    // clusterChildren[parId] = [childId, ...]   (direct children)
    // clusterParent[childId] = parId            (direct parent)
    const clusterChildren = {json.dumps(cluster_children or {})};
    const clusterParent   = {json.dumps(cluster_parent or {})};

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
      if (key) nodesByKey.set(key, node);
    }}
    // Add cluster boxes (class containers) so they participate in highlight/dim.
    for (const cluster of graphDiv.querySelectorAll("g.cluster")) {{
      const title = getTitleText(cluster);
      if (!title) continue;
      const key = title.startsWith("cluster_") ? title.slice("cluster_".length) : title;
      nodesByKey.set(key, cluster);
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

    // Resolve a node ID to its cluster ID when it acts as a compound-edge proxy.
    function canonical(id) {{ return proxyToCluster[id] ?? id; }}

    for (const edge of edgeEls) {{
      const text = getTitleText(edge);
      if (!text || !text.includes("->")) {{
        continue;
      }}

      const parts = text.split("->").map(s => s.trim());
      if (parts.length !== 2) {{
        continue;
      }}

      const [rawSrc, rawDst] = parts;
      const src = canonical(rawSrc);
      const dst = canonical(rawDst);

      edges.push({{
        el: edge,
        src: src,
        dst: dst,
        key: `${{rawSrc}}->${{rawDst}}`
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
      for (const n of nodesByKey.values()) {{
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

      // Expand keepNodes to include:
      //   descendants — nested child clusters (and their children, recursively)
      //   ancestors   — parent clusters up to the root (so container boxes stay lit)
      // for...of on a Set visits items added during iteration, so one pass suffices.
      for (const key of keepNodes) {{
        for (const child of (clusterChildren[key] || [])) {{
          keepNodes.add(child);
        }}
        let anc = clusterParent[key];
        while (anc && !keepNodes.has(anc)) {{
          keepNodes.add(anc);
          anc = clusterParent[anc];
        }}
      }}

      const keepEdges = new Set();

      for (const e of edges) {{
        if (keepNodes.has(e.src) && keepNodes.has(e.dst)) {{
          keepEdges.add(e.key);
        }}
      }}

      for (const [key, el] of nodesByKey.entries()) {{
        // A method/node is "on" if it's directly in keepNodes, OR if its
        // enclosing cluster is (so method boxes inside a focused cluster stay visible).
        const on = keepNodes.has(key) ||
          [...keepNodes].some(ck => key.startsWith(ck + "__"));
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
      if (tooltipPinned) return;
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

    let tooltipPinned = false;  // true while cursor is over the tooltip itself

    function hideTooltip() {{
      if (tooltipPinned) return;
      tooltip.style.display = "none";
    }}

    // Keep tooltip visible while the user scrolls it; stop following the cursor.
    tooltip.addEventListener("mouseenter", () => {{ tooltipPinned = true; }});
    tooltip.addEventListener("mouseleave", () => {{
      tooltipPinned = false;
      tooltip.style.display = "none";
    }});

    // ── Interaction mode ────────────────────────────────────────────────────
    let mode = "select";  // "select" | "pan" | "boxzoom"

    const modeButtons = document.querySelectorAll(".mode-btn");
    function setMode(m) {{
      mode = m;
      modeButtons.forEach(b => b.classList.toggle("active", b.dataset.mode === m));
      const cursors = {{ select: "default", pan: "grab", boxzoom: "crosshair" }};
      wrap.style.cursor = cursors[m] || "default";
    }}
    modeButtons.forEach(b => b.addEventListener("click", () => setMode(b.dataset.mode)));

    // ── Pan & box-zoom drag state ────────────────────────────────────────────
    let dragging  = false;
    let dragMoved = false;
    let dragOrigin = null;
    let selBox = null;

    wrap.addEventListener("mousedown", evt => {{
      if (mode === "pan") {{
        dragging   = true;
        dragMoved  = false;
        dragOrigin = {{
          clientX: evt.clientX, clientY: evt.clientY,
          scrollLeft: wrap.scrollLeft, scrollTop: wrap.scrollTop,
        }};
        wrap.style.cursor = "grabbing";
        evt.preventDefault();
      }} else if (mode === "boxzoom") {{
        dragging   = true;
        dragMoved  = false;
        dragOrigin = {{ clientX: evt.clientX, clientY: evt.clientY }};
        selBox = document.createElement("div");
        selBox.style.cssText =
          "position:fixed;pointer-events:none;z-index:200;" +
          "border:2px solid #0066cc;background:rgba(0,102,204,0.08);";
        selBox.style.left = evt.clientX + "px";
        selBox.style.top  = evt.clientY + "px";
        selBox.style.width  = "0";
        selBox.style.height = "0";
        document.body.appendChild(selBox);
        evt.preventDefault();
      }}
    }});

    document.addEventListener("mousemove", evt => {{
      if (!dragging) return;
      if (mode === "pan" && dragOrigin) {{
        const dx = evt.clientX - dragOrigin.clientX;
        const dy = evt.clientY - dragOrigin.clientY;
        if (Math.abs(dx) > 2 || Math.abs(dy) > 2) dragMoved = true;
        wrap.scrollLeft = dragOrigin.scrollLeft - dx;
        wrap.scrollTop  = dragOrigin.scrollTop  - dy;
      }} else if (mode === "boxzoom" && selBox && dragOrigin) {{
        const x = Math.min(evt.clientX, dragOrigin.clientX);
        const y = Math.min(evt.clientY, dragOrigin.clientY);
        const w = Math.abs(evt.clientX - dragOrigin.clientX);
        const h = Math.abs(evt.clientY - dragOrigin.clientY);
        selBox.style.left   = x + "px";
        selBox.style.top    = y + "px";
        selBox.style.width  = w + "px";
        selBox.style.height = h + "px";
        if (w > 4 || h > 4) dragMoved = true;
      }}
    }});

    document.addEventListener("mouseup", evt => {{
      if (!dragging) return;
      dragging = false;

      if (mode === "pan") {{
        wrap.style.cursor = "grab";
      }} else if (mode === "boxzoom" && selBox) {{
        if (dragMoved) {{
          const wrapRect = wrap.getBoundingClientRect();
          const x1c = Math.min(evt.clientX, dragOrigin.clientX);
          const y1c = Math.min(evt.clientY, dragOrigin.clientY);
          const x2c = Math.max(evt.clientX, dragOrigin.clientX);
          const y2c = Math.max(evt.clientY, dragOrigin.clientY);
          // Convert viewport coords → graph coords (before scaling)
          const x1g = (x1c - wrapRect.left + wrap.scrollLeft) / scale;
          const y1g = (y1c - wrapRect.top  + wrap.scrollTop)  / scale;
          const x2g = (x2c - wrapRect.left + wrap.scrollLeft) / scale;
          const y2g = (y2c - wrapRect.top  + wrap.scrollTop)  / scale;
          const selW = x2g - x1g;
          const selH = y2g - y1g;
          if (selW > 10 && selH > 10) {{
            scale = Math.max(0.2, Math.min(8,
              Math.min(wrapRect.width / selW, wrapRect.height / selH) * 0.92
            ));
            graphDiv.style.transform = `scale(${{scale}})`;
            wrap.scrollLeft = x1g * scale;
            wrap.scrollTop  = y1g * scale;
          }}
        }}
        document.body.removeChild(selBox);
        selBox = null;
      }}
      dragOrigin = null;
      dragMoved  = false;
    }});

    // ── Node / cluster interaction (select mode only) ────────────────────────
    for (const [key, el] of nodesByKey.entries()) {{
      el.addEventListener("click", (evt) => {{
        if (mode !== "select") return;
        evt.stopPropagation();
        highlightFrom(key);
      }});
      el.addEventListener("mouseenter", (evt) => showTooltip(evt, key));
      el.addEventListener("mousemove", moveTooltip);
      el.addEventListener("mouseleave", hideTooltip);
    }}

    graphDiv.addEventListener("click", () => {{
      if (mode !== "select") return;
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
    proxy_to_cluster = build_proxy_to_cluster(dot_source)
    files = [f for f in PACKAGE_DIR.rglob("*.py") if not should_skip_file(f)]
    inh_pairs = collect_inheritance_pairs(files)
    cluster_children, cluster_parent = build_cluster_hierarchy(dot_source, inh_pairs)
    html_path.write_text(
        build_html(svg, docs_map, proxy_to_cluster, cluster_children, cluster_parent),
        encoding="utf-8",
    )

    print(f"\nWrote: {preview_path.resolve()}")
    print(f"Wrote: {html_path.resolve()}")
    print(f"Collected docstrings for {len(docs_map)} nodes")


if __name__ == "__main__":
    main()
