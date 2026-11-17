"""Check suite import boundaries without importing apps, devices or model code.

Run from any directory: python scripts/check_architecture.py
The graph includes function-local and TYPE_CHECKING imports. It describes static
dependencies, not proof that an import cycle fails at runtime. Existing delayed
cycles are listed by their complete directed path; new cycles fail the check.
Named child imports point to that child, without expanding package-initializer
side effects. Nonliteral dynamic import targets cannot be resolved statically.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "ningsi": "ningsi/src/ningsi",
    "ningsi_studio": "ningsi-studio/src/ningsi_studio",
    "ningsi_capture": "ningsi-capture/src/ningsi_capture",
}

# These exact dependencies participate in the frozen feature fingerprint.
# New signal/acquisition code must keep model and orchestration dependencies out.
KNOWN_LAYER_EXCEPTIONS = {
    ("ningsi.signal.conditioning", "ningsi.models.logistic"):
        "Frozen conditioning calls the legacy six-feature extractor.",
    ("ningsi.signal.conditioning", "ningsi.models.research_features"):
        "Frozen conditioning preserves legacy processing and implementation hashes.",
}

# Exact paths, rather than a whole-package exemption, keep a new cycle visible.
KNOWN_CYCLES = {
    ("ningsi_capture.export", "ningsi_capture.storage", "ningsi_capture.probe_prediction"):
        "SessionStore creates the probe worker with a local import; workers reuse export window extraction.",
    ("ningsi_capture.probe_prediction", "ningsi_capture.storage"):
        "Probe workers reuse read-only storage helpers; their construction is delayed until SessionStore starts.",
    ("ningsi_studio.__main__", "ningsi_studio.app"):
        "The legacy app.main entry point forwards to the CLI with a function-local import.",
}


@dataclass(frozen=True)
class Module:
    name: str
    path: Path
    is_package: bool = False


@dataclass(frozen=True)
class Dependency:
    source: str
    target: str
    path: Path
    line: int


@dataclass
class Audit:
    modules: dict[str, Module]
    dependencies: list[Dependency]
    cycles: list[tuple[str, ...]]
    problems: list[str]


def module_index(root: Path) -> dict[str, Module]:
    modules = {}
    for package, folder in PACKAGES.items():
        directory = root / folder
        if not (directory / "__init__.py").is_file():
            raise ValueError(f"Missing package source: {directory}")
        for path in sorted(directory.rglob("*.py")):
            relative = path.relative_to(directory).with_suffix("")
            parts = list(relative.parts)
            is_package = parts[-1] == "__init__"
            if is_package:
                parts.pop()
            name = ".".join([package, *parts])
            if name in modules:
                raise ValueError(f"Duplicate module {name}: {path}")
            modules[name] = Module(name, path, is_package)
    return modules


def _relative_name(name: str, module: Module, package: str | None = None) -> str:
    if not name.startswith("."):
        return name
    parent = package or (module.name if module.is_package else module.name.rpartition(".")[0])
    return importlib.util.resolve_name(name, parent)


def imports(module: Module, known_modules: set[str]) -> list[Dependency]:
    tree = ast.parse(module.path.read_text(encoding="utf-8-sig"), filename=str(module.path))
    nodes = list(ast.walk(tree))
    importlib_aliases = {"importlib"}
    loader_aliases = {"__import__"}
    for node in nodes:
        if isinstance(node, ast.Import):
            importlib_aliases.update(alias.asname or alias.name for alias in node.names if alias.name == "importlib")
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib" and node.level == 0:
            loader_aliases.update(alias.asname or alias.name for alias in node.names if alias.name == "import_module")
    found = []
    for node in nodes:
        targets = []
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = _relative_name("." * node.level + (node.module or ""), module)
            # A named child module is the direct dependency. For an imported
            # class/function/constant, the containing module is the dependency.
            for alias in node.names:
                child = f"{base}.{alias.name}"
                targets.append(child if child in known_modules else base)
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            name = node.args[0].value
            direct = isinstance(node.func, ast.Name) and node.func.id in loader_aliases
            dotted = (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"
                      and isinstance(node.func.value, ast.Name) and node.func.value.id in importlib_aliases)
            if isinstance(name, str) and (direct or dotted):
                package_node = node.args[1] if len(node.args) > 1 else next(
                    (kw.value for kw in node.keywords if kw.arg == "package"), None)
                package = package_node.value if isinstance(package_node, ast.Constant) and isinstance(package_node.value, str) else None
                targets = [_relative_name(name, module, package)]
        for target in targets:
            if target and target != module.name:
                found.append(Dependency(module.name, target, module.path, node.lineno))
    return found


def boundary_problem(source: str, target: str) -> str | None:
    origin, destination = source.split(".")[0], target.split(".")[0]
    if origin == "ningsi" and destination in {"ningsi_studio", "ningsi_capture"}:
        return "engine must not depend on a product application"
    if {origin, destination} == {"ningsi_studio", "ningsi_capture"}:
        return "Studio and Capture exchange artifacts; they must not import each other"
    if origin == destination == "ningsi":
        source_layer = source.split(".")[1] if "." in source else ""
        target_layer = target.split(".")[1] if "." in target else ""
        if source_layer not in {"app", "adapters"} and target_layer in {"app", "adapters"}:
            return "engine algorithms must not depend on CLI/UI orchestration or external adapters"
        if (source_layer in {"signal", "acquisition"} and target_layer in {"models", "processing"}
                and (source, target) not in KNOWN_LAYER_EXCEPTIONS):
            return "signal/acquisition must not depend on model or processing orchestration"
    return None


def find_cycles(graph: dict[str, set[str]], limit: int = 1000) -> list[tuple[str, ...]]:
    """Enumerate exact directed cycles, each starting with its smallest module.

    Restrict searches to strongly connected components first. The explicit cap
    fails closed if future coupling makes elementary-cycle enumeration too large.
    """
    index = 0
    indices, low = {}, {}
    stack, active, components = [], set(), []

    def component(node):
        nonlocal index
        indices[node] = low[node] = index
        index += 1
        stack.append(node)
        active.add(node)
        for target in sorted(graph[node]):
            if target not in indices:
                component(target)
                low[node] = min(low[node], low[target])
            elif target in active:
                low[node] = min(low[node], indices[target])
        if low[node] == indices[node]:
            group = set()
            while True:
                target = stack.pop()
                active.remove(target)
                group.add(target)
                if target == node:
                    break
            if len(group) > 1:
                components.append(group)

    for node in sorted(graph):
        if node not in indices:
            component(node)
    cycles = []
    for group in components:
        def visit(start, node, path, seen):
            for target in sorted(graph[node] & group):
                if target == start:
                    cycles.append(tuple(path))
                    if len(cycles) > limit:
                        raise ValueError(f"More than {limit} static import cycles; reduce coupling before continuing")
                elif target > start and target not in seen:
                    visit(start, target, path + [target], seen | {target})
        for start in sorted(group):
            visit(start, start, [start], {start})
    return sorted(cycles)


def audit(root: Path, allowed_cycles=None) -> Audit:
    allowed_cycles = KNOWN_CYCLES if allowed_cycles is None else allowed_cycles
    modules = module_index(root)
    dependencies, problems = [], []
    for module in modules.values():
        try:
            dependencies.extend(imports(module, set(modules)))
        except (SyntaxError, ValueError, ImportError) as exc:
            problems.append(f"{module.path.relative_to(root)}: {exc}")
    graph = {name: set() for name in modules}
    for edge in dependencies:
        reason = boundary_problem(edge.source, edge.target)
        if reason:
            problems.append(f"{edge.path.relative_to(root)}:{edge.line}: {edge.source} -> {edge.target}: {reason}")
        if edge.target in graph:
            graph[edge.source].add(edge.target)
    try:
        cycles = find_cycles(graph)
    except (ValueError, RecursionError) as exc:
        cycles = []
        problems.append(f"Import cycle analysis failed: {exc}")
    for cycle in cycles:
        if cycle not in allowed_cycles:
            problems.append("New static import cycle: " + " -> ".join((*cycle, cycle[0])))
    return Audit(modules, dependencies, cycles, problems)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="suite repository root")
    args = parser.parse_args(argv)
    try:
        result = audit(args.root.resolve())
    except (OSError, ValueError) as exc:
        print(f"Architecture check failed: {exc}", file=sys.stderr)
        return 1
    print(f"Architecture: {len(result.modules)} modules, {len(result.dependencies)} static import references")
    for edge in sorted({(edge.source, edge.target) for edge in result.dependencies}):
        if edge in KNOWN_LAYER_EXCEPTIONS:
            print("  Frozen-feature dependency: " + " -> ".join(edge))
            print("    " + KNOWN_LAYER_EXCEPTIONS[edge])
    for cycle in result.cycles:
        if cycle in KNOWN_CYCLES:
            print("  Existing delayed-import cycle: " + " -> ".join((*cycle, cycle[0])))
            print("    " + KNOWN_CYCLES[cycle])
    for problem in result.problems:
        print("  ERROR: " + problem)
    if result.problems:
        return 1
    print("Architecture check passed (package boundaries and exact cycle exceptions).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
