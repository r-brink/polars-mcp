import inspect
import json
from pathlib import Path
from typing import Any

import polars as pl
from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("polars_mcp")

# Constants
CHARACTER_LIMIT = 25000
GUIDES_DIR = Path(__file__).parent / "guides"

GUIDE_FILES = {
    "contexts": "contexts.md",
    "expressions": "expressions.md",
    "lazy-api": "lazy-api.md",
    "pandas-to-polars": "pandas-to-polars.md",
}


def load_guide(filename: str) -> str:
    """Load a guide file from the guides directory."""
    guide_path = GUIDES_DIR / filename
    try:
        with open(guide_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Guide file not found: {filename}"
    except Exception as e:
        return f"Error loading guide: {str(e)}"


# API Index - built on first access
_api_index: dict[str, Any] | None = None
_namespace_registry: dict[str, dict[str, list[str]]] | None = None


def _create_test_instance(class_name: str, cls: type) -> Any:
    """Create a minimal test instance for namespace inspection."""
    try:
        if class_name == "Expr":
            return pl.col("_")
        elif class_name == "Series":
            return pl.Series("_", [1])
        elif class_name == "DataFrame":
            return pl.DataFrame({"_": [1]})
        elif class_name == "LazyFrame":
            return pl.LazyFrame({"_": [1]})
    except Exception:
        return None
    return None


def _extract_short_description(docstring: str | None) -> str:
    """Extract first sentence/line as short description."""
    if not docstring:
        return ""
    first_line = docstring.split("\n")[0].strip()
    if "." in first_line:
        return first_line.split(".")[0] + "."
    return first_line[:100] + ("..." if len(first_line) > 100 else "")


def build_api_index() -> tuple[dict[str, Any], dict[str, dict[str, list[str]]]]:
    """Build index of Polars API by introspecting the module.

    Returns:
        Tuple of (api_index, namespace_registry)
    """
    index = {}
    namespace_registry: dict[str, dict[str, list[str]]] = {}

    main_classes = [
        ("DataFrame", pl.DataFrame),
        ("LazyFrame", pl.LazyFrame),
        ("Series", pl.Series),
        ("Expr", pl.Expr),
    ]

    for class_name, cls in main_classes:
        namespace_registry[class_name] = {"_methods": []}

        doc = inspect.getdoc(cls)
        index[class_name] = {
            "type": "class",
            "class": class_name,
            "namespace": None,
            "full_name": f"polars.{class_name}",
            "docstring": doc or "No documentation available",
            "signature": "",
            "short_description": _extract_short_description(doc),
        }

        namespace_names = getattr(cls, "_accessors", set())
        test_instance = (
            _create_test_instance(class_name, cls) if namespace_names else None
        )

        for method_name in dir(cls):
            if method_name.startswith("_"):
                continue

            try:
                method = inspect.getattr_static(cls, method_name)
                method_doc = inspect.getdoc(method)

                try:
                    sig = inspect.signature(method)
                    sig_str = str(sig)
                except (ValueError, TypeError):
                    sig_str = "(...)"

                full_name = f"{class_name}.{method_name}"
                namespace_registry[class_name]["_methods"].append(method_name)

                index[full_name] = {
                    "type": "method",
                    "class": class_name,
                    "namespace": None,
                    "full_name": f"polars.{full_name}",
                    "docstring": method_doc or "No documentation available",
                    "signature": sig_str,
                    "short_description": _extract_short_description(method_doc),
                }

                # Index namespace methods
                if method_name in namespace_names and test_instance is not None:
                    try:
                        namespace_obj = getattr(test_instance, method_name)
                        namespace_class = type(namespace_obj)
                        namespace_registry[class_name][method_name] = []

                        for ns_method_name in dir(namespace_class):
                            if ns_method_name.startswith("_"):
                                continue

                            try:
                                ns_method = getattr(namespace_class, ns_method_name)
                                if not callable(ns_method):
                                    continue

                                ns_doc = inspect.getdoc(ns_method)
                                try:
                                    ns_sig = inspect.signature(ns_method)
                                    ns_sig_str = str(ns_sig)
                                except (ValueError, TypeError):
                                    ns_sig_str = "(...)"

                                ns_full_name = (
                                    f"{class_name}.{method_name}.{ns_method_name}"
                                )
                                namespace_registry[class_name][method_name].append(
                                    ns_method_name
                                )

                                index[ns_full_name] = {
                                    "type": "namespace_method",
                                    "class": class_name,
                                    "namespace": method_name,
                                    "full_name": f"polars.{ns_full_name}",
                                    "docstring": ns_doc or "No documentation available",
                                    "signature": ns_sig_str,
                                    "short_description": _extract_short_description(
                                        ns_doc
                                    ),
                                }
                            except (AttributeError, TypeError):
                                continue
                    except Exception:
                        continue

            except (AttributeError, TypeError):
                continue

    # Index top-level functions
    namespace_registry["functions"] = {"_methods": []}

    for name in dir(pl):
        if name.startswith("_"):
            continue

        obj = getattr(pl, name)

        if inspect.isclass(obj) and name in [c[0] for c in main_classes]:
            continue

        if inspect.isclass(obj):
            doc = inspect.getdoc(obj)
            index[name] = {
                "type": "class",
                "class": None,
                "namespace": None,
                "full_name": f"polars.{name}",
                "docstring": doc or "No documentation available",
                "signature": "",
                "short_description": _extract_short_description(doc),
            }
        elif callable(obj):
            doc = inspect.getdoc(obj)
            try:
                sig = inspect.signature(obj)
                sig_str = str(sig)
            except (ValueError, TypeError):
                sig_str = "(...)"

            namespace_registry["functions"]["_methods"].append(name)

            index[name] = {
                "type": "function",
                "class": None,
                "namespace": None,
                "full_name": f"polars.{name}",
                "docstring": doc or "No documentation available",
                "signature": sig_str,
                "short_description": _extract_short_description(doc),
            }

    return index, namespace_registry


def get_api_index() -> dict[str, Any]:
    """Get or build the API index."""
    global _api_index, _namespace_registry
    if _api_index is None:
        _api_index, _namespace_registry = build_api_index()
    return _api_index


def get_namespace_registry() -> dict[str, dict[str, list[str]]]:
    """Get or build the namespace registry."""
    global _api_index, _namespace_registry
    if _namespace_registry is None:
        _api_index, _namespace_registry = build_api_index()
    return _namespace_registry


def normalize_api_name(name: str) -> str:
    """Normalize API name by removing common prefixes."""
    name = name.strip()
    if name.startswith("polars."):
        return name[7:]
    elif name.startswith("pl."):
        return name[3:]
    return name


def calculate_match_score(
    term: str, name: str, docstring: str, short_desc: str
) -> float:
    """Score a single search term against an API item."""
    t = normalize_api_name(term).lower()
    n, d, s = name.lower(), docstring.lower(), short_desc.lower()

    # Exact name match (highest priority)
    if t == n or n.endswith(f".{t}"):
        return 5.0

    # Namespace match (Class.namespace.query)
    if f".{t}." in n:
        return 3.0

    # Name contains term
    if t in n:
        score = 2.0
        if n.startswith(t):
            score += 0.5
        score += len(t) / len(n)
        return score

    # Short description match
    if t in s:
        return 1.5

    # Full docstring match
    if t in d:
        return 0.5

    return 0.0


def search_index(
    query: str,
    limit: int = 20,
    class_filter: str | None = None,
    namespace_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Search the API index with optional filters. Supports multi-word queries (OR logic - matches any term)."""
    index = get_api_index()

    # Split query into terms for multi-word search
    terms = [t.strip() for t in query.lower().replace(",", " ").split() if t.strip()]

    if not terms:
        return []

    results = []
    for name, info in index.items():
        # Apply class filter
        if class_filter and info.get("class") != class_filter:
            continue

        # Apply namespace filter
        if namespace_filter and info.get("namespace") != namespace_filter:
            continue

        # Score each term and sum (OR logic)
        total_score = 0.0
        for term in terms:
            total_score += calculate_match_score(
                term, name, info["docstring"], info.get("short_description", "")
            )

        if total_score > 0:
            results.append({"score": total_score, "name": name, **info})

    results.sort(key=lambda x: (-x["score"], x["name"]))
    return results[:limit]


def group_search_results(
    results: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group search results by class and namespace."""
    groups: dict[str, list[dict[str, Any]]] = {
        "functions": [],
        "DataFrame": [],
        "LazyFrame": [],
        "Expr": [],
        "Series": [],
        "other": [],
    }

    expr_namespaces: dict[str, list[dict[str, Any]]] = {}

    for item in results:
        item_class = item.get("class")
        namespace = item.get("namespace")

        if item["type"] == "function":
            groups["functions"].append(item)
        elif item_class == "Expr" and namespace:
            if namespace not in expr_namespaces:
                expr_namespaces[namespace] = []
            expr_namespaces[namespace].append(item)
        elif item_class in groups:
            groups[item_class].append(item)
        else:
            groups["other"].append(item)

    for ns_name, ns_items in sorted(expr_namespaces.items()):
        groups[f"Expr.{ns_name}"] = ns_items

    return {k: v for k, v in groups.items() if v}


def format_grouped_results(
    query: str, grouped: dict[str, list[dict[str, Any]]], total: int
) -> str:
    """Format grouped search results as markdown."""
    lines = [f"# Search Results for '{query}'\n"]
    lines.append(f"Found {total} results:\n")

    order = ["functions", "DataFrame", "LazyFrame", "Series", "Expr"]

    def sort_key(group_name: str) -> tuple:
        if group_name in order:
            return (0, order.index(group_name))
        elif group_name.startswith("Expr."):
            return (1, group_name)
        else:
            return (2, group_name)

    for group_name in sorted(grouped.keys(), key=sort_key):
        items = grouped[group_name]

        if group_name == "functions":
            lines.append("## Top-level Functions\n")
        elif group_name.startswith("Expr."):
            lines.append(f"## {group_name} (Expression Namespace)\n")
        else:
            lines.append(f"## {group_name} Methods\n")

        for item in items:
            sig = item.get("signature", "")
            short_desc = item.get("short_description", "")

            if sig and sig != "(...)":
                sig_preview = sig if len(sig) < 60 else sig[:57] + "..."
                lines.append(f"- **{item['name']}**`{sig_preview}`")
            else:
                lines.append(f"- **{item['name']}**")

            if short_desc:
                lines.append(f"  {short_desc}")

        lines.append("")

    lines.append("---")
    lines.append("*Use `polars_get_docstring(name)` for full documentation.*")
    lines.append(
        "*Use `polars_browse(category)` to explore all methods in a class/namespace.*"
    )

    return "\n".join(lines)


def find_related_methods(name: str, index: dict[str, Any]) -> list[str]:
    """Find methods related to the given one."""
    related = []
    info = index.get(name)
    if not info:
        return related

    item_class = info.get("class")
    namespace = info.get("namespace")
    method_name = name.split(".")[-1]

    for cls in ["DataFrame", "LazyFrame", "Expr", "Series"]:
        if cls != item_class:
            candidate = f"{cls}.{method_name}"
            if candidate in index:
                related.append(candidate)
            if namespace:
                candidate = f"{cls}.{namespace}.{method_name}"
                if candidate in index:
                    related.append(candidate)

    if namespace and item_class:
        prefix = f"{item_class}.{namespace}."
        siblings = [k for k in index.keys() if k.startswith(prefix) and k != name]
        related.extend(siblings[:5])

    return related[:8]


@mcp.tool(
    name="polars_get_guide",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
async def polars_get_guide(guide: str) -> str:
    """Get a conceptual guide about Polars usage patterns.

    Available guides:
    - contexts: Expression contexts (select, filter, group_by, with_columns, over)
    - expressions: Expression system and composition patterns
    - lazy-api: Lazy evaluation and query optimization
    - pandas-to-polars: pandas to Polars translation guide

    Args:
        guide: Which guide to retrieve: 'contexts', 'expressions', 'lazy-api', or 'pandas-to-polars'
    """
    if guide not in GUIDE_FILES:
        available = ", ".join(GUIDE_FILES.keys())
        return f"Unknown guide: '{guide}'\n\nAvailable guides: {available}"
    return load_guide(GUIDE_FILES[guide])


@mcp.tool(
    name="polars_search_api",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
async def polars_search_api(
    query: str,
    limit: int = 20,
    class_filter: str | None = None,
    namespace_filter: str | None = None,
    response_format: str = "markdown",
) -> str:
    """Search the Polars API for functions, methods, and classes.

    Results are grouped by class/namespace for easy navigation.

    Args:
        query: Search term (e.g., "filter", "string", "aggregate")
        limit: Maximum results (default 20)
        class_filter: Limit to specific class: "DataFrame", "LazyFrame", "Expr", "Series"
        namespace_filter: Limit to namespace: "str", "dt", "list", "arr", "cat", "struct", "bin", "name", "meta"
        response_format: "markdown" (default) or "json"

    Examples:
        - polars_search_api("filter") -> all filter-related methods
        - polars_search_api("contains", class_filter="Expr", namespace_filter="str") -> Expr.str.contains
        - polars_search_api("date", namespace_filter="dt") -> all datetime methods
    """
    results = search_index(query, limit, class_filter, namespace_filter)

    if not results:
        msg = f"No results for '{query}'"
        if class_filter or namespace_filter:
            msg += f" with filters (class={class_filter}, namespace={namespace_filter})"
        msg += "\n\nTry:\n- Broader search terms\n- Remove filters\n- Use `polars_browse` to explore available methods"
        return msg

    if response_format == "json":
        return json.dumps(
            {"query": query, "count": len(results), "results": results}, indent=2
        )

    grouped = group_search_results(results)
    output = format_grouped_results(query, grouped, len(results))

    return output


@mcp.tool(
    name="polars_browse",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
async def polars_browse(category: str, limit: int = 50) -> str:
    """Browse all methods in a class or namespace.

    Use this to explore what's available without searching.

    Args:
        category: What to browse. Options:
            - "DataFrame", "LazyFrame", "Expr", "Series" - all methods in class
            - "Expr.str", "Expr.dt", "Expr.list", etc. - methods in namespace
            - "functions" - top-level functions like col(), lit(), when()
            - "namespaces" - list all available namespaces
        limit: Maximum methods to show (default 50)

    Examples:
        - polars_browse("namespaces") -> see all available namespaces
        - polars_browse("Expr.str") -> all string methods
        - polars_browse("functions") -> col, lit, when, concat, etc.
    """
    registry = get_namespace_registry()
    index = get_api_index()

    # Special case: list all namespaces
    if category.lower() == "namespaces":
        lines = ["# Available Namespaces\n"]
        lines.append('Use `polars_browse("Class.namespace")` to explore methods.\n')

        for class_name in ["DataFrame", "LazyFrame", "Expr", "Series"]:
            if class_name in registry:
                namespaces = [k for k in registry[class_name].keys() if k != "_methods"]
                if namespaces:
                    lines.append(f"## {class_name}")
                    for ns in sorted(namespaces):
                        count = len(registry[class_name][ns])
                        lines.append(f"- **{class_name}.{ns}** ({count} methods)")
                    lines.append("")

        return "\n".join(lines)

    parts = category.split(".")

    if len(parts) == 1:
        class_name = parts[0]
        if class_name == "functions":
            methods = registry.get("functions", {}).get("_methods", [])
        elif class_name in registry:
            methods = registry[class_name].get("_methods", [])
        else:
            return f"Unknown category: {category}\n\nValid options: DataFrame, LazyFrame, Expr, Series, functions, namespaces, or Class.namespace (e.g., Expr.str)"

        lines = [f"# {class_name} Methods\n"]
        lines.append(f"Showing {min(len(methods), limit)} of {len(methods)} methods:\n")

        for method in sorted(methods)[:limit]:
            full_name = (
                method if class_name == "functions" else f"{class_name}.{method}"
            )
            info = index.get(full_name, {})
            sig = info.get("signature", "")
            short_desc = info.get("short_description", "")

            if sig and sig != "(...)":
                sig_preview = sig if len(sig) < 50 else sig[:47] + "..."
                lines.append(f"- **{method}**`{sig_preview}`")
            else:
                lines.append(f"- **{method}**")

            if short_desc:
                lines.append(f"  {short_desc}")

        if class_name in registry:
            namespaces = [k for k in registry[class_name].keys() if k != "_methods"]
            if namespaces:
                lines.append(f"\n## Available Namespaces for {class_name}")
                for ns in sorted(namespaces):
                    lines.append(
                        f"- {class_name}.{ns} ({len(registry[class_name][ns])} methods)"
                    )

        return "\n".join(lines)

    elif len(parts) == 2:
        class_name, ns_name = parts
        if class_name not in registry or ns_name not in registry[class_name]:
            available = [
                k for k in registry.get(class_name, {}).keys() if k != "_methods"
            ]
            return f"Namespace '{ns_name}' not found in {class_name}.\n\nAvailable: {', '.join(available) if available else 'none'}"

        methods = registry[class_name][ns_name]

        lines = [f"# {class_name}.{ns_name} Namespace\n"]
        lines.append(f"Showing {min(len(methods), limit)} of {len(methods)} methods:\n")

        for method in sorted(methods)[:limit]:
            full_name = f"{class_name}.{ns_name}.{method}"
            info = index.get(full_name, {})
            sig = info.get("signature", "")
            short_desc = info.get("short_description", "")

            if sig and sig != "(...)":
                sig_preview = sig if len(sig) < 50 else sig[:47] + "..."
                lines.append(f"- **{method}**`{sig_preview}`")
            else:
                lines.append(f"- **{method}**")

            if short_desc:
                lines.append(f"  {short_desc}")

        return "\n".join(lines)

    return f"Invalid category format: {category}\n\nUse: 'DataFrame', 'Expr.str', 'functions', or 'namespaces'"


@mcp.tool(
    name="polars_get_docstring",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
async def polars_get_docstring(name: str, response_format: str = "markdown") -> str:
    """Get complete documentation for a specific Polars API element.

    Includes signature, full docstring, and related methods.

    Args:
        name: Full API name like 'DataFrame.filter', 'Expr.str.contains', 'col'
        response_format: "markdown" (default) or "json"
    """
    index = get_api_index()
    normalized = normalize_api_name(name)

    # Exact match
    if normalized in index:
        item = {"name": normalized, **index[normalized]}
    else:
        # Case-insensitive fallback
        name_lower = normalized.lower()
        matches = [k for k in index.keys() if k.lower() == name_lower]
        if not matches:
            similar = [k for k in index.keys() if name_lower in k.lower()][:5]
            msg = f"'{name}' not found."
            if similar:
                msg += f"\n\nDid you mean: {', '.join(similar)}"
            msg += (
                "\n\nUse `polars_search_api` to search, or `polars_browse` to explore."
            )
            return msg
        item = {"name": matches[0], **index[matches[0]]}

    if response_format == "json":
        return json.dumps(item, indent=2)

    lines = [f"# {item['full_name']}\n"]
    lines.append(f"**Type:** {item['type']}")

    if item.get("class"):
        lines.append(f"**Class:** {item['class']}")

    if item.get("namespace"):
        lines.append(f"**Namespace:** {item['namespace']}")

    if item["signature"]:
        lines.append(
            f"\n**Signature:**\n```python\n{item['name']}{item['signature']}\n```"
        )

    lines.append(f"\n## Documentation\n\n{item['docstring']}")

    related = find_related_methods(item["name"], index)
    if related:
        lines.append("\n## Related Methods")
        for rel in related:
            rel_info = index.get(rel, {})
            short_desc = rel_info.get("short_description", "")
            lines.append(f"- **{rel}**: {short_desc}" if short_desc else f"- **{rel}**")

    return "\n".join(lines)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
