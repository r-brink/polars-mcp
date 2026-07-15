"""MCP server that introspects the locally installed Polars package.

Companion to the Polars agent skill
(https://github.com/polars-inc/skills/tree/main/polars): the skill covers
methodology, this server verifies API names and signatures. What agents see
of this server is the FastMCP ``instructions`` string and the tool
docstrings below — keep those aligned with the skill when either changes.
"""

import inspect
import json
from typing import Any, Literal

import polars as pl
from fastmcp import Context, FastMCP
from fastmcp.server.lifespan import lifespan


@lifespan
async def api_lifespan(server: FastMCP):
    """Build the API index once at startup and make it available to all tools."""
    api_index, namespace_registry = build_api_index()
    yield {"api_index": api_index, "namespace_registry": namespace_registry}


mcp = FastMCP(
    "polars-mcp",
    version="0.3.0",
    website_url="https://github.com/r-brink/polars-mcp",
    lifespan=api_lifespan,
    instructions=(
        "Companion tool for the Polars agent skill "
        "(https://github.com/polars-inc/skills/tree/main/polars). If a 'polars' "
        "skill is available in your environment, load it for methodology "
        "(lazy API, expression contexts, pandas translation) and use these "
        "tools to verify Polars API method names and signatures before writing "
        "code. Three tools available: polars_search_api to find methods by "
        "keyword, polars_browse to explore a class or namespace, "
        "polars_get_docstring for the full signature and documentation of a "
        "specific method. All results reflect the locally installed Polars "
        f"version: {pl.__version__}."
    ),
)

MAIN_CLASSES = [
    ("DataFrame", pl.DataFrame),
    ("LazyFrame", pl.LazyFrame),
    ("Series", pl.Series),
    ("Expr", pl.Expr),
]


def _discover_namespaces() -> tuple[str, ...]:
    names: set[str] = set()
    for _, cls in MAIN_CLASSES:
        names |= set(getattr(cls, "_accessors", ()))
    # fall back to the namespaces known at time of writing if the internal
    # _accessors attribute ever disappears (Literal needs >= 1 value)
    if not names:
        names = {"str", "dt", "list", "arr", "cat", "struct", "bin", "name", "meta"}
    return tuple(sorted(names))


ClassFilter = Literal["DataFrame", "LazyFrame", "Expr", "Series"]
NamespaceFilter = Literal[_discover_namespaces()]  # type: ignore[valid-type]
ResponseFormat = Literal["markdown", "json"]


def _create_test_instance(class_name: str, cls: type) -> Any:
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
    if not docstring:
        return ""
    first_line = docstring.split("\n")[0].strip()
    if "." in first_line:
        return first_line.split(".")[0] + "."
    return first_line[:100] + ("..." if len(first_line) > 100 else "")


def build_api_index() -> tuple[dict[str, Any], dict[str, dict[str, list[str]]]]:
    index = {}
    namespace_registry: dict[str, dict[str, list[str]]] = {}

    for class_name, cls in MAIN_CLASSES:
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

    namespace_registry["functions"] = {"_methods": []}

    for name in dir(pl):
        if name.startswith("_"):
            continue
        obj = getattr(pl, name)
        if inspect.isclass(obj) and name in [c[0] for c in MAIN_CLASSES]:
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


def normalize_api_name(name: str) -> str:
    name = name.strip()
    if name.startswith("polars."):
        return name[7:]
    elif name.startswith("pl."):
        return name[3:]
    return name


def calculate_match_score(
    term: str, name: str, docstring: str, short_desc: str
) -> float:
    t = normalize_api_name(term).lower()
    n, d, s = name.lower(), docstring.lower(), short_desc.lower()
    if t == n or n.endswith(f".{t}"):
        return 5.0
    if f".{t}." in n:
        return 3.0
    if t in n:
        score = 2.0 + (0.5 if n.startswith(t) else 0) + len(t) / len(n)
        return score
    if t in s:
        return 1.5
    if t in d:
        return 0.5
    return 0.0


def search_index(
    index: dict[str, Any],
    query: str,
    limit: int = 20,
    class_filter: str | None = None,
    namespace_filter: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    terms = [t.strip() for t in query.lower().replace(",", " ").split() if t.strip()]
    if not terms:
        return [], 0

    results = []
    for name, info in index.items():
        if class_filter and info.get("class") != class_filter:
            continue
        if namespace_filter and info.get("namespace") != namespace_filter:
            continue
        total_score = sum(
            calculate_match_score(
                term, name, info["docstring"], info.get("short_description", "")
            )
            for term in terms
        )
        if total_score > 0:
            results.append({"score": total_score, "name": name, **info})

    results.sort(key=lambda x: (-x["score"], x["name"]))
    return results[:limit], len(results)


def group_search_results(
    results: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
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
            expr_namespaces.setdefault(namespace, []).append(item)
        elif item_class in groups:
            groups[item_class].append(item)
        else:
            groups["other"].append(item)

    for ns_name, ns_items in sorted(expr_namespaces.items()):
        groups[f"Expr.{ns_name}"] = ns_items

    return {k: v for k, v in groups.items() if v}


def format_grouped_results(
    query: str, grouped: dict[str, list[dict[str, Any]]], shown: int, total: int
) -> str:
    lines = [f"# Search Results for '{query}'\n"]
    if total > shown:
        lines.append(
            f"Showing top {shown} of {total} matches "
            "(raise `limit` or add filters to narrow down):\n"
        )
    else:
        lines.append(f"Found {total} results:\n")
    order = ["functions", "DataFrame", "LazyFrame", "Series", "Expr"]

    def sort_key(g: str) -> tuple:
        if g in order:
            return (0, order.index(g))
        elif g.startswith("Expr."):
            return (1, g)
        return (2, g)

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

    lines.extend(
        [
            "---",
            "*Use `polars_get_docstring(name)` for full documentation.*",
            "*Use `polars_browse(category)` to explore all methods in a class/namespace.*",
        ]
    )
    return "\n".join(lines)


def find_related_methods(name: str, index: dict[str, Any]) -> list[str]:
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
    name="polars_search_api",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
async def polars_search_api(
    query: str,
    ctx: Context,
    limit: int = 20,
    class_filter: ClassFilter | None = None,
    namespace_filter: NamespaceFilter | None = None,
    response_format: ResponseFormat = "markdown",
) -> str:
    """Find Polars methods and functions by keyword.

    Use before writing any method call you are not certain about. Results
    reflect the locally installed Polars version.

    Args:
        query: Search term (e.g., "filter", "string upper", "aggregate")
        limit: Maximum results (default 20)
        class_filter: Limit to "DataFrame", "LazyFrame", "Expr", or "Series"
        namespace_filter: Limit to an expression namespace such as "str" or
            "dt" (valid values are introspected from the installed Polars;
            see polars_browse("namespaces") for the full list)
        response_format: "markdown" (default) or "json"

    Examples:
        polars_search_api("filter")
        polars_search_api("contains", class_filter="Expr", namespace_filter="str")
        polars_search_api("date", namespace_filter="dt")
    """
    index = ctx.lifespan_context["api_index"]
    results, total = search_index(index, query, limit, class_filter, namespace_filter)

    if not results:
        msg = f"No results for '{query}'"
        if class_filter or namespace_filter:
            msg += f" with filters (class={class_filter}, namespace={namespace_filter})"
        msg += "\n\nTry:\n- Broader search terms\n- Remove filters\n- Use polars_browse to explore available methods"
        return msg

    if response_format == "json":
        return json.dumps(
            {
                "query": query,
                "shown": len(results),
                "total_matches": total,
                "results": results,
            },
            indent=2,
        )

    grouped = group_search_results(results)
    return format_grouped_results(query, grouped, len(results), total)


@mcp.tool(
    name="polars_browse",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
async def polars_browse(category: str, ctx: Context, limit: int = 50) -> str:
    """Explore all methods in a Polars class or namespace.

    Use to see what is available when you do not know the exact method name.
    Particularly useful for namespace methods (str, dt, list, struct).

    Args:
        category: What to browse:
            - "DataFrame", "LazyFrame", "Expr", "Series"
            - "Expr.str", "Expr.dt", "Expr.list", "Expr.struct", etc.
            - "functions" -- top-level functions: col(), lit(), when(), etc.
            - "namespaces" -- list all available namespaces
        limit: Maximum methods to show (default 50)

    Examples:
        polars_browse("namespaces")
        polars_browse("Expr.str")
        polars_browse("functions")
    """
    registry = ctx.lifespan_context["namespace_registry"]
    index = ctx.lifespan_context["api_index"]

    if category.lower() == "namespaces":
        lines = ["# Available Namespaces\n"]
        lines.append('Use polars_browse("Class.namespace") to explore methods.\n')
        for class_name in ["DataFrame", "LazyFrame", "Expr", "Series"]:
            if class_name in registry:
                namespaces = [k for k in registry[class_name].keys() if k != "_methods"]
                if namespaces:
                    lines.append(f"## {class_name}")
                    for ns in sorted(namespaces):
                        lines.append(
                            f"- **{class_name}.{ns}** ({len(registry[class_name][ns])} methods)"
                        )
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
            return (
                f"Unknown category: {category}\n\n"
                "Valid options: DataFrame, LazyFrame, Expr, Series, functions, "
                "namespaces, or Class.namespace (e.g., Expr.str)"
            )

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
            return (
                f"Namespace '{ns_name}' not found in {class_name}.\n\n"
                f"Available: {', '.join(available) if available else 'none'}"
            )

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

    return (
        f"Invalid category format: {category}\n\n"
        "Use: 'DataFrame', 'Expr.str', 'functions', or 'namespaces'"
    )


@mcp.tool(
    name="polars_get_docstring",
    annotations={"readOnlyHint": True, "idempotentHint": True},
)
async def polars_get_docstring(
    name: str, ctx: Context, response_format: ResponseFormat = "markdown"
) -> str:
    """Get the full signature and documentation for a specific Polars API element.

    Use to confirm exact parameter names, types, and defaults before writing
    a method call. Reflects the locally installed Polars version.

    Args:
        name: Full API name, e.g. "DataFrame.filter", "Expr.str.contains", "col"
        response_format: "markdown" (default) or "json"

    Examples:
        polars_get_docstring("LazyFrame.collect")
        polars_get_docstring("Expr.str.to_uppercase")
        polars_get_docstring("Expr.over")
    """
    index = ctx.lifespan_context["api_index"]
    normalized = normalize_api_name(name)

    if normalized in index:
        item = {"name": normalized, **index[normalized]}
    else:
        name_lower = normalized.lower()
        matches = [k for k in index.keys() if k.lower() == name_lower]
        if not matches:
            similar = [k for k in index.keys() if name_lower in k.lower()][:5]
            msg = f"'{name}' not found."
            if similar:
                msg += f"\n\nDid you mean: {', '.join(similar)}"
            msg += "\n\nUse polars_search_api to search, or polars_browse to explore."
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
