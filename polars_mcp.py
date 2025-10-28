"""Polars MCP Server - API Reference via Introspection

A minimal MCP server that provides access to Polars API documentation
by introspecting the installed Polars package.
"""

import inspect
import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

# Initialize MCP server
mcp = FastMCP("polars_mcp")

# Constants
CHARACTER_LIMIT = 25000
GUIDES_DIR = Path(__file__).parent / "guides"


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


class ResponseFormat(str, Enum):
    """Output format for tool responses."""

    MARKDOWN = "markdown"
    JSON = "json"


class GuideType(str, Enum):
    """Available guide types."""

    CONTEXTS = "contexts"
    EXPRESSIONS = "expressions"
    LAZY_API = "lazy-api"


# API Index - built on first access
_api_index: Optional[Dict[str, Any]] = None


def build_api_index() -> Dict[str, Any]:
    """Build index of Polars API by introspecting the module.

    Returns:
        Dictionary mapping names to API metadata.
    """
    import polars as pl

    index = {}

    # Main classes to document (with methods)
    main_classes = [
        ("DataFrame", pl.DataFrame),
        ("LazyFrame", pl.LazyFrame),
        ("Series", pl.Series),
        ("Expr", pl.Expr),
    ]

    # Index main classes and their methods
    for class_name, cls in main_classes:
        # Add the class itself
        doc = inspect.getdoc(cls)
        index[class_name] = {
            "type": "class",
            "full_name": f"polars.{class_name}",
            "docstring": doc or "No documentation available",
            "signature": "",
        }

        # Add all public methods
        for method_name in dir(cls):
            if method_name.startswith("_"):
                continue

            try:
                method = getattr(cls, method_name)
                method_doc = inspect.getdoc(method)

                # Try to get signature
                try:
                    sig = inspect.signature(method)
                    sig_str = str(sig)
                except (ValueError, TypeError):
                    sig_str = "(...)"

                full_name = f"{class_name}.{method_name}"
                index[full_name] = {
                    "type": "method",
                    "class": class_name,
                    "full_name": f"polars.{full_name}",
                    "docstring": method_doc or "No documentation available",
                    "signature": sig_str,
                }
            except (AttributeError, TypeError):
                continue

    # Index top-level functions and classes
    for name in dir(pl):
        if name.startswith("_"):
            continue

        obj = getattr(pl, name)

        # Skip classes we already indexed with methods
        if inspect.isclass(obj) and name in [c[0] for c in main_classes]:
            continue

        # Add other classes
        if inspect.isclass(obj):
            doc = inspect.getdoc(obj)
            index[name] = {
                "type": "class",
                "full_name": f"polars.{name}",
                "docstring": doc or "No documentation available",
                "signature": "",
            }
        # Add functions
        elif callable(obj):
            doc = inspect.getdoc(obj)

            try:
                sig = inspect.signature(obj)
                sig_str = str(sig)
            except (ValueError, TypeError):
                sig_str = "(...)"

            index[name] = {
                "type": "function",
                "full_name": f"polars.{name}",
                "docstring": doc or "No documentation available",
                "signature": sig_str,
            }

    return index


def get_api_index() -> Dict[str, Any]:
    """Get or build the API index."""
    global _api_index
    if _api_index is None:
        _api_index = build_api_index()
    return _api_index


def search_index(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search the API index for matching items.

    Args:
        query: Search query (matches against names and docstrings)
               Can be multiple space-separated terms (OR logic)
        limit: Maximum number of results

    Returns:
        List of matching API items, prioritizing name matches
    """
    index = get_api_index()

    # Split query into terms for OR logic
    query_terms = [term.strip().lower() for term in query.split() if term.strip()]

    name_matches = []  # Matches in name (higher priority)
    doc_matches = []  # Matches only in docstring (lower priority)
    seen = set()  # Track which items we've already added

    for name, info in index.items():
        name_lower = name.lower()
        doc_lower = info["docstring"].lower()

        # Check if any query term matches
        for term in query_terms:
            if name not in seen:
                if term in name_lower:
                    # Name match - high priority
                    name_matches.append({"name": name, **info})
                    seen.add(name)
                    break
                elif term in doc_lower:
                    # Docstring match - lower priority
                    doc_matches.append({"name": name, **info})
                    seen.add(name)
                    break

    # Combine results: name matches first, then doc matches
    results = name_matches + doc_matches
    return results[:limit]


def format_api_item_markdown(item: Dict[str, Any]) -> str:
    """Format an API item as markdown."""
    lines = [f"## {item['full_name']}"]

    lines.append(f"\n**Type:** {item['type']}")

    if item["signature"]:
        lines.append(f"\n**Signature:** `{item['signature']}`")

    if item.get("class"):
        lines.append(f"\n**Class:** {item['class']}")

    lines.append(f"\n### Documentation\n\n{item['docstring']}")

    return "\n".join(lines)


# Tool Input Models
class GetGuideInput(BaseModel):
    """Input for getting a conceptual guide."""

    model_config = ConfigDict(str_strip_whitespace=True)

    guide: GuideType = Field(
        ...,
        description="Which guide to retrieve: 'contexts' for expression contexts, 'expressions' for expression patterns, 'lazy-api' for lazy evaluation",
    )


class SearchAPIInput(BaseModel):
    """Input for searching Polars API."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(
        ...,
        description="Search query to find API functions, methods, or classes. Can include multiple space-separated terms (e.g., 'Int32 Float64', 'filter select')",
        min_length=1,
        max_length=100,
    )
    limit: int = Field(
        default=20, description="Maximum number of results to return", ge=1, le=100
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable",
    )


class GetDocstringInput(BaseModel):
    """Input for getting full documentation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(
        ...,
        description="Full name of API element (e.g., 'DataFrame.filter', 'col', 'LazyFrame')",
        min_length=1,
        max_length=200,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable",
    )


# MCP Tools
@mcp.tool(
    name="polars_get_guide",
    annotations={
        "title": "Get Polars Conceptual Guide",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def polars_get_guide(params: GetGuideInput) -> str:
    """Get a conceptual guide about Polars usage patterns.

    IMPORTANT: Use this tool FIRST for conceptual "how to" questions like:
    - "how do I use contexts?" → use guide='contexts'
    - "when should I use group_by?" → use guide='contexts'
    - "what's the difference between select and with_columns?" → use guide='contexts'
    - "how do expressions work?" → use guide='expressions'
    - "should I use lazy or eager?" → use guide='lazy-api'

    Only use the API search tools (polars_search_api, polars_get_docstring) after
    checking the relevant guide, or when looking up specific function names.

    Args:
        params (GetGuideInput): Parameters containing:
            - guide (str): Which guide to get - 'contexts', 'expressions', or 'lazy-api'

    Returns:
        str: Complete guide content in markdown format

    Available Guides:
        - contexts: Expression contexts (select, filter, group_by, with_columns, over)
                   When to use each context, how they work, combining contexts
        - expressions: Expression system and composition patterns
                      What expressions are, chaining, operators, expansion
        - lazy-api: Lazy evaluation and query optimization
                   LazyFrame vs DataFrame, optimization patterns, best practices
    """
    try:
        guide_files = {
            GuideType.CONTEXTS: "contexts.md",
            GuideType.EXPRESSIONS: "expressions.md",
            GuideType.LAZY_API: "lazy-api.md",
        }

        filename = guide_files[params.guide]
        content = load_guide(filename)

        return content

    except Exception as e:
        return f"Error loading guide: {str(e)}"


@mcp.tool(
    name="polars_search_api",
    annotations={
        "title": "Search Polars API",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def polars_search_api(params: SearchAPIInput) -> str:
    """Search the Polars API reference for functions, methods, and classes.

    IMPORTANT: For conceptual "how to" questions, use polars_get_guide FIRST:
    - Questions about contexts? → polars_get_guide(guide='contexts')
    - Questions about expressions? → polars_get_guide(guide='expressions')
    - Questions about lazy/eager? → polars_get_guide(guide='lazy-api')

    Only use this tool for looking up specific API function names or browsing
    what methods are available.

    Searches through all public Polars API elements including DataFrame methods,
    LazyFrame methods, Series methods, expressions, and top-level functions.

    Args:
        params (SearchAPIInput): Search parameters containing:
            - query (str): Search term to match against API names and documentation
            - limit (int): Maximum number of results to return (default 20)
            - response_format (str): 'markdown' or 'json'

    Returns:
        str: Search results in the specified format
    """
    try:
        results = search_index(params.query, params.limit)

        if not results:
            return f"No results found for query: {params.query}"

        if params.response_format == ResponseFormat.JSON:
            return json.dumps(
                {"query": params.query, "count": len(results), "results": results},
                indent=2,
            )
        else:
            # Markdown format
            lines = [f"# Search Results for '{params.query}'\n"]
            lines.append(f"Found {len(results)} results:\n")

            for i, item in enumerate(results, 1):
                lines.append(f"### {i}. {item['full_name']}")
                lines.append(f"**Type:** {item['type']}")

                # Show truncated docstring
                doc = item["docstring"]
                if len(doc) > 200:
                    doc = doc[:200] + "..."
                lines.append(f"{doc}\n")

            lines.append(
                "\n*Use `polars_get_docstring` with the full name to see complete documentation.*"
            )

            result = "\n".join(lines)

            # Check character limit
            if len(result) > CHARACTER_LIMIT:
                truncated_results = results[: len(results) // 2]
                return polars_search_api(
                    SearchAPIInput(
                        query=params.query,
                        limit=len(truncated_results),
                        response_format=params.response_format,
                    )
                )

            return result

    except Exception as e:
        return f"Error searching API: {str(e)}"


@mcp.tool(
    name="polars_get_docstring",
    annotations={
        "title": "Get Polars API docstrings",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def polars_get_docstring(params: GetDocstringInput) -> str:
    """Get complete documentation for a specific Polars API element.

    IMPORTANT: For conceptual "how to" questions, use polars_get_guide FIRST:
    - Questions about contexts? → polars_get_guide(guide='contexts')
    - Questions about expressions? → polars_get_guide(guide='expressions')
    - Questions about lazy/eager? → polars_get_guide(guide='lazy-api')

    Only use this tool when you need the full docstring for a specific API element
    that you already know the name of.

    Retrieves the full docstring, signature, and metadata for any public
    Polars function, method, or class.

    Args:
        params (GetDocstringInput): Parameters containing:
            - name (str): Full name like 'DataFrame.filter', 'col', 'LazyFrame'
            - response_format (str): 'markdown' or 'json'

    Returns:
        str: Complete documentation in the specified format
    """
    try:
        index = get_api_index()

        # Try exact match first
        if params.name in index:
            item = {"name": params.name, **index[params.name]}
        else:
            # Try case-insensitive match
            name_lower = params.name.lower()
            matches = [k for k in index.keys() if k.lower() == name_lower]

            if not matches:
                return f"API element '{params.name}' not found. Use polars_search_api to find the correct name."

            item = {"name": matches[0], **index[matches[0]]}

        if params.response_format == ResponseFormat.JSON:
            return json.dumps(item, indent=2)
        else:
            return format_api_item_markdown(item)

    except Exception as e:
        return f"Error retrieving documentation: {str(e)}"


if __name__ == "__main__":
    mcp.run()
