import inspect
import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl
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
    PANDAS_TO_POLARS = "pandas-to-polars"


# API Index - built on first access
_api_index: Optional[Dict[str, Any]] = None


def _create_test_instance(class_name: str, cls: type) -> Any:
    """Create a minimal test instance for namespace inspection.

    Args:
        class_name: Name of the class ("Expr", "Series", etc.)
        cls: The actual class object

    Returns:
        A minimal instance of the class, or None if creation fails
    """
    try:
        if class_name == "Expr":
            return pl.col("_")
        elif class_name == "Series":
            return pl.Series("_", [1])
        elif class_name == "DataFrame":
            return pl.DataFrame({"_": [1]})
        elif class_name == "LazyFrame":
            return pl.DataFrame({"_": [1]}).lazy()
    except Exception:
        return None

    return None


def build_api_index() -> Dict[str, Any]:
    """Build index of Polars API by introspecting the module.

    Uses _accessors metadata + minimal test instances for namespace discovery.

    Returns:
        Dictionary mapping names to API metadata.
    """

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

        # Get registered namespace accessors (Polars tracks all namespaces)
        namespace_names = getattr(cls, "_accessors", set())

        # Create a test instance once for this class (for namespace inspection)
        test_instance = (
            _create_test_instance(class_name, cls) if namespace_names else None
        )

        # Add all public methods
        for method_name in dir(cls):
            if method_name.startswith("_"):
                continue

            try:
                # Use getattr_static to avoid triggering descriptors for regular methods
                method = inspect.getattr_static(cls, method_name)
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

                # Check if this is a registered namespace accessor
                if method_name in namespace_names and test_instance is not None:
                    try:
                        # Access the namespace property on the test instance
                        namespace_obj = getattr(test_instance, method_name)
                        namespace_class = type(namespace_obj)

                        # Index all methods in the namespace class
                        for ns_method_name in dir(namespace_class):
                            if ns_method_name.startswith("_"):
                                continue

                            try:
                                # Get the method from the namespace class
                                ns_method = getattr(namespace_class, ns_method_name)

                                # Only index callable methods
                                if not callable(ns_method):
                                    continue

                                ns_doc = inspect.getdoc(ns_method)
                                try:
                                    ns_sig = inspect.signature(ns_method)
                                    ns_sig_str = str(ns_sig)
                                except (ValueError, TypeError):
                                    ns_sig_str = "(...)"

                                # Index with full dotted name: Class.namespace.method
                                ns_full_name = (
                                    f"{class_name}.{method_name}.{ns_method_name}"
                                )
                                index[ns_full_name] = {
                                    "type": "namespace_method",
                                    "class": class_name,
                                    "namespace": method_name,
                                    "full_name": f"polars.{ns_full_name}",
                                    "docstring": ns_doc or "No documentation available",
                                    "signature": ns_sig_str,
                                }
                            except (AttributeError, TypeError):
                                # Skip methods that can't be introspected
                                continue
                    except Exception:
                        # If namespace inspection fails, continue with other methods
                        continue

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


def calculate_match_score(query: str, name: str, docstring: str) -> float:
    """Score matches with namespace priority and prefix handling."""
    q = query.lower().strip()

    # Strip common import aliases
    if q.startswith("pl."):
        q = q[3:]
    elif q.startswith("polars."):
        q = q[7:]

    n, d = name.lower(), docstring.lower()

    # Namespace match
    if f".{q}." in n or n.endswith(f".{q}"):
        return 2.0

    # Name matches with differentiation
    if q in n:
        score = 1.0

        # Exact match bonus
        if q == n:
            score += 3.0  # Total: 4.0

        # Starts with bonus
        elif n.startswith(q):
            score += 1.0  # Total: 2.0

        # Coverage bonus (longer match = higher score)
        score += len(q) / len(n)  # 0.0 to 1.0

        return score

    # Docstring match
    if q in d:
        return 0.5

    return 0.0


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

    results = [
        {
            "score": calculate_match_score(query, name, info["docstring"]),
            "name": name,
            **info,
        }
        for name, info in index.items()
    ]

    results = [r for r in results if r["score"] > 0]
    results.sort(
        key=lambda x: (-x["score"], x["name"])
    )  # Score desc, then alphabetical

    return results[:limit]


def format_api_item_markdown(item: Dict[str, Any]) -> str:
    """Format an API item as markdown."""
    lines = [f"## {item['full_name']}"]

    lines.append(f"\n**Type:** {item['type']}")

    if item["signature"]:
        lines.append(f"\n**Signature:** `{item['signature']}`")

    if item.get("class"):
        lines.append(f"\n**Class:** {item['class']}")

    if item.get("namespace"):
        lines.append(f"\n**Namespace:** {item['namespace']}")

    lines.append(f"\n### Documentation\n\n{item['docstring']}")

    return "\n".join(lines)


# Tool Input Models
class GetGuideInput(BaseModel):
    """Input for getting a conceptual guide."""

    model_config = ConfigDict(str_strip_whitespace=True)

    guide: GuideType = Field(
        ...,
        description="Which guide to retrieve: 'contexts' for expression contexts, 'expressions' for expression patterns, 'lazy-api' for lazy evaluation, 'pandas-to-polars' for pandas to Polars migration",
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
        "title": "Get Polars Concepts Guides",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def polars_get_guide(params: GetGuideInput) -> str:
    """Get a conceptual guide about Polars usage patterns.

    For conceptual "how to" questions, combine this with polars_search_api to get both:
    - Conceptual understanding (from the guide)
    - Concrete available functions (from API search)

    Example: For "how do I filter data?", use BOTH:
    - polars_get_guide(guide='contexts') for filtering concepts
    - polars_search_api(query='filter') for specific filter methods

    Args:
        params (GetGuideInput): Parameters containing:
            - guide (str): Which guide to get - 'contexts', 'expressions', 'lazy-api', or 'pandas-to-polars'

    Returns:
        str: Complete guide content in markdown format

    Available Guides:
        - contexts: Expression contexts (select, filter, group_by, with_columns, over)
                   When to use each context, how they work, combining contexts
        - expressions: Expression system and composition patterns
                      What expressions are, chaining, operators, expansion
        - lazy-api: Lazy evaluation and query optimization
                   LazyFrame vs DataFrame, optimization patterns, best practices
        - pandas-to-polars: Complete pandas to Polars translation guide
                         Side-by-side comparisons, common patterns, anti-patterns
    """
    try:
        guide_files = {
            GuideType.CONTEXTS: "contexts.md",
            GuideType.EXPRESSIONS: "expressions.md",
            GuideType.LAZY_API: "lazy-api.md",
            GuideType.PANDAS_TO_POLARS: "pandas-to-polars.md",
        }

        filename = guide_files[params.guide]
        content = load_guide(filename)

        return content

    except Exception as e:
        return f"Error loading guide: {str(e)}"


def format_search_results(
    query: str,
    results: List[Dict[str, Any]],
    total_count: Optional[int] = None,
    is_truncated: bool = False,
) -> str:
    """Format search results as markdown.

    Args:
        query: The search query
        results: List of matching API items
        total_count: Total number of matches before truncation (optional)
        is_truncated: Whether results were truncated due to size

    Returns:
        Formatted markdown string
    """
    lines = [f"# Search Results for '{query}'\n"]

    if is_truncated and total_count:
        lines.append(
            f"Showing {len(results)} of {total_count} results (truncated due to size):\n"
        )
    else:
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

    if is_truncated:
        lines.append(
            "\n*Note: Results were truncated to fit size limits. "
            "Try a more specific query or use fewer results.*"
        )

    return "\n".join(lines)


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
async def polars_search_api(
    query: str, limit: int = 20, response_format: str = "markdown"
):
    """Search the Polars API for functions, methods, and classes.

    For conceptual questions, combine this with polars_get_guide to get both:
    - Available functions and their signatures (from this search)
    - Conceptual understanding of when/how to use them (from guides)

    For specific API lookup (e.g., "what parameters does filter take?"), you can use this alone or follow up with polars_get_docstring.

    Searches through all public Polars API elements including DataFrame methods,
    LazyFrame methods, Series methods, expressions, and top-level functions.

    Args:
        query (str): Search term to match against API names and documentation
        limit (int): Maximum number of results to return (default 20)
        response_format (str): 'markdown' or 'json'

    Returns:
        str: Search results in the specified format
    """
    try:
        # Perform the search
        results = search_index(query, limit)

        # Handle no results
        if not results:
            return f"No results found for query: '{query}'\n\nTry:\n- Using different search terms\n- Checking spelling\n- Using polars_get_guide for conceptual questions"

        # JSON format - always return full results
        if response_format == "json":
            return json.dumps(
                {"query": query, "count": len(results), "results": results},
                indent=2,
            )

        # Markdown format with size checking
        result = format_search_results(query, results, is_truncated=False)

        # Check if we exceed character limit
        if len(result) > CHARACTER_LIMIT:
            # Truncate results to approximately half
            original_count = len(results)
            truncated_count = max(1, len(results) // 2)  # Keep at least 1 result
            results = results[:truncated_count]

            # Rebuild with truncated results
            result = format_search_results(
                query, results, total_count=original_count, is_truncated=True
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

    Use this when you need the full docstring for a specific function/method you already know the name of.

    For discovering what functions are available, use polars_search_api first.
    For understanding concepts, combine polars_get_guide with polars_search_api.

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


def main():
    mcp.run()


if __name__ == "__main__":
    main()
