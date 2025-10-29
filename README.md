# Polars MCP Server

MCP server providing Polars API reference documentation via introspection of the installed Polars package and additional information via conceptual guides.

## Features

- Polars API searchable by keyword
- Full documentation for any Polars function, method, or class
- Conceptual guides for Polars usage patterns
- pandas to Polars translation guide
- Always up-to-date with your installed Polars version
- No external dependencies or pre-built docs needed

## Installation

Requires Python 3.10+ and uv.

```bash
# Clone repository
git clone https://github.com/r-brink/polars-mcp
cd polars-mcp

# Install with uv
uv pip install .
```

## Project Structure

```
polars-mcp/
├── polars_mcp.py              # Main server implementation
├── pyproject.toml             # Dependencies
├── guides/                    # Conceptual documentation
│   ├── lazy-api.md           # Lazy API guide
│   ├── expressions.md        # Expression patterns
│   ├── contexts.md           # Context behaviors
│   └── pandas-to-polars.md   # pandas to Polars translation guide
└── README.md
```

## Usage

Refer to the documentation on the Model Context Protocol website on how to [connect to local MCP servers](https://modelcontextprotocol.io/docs/develop/connect-local-servers) for an extensive guide.

```json
{
    "mcpServers": {
        "polars": {
            "command": "uv",
            "args": [
                "--directory",
                "/path/to/polars-mcp",
                "run",
                "polars_mcp.py"
            ]
        }
    }
}
```

### Standalone Testing

```bash
# Test the server
python polars_mcp.py --help

# Run with MCP inspector
npx @modelcontextprotocol/inspector uv run polars_mcp.py
```

## Available Tools

### polars_get_guide

Get conceptual guides about Polars usage patterns.

**Parameters:**

- `guide` (str): Which guide to retrieve
    - `'contexts'` - Expression contexts (select, filter, group_by, with_columns, over)
    - `'expressions'` - Expression system and composition patterns
    - `'lazy-api'` - Lazy evaluation and query optimization
    - `'pandas-to-polars'` - Complete pandas to Polars translation guide

**Example:**

```
Guide: "pandas-to-polars"
Returns: Complete guide with side-by-side pandas/Polars comparisons
```

### polars_search_api

Search for Polars API elements by keyword.

**Parameters:**

- `query` (str): Search term (e.g., "groupby", "filter", "lazy")
- `limit` (int, optional): Max results (default: 20)
- `response_format` (str, optional): "markdown" or "json" (default: "markdown")

**Example:**

```
Query: "group by"
Returns: List of all group by-related functions and methods
```

### polars_get_docstring

Get complete documentation for a specific API element.

**Parameters:**

- `name` (str): Full name (e.g., "DataFrame.filter", "col", "LazyFrame")
- `response_format` (str, optional): "markdown" or "json" (default: "markdown")

**Example:**

```
Name: "DataFrame.filter"
Returns: Full documentation with signature and examples
```

## Available Guides

### polars://guides/contexts

Guide to Polars expression contexts:

- What contexts are (select, filter, group_by, with_columns)
- When to use each context
- How contexts affect expression evaluation
- Common patterns for each context

### polars://guides/expressions

Complete guide to the Polars expression system:

- What expressions are and how they work
- Composition patterns (chaining, operators, nesting)
- Expression expansion
- Common patterns (aggregations, window functions, conditionals)

### polars://guides/lazy-api

Guide to lazy evaluation:

- Eager (DataFrame) vs Lazy (LazyFrame)
- Query optimization (predicate pushdown, projection pushdown)
- When to use each approach
- Best practices for large datasets

### polars://guides/pandas-to-polars

- Side-by-side syntax comparisons for all common operations
- Reading data, column selection, filtering, sorting
- Group by aggregations and window functions
- Joins (including semi/anti joins)
- String, datetime, and missing value operations
- Complete worked examples
- Anti-patterns to avoid
- Performance optimization checklist

This guide is specifically optimized for LLMs to quickly translate pandas code to efficient Polars queries.

## Why This Server?

- **Always current**: Documentation reflects your installed Polars version
- **Lightweight**: No large pre-built documentation databases
- **Comprehensive**: Combines API reference with conceptual guides
- **Migration-friendly**: Includes complete pandas translation guide
- **Fast**: Uses introspection for instant access to documentation

## Contributing

Contributions welcome! Please submit issues or pull requests.
