# Polars MCP Server

MCP server providing Polars API reference documentation via introspection of the installed Polars package.

## Features

- Search Polars API by keyword
- Get full documentation for any Polars function, method, or class
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
├── polars_mcp.py          # Main server implementation
├── pyproject.toml         # Dependencies
├── guides/                # Conceptual documentation
│   ├── lazy-api.md       # Lazy API guide
│   ├── expressions.md    # Expression patterns
│   └── contexts.md       # Context behaviors
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

## Available Resources

MCP resources provide conceptual guides that LLMs can reference:

- `polars://guide/lazy-api` - Complete lazy API guide with optimization patterns
- `polars://guide/expressions` - Expression system and composition
- `polars://guide/contexts` - How expressions work in different contexts

These guides emphasize **lazy API usage** for optimal performance.

## Available Tools

### polars_search_api

Search for Polars API elements by keyword.

**Parameters:**

- `query` (str): Search term (e.g., "groupby", "filter", "lazy")
- `limit` (int, optional): Max results (default: 20)
- `response_format` (str, optional): "markdown" or "json" (default: "markdown")

**Example:**

```
Query: "groupby"
Returns: List of all groupby-related functions and methods
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

## Available Resources

Resources provide conceptual guides that help understand how to compose expressions and build complex queries.

### polars://guide/expressions

Complete guide to the Polars expression system:

- What expressions are and how they work
- Composition patterns (chaining, operators, nesting)
- Expression expansion
- Common patterns (aggregations, window functions, conditionals)

### polars://guide/contexts

Guide to Polars contexts:

- What contexts are (select, filter, group_by, with_columns)
- When to use each context
- How contexts affect expression evaluation
- Common patterns for each context

### polars://guide/lazy-evaluation

Guide to lazy evaluation:

- Eager (DataFrame) vs Lazy (LazyFrame)
- Query optimization (predicate pushdown, projection pushdown)
- When to use each approach
- Best practices for large datasets
