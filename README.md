# Polars MCP Server

A lightweight MCP server that introspects your locally installed [Polars](https://pola.rs) package, so coding agents can verify API names, signatures, and documentation before writing code — no hallucinated methods, no version mismatches.

It is designed as the companion to the [Polars agent skill](https://github.com/polars-inc/skills/tree/main/polars): the skill teaches the agent *how* to write idiomatic, performant Polars code (lazy evaluation, expression contexts, pandas translation), while this server lets it verify the exact API surface of *your* Polars version — fast, offline, and without the guide content that now lives in the skill's reference files. The skill works without this server, falling back to docs.pola.rs for API verification.

## How it works

At startup the server builds an index of the installed Polars package via introspection — every `DataFrame`, `LazyFrame`, `Series`, and `Expr` method, all expression namespaces (`str`, `dt`, `list`, `struct`, ...), and all top-level functions. Because the index is built from the package you have installed, the documentation always matches the Polars version your code actually runs against.

## Tools

The server exposes three read-only tools:

| Tool | Purpose |
| --- | --- |
| `polars_search_api` | Find methods and functions by keyword, with optional class and namespace filters |
| `polars_browse` | List all methods in a class or namespace when you don't know the exact name |
| `polars_get_docstring` | Get the full signature and documentation for a specific API element |

### polars_search_api

Search the API by keyword. Results are grouped by class and namespace.

```
polars_search_api("filter")
polars_search_api("contains", class_filter="Expr", namespace_filter="str")
polars_search_api("date", namespace_filter="dt")
```

### polars_browse

Explore what's available in a class or namespace.

```
polars_browse("namespaces")   # list all expression namespaces
polars_browse("Expr.str")     # all string expression methods
polars_browse("functions")    # top-level functions: col(), lit(), when(), ...
```

### polars_get_docstring

Get complete documentation, including the exact signature with parameter names, types, and defaults.

```
polars_get_docstring("LazyFrame.collect")
polars_get_docstring("Expr.str.to_uppercase")
```

## Installation

**Important:** The server documents whatever Polars version is installed alongside it. Pin it to the Polars version your project uses.

### From PyPI

```bash
# with the latest Polars
uv pip install polars-mcp polars

# with your project's Polars version
uv pip install polars-mcp polars==1.35.0
```

### From source

```bash
git clone https://github.com/r-brink/polars-mcp
cd polars-mcp
uv pip install .
```

## Usage

Add to your MCP settings configuration file.

### With `uvx`

```json
{
    "mcpServers": {
        "polars-mcp": {
            "command": "uvx",
            "args": [
                "--with",
                "polars",
                "polars-mcp"
            ]
        }
    }
}
```

To pin the documented version, use e.g. `"polars==1.35.0"` in place of `"polars"`.

### When installing from source

```json
{
    "mcpServers": {
        "polars-mcp": {
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

For Claude Code:

```bash
claude mcp add polars-mcp -- uvx --with polars polars-mcp
```

Refer to the Model Context Protocol documentation on [connecting to local MCP servers](https://modelcontextprotocol.io/docs/develop/connect-local-servers) for other clients.

### Testing the server

```bash
# Run with the MCP inspector
npx @modelcontextprotocol/inspector uv run polars_mcp.py
```

## Contributing

Contributions welcome! Please submit issues or pull requests.
