import asyncio
import os
import sys
from pathlib import Path

from mcp import Client
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextContent, TextResourceContents

import polars_mcp

ROOT = Path(__file__).resolve().parents[1]


def test_mcp_client_can_use_tools_and_guide_resources():
    async def check():
        async with Client(polars_mcp.mcp) as client:
            tools = await client.list_tools()
            assert {tool.name for tool in tools.tools} == {
                "polars_get_guide",
                "polars_search_api",
                "polars_browse",
                "polars_get_docstring",
            }

            resources = await client.list_resources()
            assert {resource.uri for resource in resources.resources} == {
                f"polars://guides/{guide}" for guide in polars_mcp.GUIDE_FILES
            }

            for guide, filename in polars_mcp.GUIDE_FILES.items():
                result = await client.read_resource(f"polars://guides/{guide}")
                assert isinstance(result.contents[0], TextResourceContents)
                assert result.contents[0].mime_type == "text/markdown"
                assert result.contents[0].text == polars_mcp.load_guide(filename)

            tool_calls = [
                ("polars_get_guide", {"guide": "contexts"}),
                ("polars_search_api", {"query": "filter", "limit": 2}),
                ("polars_browse", {"category": "DataFrame", "limit": 2}),
                ("polars_get_docstring", {"name": "DataFrame.filter"}),
            ]
            for name, arguments in tool_calls:
                result = await client.call_tool(name, arguments)
                assert not result.is_error
                assert result.content

    asyncio.run(check())


def test_tool_limits_reject_negative_values():
    async def check():
        async with Client(polars_mcp.mcp) as client:
            search_result = await client.call_tool(
                "polars_search_api", {"query": "filter", "limit": -1}
            )
            browse_result = await client.call_tool(
                "polars_browse", {"category": "DataFrame", "limit": -1}
            )

        assert isinstance(search_result.content[0], TextContent)
        assert search_result.content[0].text == "limit must be non-negative"
        assert isinstance(browse_result.content[0], TextContent)
        assert browse_result.content[0].text == "limit must be non-negative"

    asyncio.run(check())


def test_response_limit_adds_truncation_notice():
    response = polars_mcp._limit_response("x" * (polars_mcp.CHARACTER_LIMIT + 1))

    assert len(response) == polars_mcp.CHARACTER_LIMIT
    assert response.endswith("[Response truncated to fit the character limit.]")


def test_installed_command_serves_tools_and_resources():
    async def check():
        env = os.environ.copy()
        params = StdioServerParameters(
            command=str(Path(sys.executable).parent / "polars-mcp"),
            args=[],
            env=env,
            cwd=ROOT,
        )
        async with stdio_client(params) as streams, ClientSession(*streams) as client:
            await client.initialize()
            tools = await client.list_tools()
            assert any(tool.name == "polars_search_api" for tool in tools.tools)
            result = await client.call_tool(
                "polars_browse", {"category": "namespaces", "limit": 1}
            )
            assert not result.is_error
            guide = await client.read_resource("polars://guides/contexts")
            assert guide.contents[0].mime_type == "text/markdown"

    asyncio.run(check())
