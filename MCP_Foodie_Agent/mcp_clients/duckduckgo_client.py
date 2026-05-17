"""DuckDuckGo search client.

Primary path: real MCP protocol via uvx duckduckgo-mcp-server (stdio).
Fallback: duckduckgo-search Python package — used automatically when the MCP
server is unavailable (uvx not installed, server startup failed, etc.).

Both paths produce the same result shape so the rest of the graph is unaffected.
"""
import asyncio
import json as _json
import concurrent.futures
from contextlib import asynccontextmanager

from config import DUCKDUCKGO_MCP_COMMAND, DUCKDUCKGO_MCP_ARGS


# ---------------------------------------------------------------------------
# MCP path
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _mcp_session():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command=DUCKDUCKGO_MCP_COMMAND,
        args=DUCKDUCKGO_MCP_ARGS,
        env=None,
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def _search_via_mcp(query: str, max_results: int) -> list[dict]:
    async with _mcp_session() as session:
        tool_name = "search"
        try:
            result = await session.call_tool(
                tool_name, arguments={"query": query, "max_results": max_results}
            )
        except Exception:
            tools_result = await session.list_tools()
            available = [t.name for t in tools_result.tools]
            fallback_name = next(
                (n for n in available if "search" in n.lower()),
                available[0] if available else None,
            )
            if not fallback_name:
                raise RuntimeError("No search tool found on MCP server")
            result = await session.call_tool(
                fallback_name, arguments={"query": query, "max_results": max_results}
            )

        out = []
        for item in result.content:
            text = getattr(item, "text", str(item))
            try:
                parsed = _json.loads(text)
                out.extend(parsed) if isinstance(parsed, list) else out.append(parsed)
            except Exception:
                out.append({"snippet": text})
        return out


def _run_in_fresh_thread(coro) -> list[dict]:
    """Run an async coroutine in a dedicated thread with its own event loop.

    Required because Streamlit runs in a thread that may already have an event
    loop, and asyncio.run() raises RuntimeError if called inside a running loop.
    """
    def _target():
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_target).result(timeout=30)


# ---------------------------------------------------------------------------
# Python-package fallback
# ---------------------------------------------------------------------------

def _search_via_sdk(query: str, max_results: int) -> list[dict]:
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return [
        {"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")}
        for r in results
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search(query: str, max_results: int = 5) -> list[dict]:
    """Search DuckDuckGo. Uses MCP server when available, SDK package as fallback."""
    try:
        results = _run_in_fresh_thread(_search_via_mcp(query, max_results))
        if results and not all("error" in r for r in results):
            print("[MCP] Search succeeded via MCP server")
            return results
        raise RuntimeError("MCP returned only errors")
    except Exception as mcp_err:
        print(f"[MCP] Falling back to Python SDK ({mcp_err})")
        try:
            return _search_via_sdk(query, max_results)
        except Exception as sdk_err:
            return [{"error": f"All search methods failed: {sdk_err}"}]


def list_tools() -> list[dict]:
    """List tools exposed by the MCP server (diagnostic helper)."""
    async def _list():
        async with _mcp_session() as session:
            result = await session.list_tools()
            return [{"name": t.name, "description": t.description} for t in result.tools]

    try:
        return _run_in_fresh_thread(_list())
    except Exception as e:
        return [{"error": str(e)}]
