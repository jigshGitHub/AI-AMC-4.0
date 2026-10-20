"""
Windows Integrated Proxy Utility for HTTPX / LangChain.
Leverages native Windows WinHTTP to mirror '-ProxyUseDefaultCredentials'.
"""

import os
import time
import httpx
import win32com.client

class WinHttpStream(httpx.SyncByteStream):
    """Handles memory-safe byte streaming for HTTPX responses."""
    def __init__(self, response_body_variant):
        # Convert native Windows COM memory variant safely to standard Python bytes
        self._bytes = bytes(memoryview(response_body_variant))

    def __iter__(self):
        # Yield the bytes in chunks to enable downstream token streaming
        chunk_size = 1024
        for i in range(0, len(self._bytes), chunk_size):
            yield self._bytes[i:i + chunk_size]


class WinHttpTransport(httpx.BaseTransport):
    """
    Custom HTTPX Transport that routes requests through the native Windows
    WinHTTP engine to seamlessly authenticate past  corporate proxies.
    """
    def __init__(self, proxy_url, default_timeout=60.0):
        self.proxy_url = proxy_url
        self.default_timeout = default_timeout
        # Ensure LangChain doesn't attempt to inject internal TCP keep-alive socket configurations
        os.environ["LANGCHAIN_OPENAI_TCP_KEEPALIVE"] = "0"

    def handle_request(self, request):
        # Instantiate native Windows HTTP automation layer
        win_http = win32com.client.Dispatch("WinHttp.WinHttpRequest.5.1")

        # Determine strict timeout constraints from HTTPX context or fallback to default
        timeout_val = request.extensions.get("timeout", {}).get("read") or self.default_timeout
        timeout_ms = int(timeout_val * 1000) if timeout_val else 60000

        # Set Windows Internal Timeouts (Resolve, Connect, Send, Receive in milliseconds)
        win_http.SetTimeouts(5000, 10000, timeout_ms, timeout_ms)

        # Open connection
        win_http.Open(request.method, str(request.url), False)

        # Force routing through designated agency proxy server
        win_http.SetProxy(2, self.proxy_url)

        # 0 = Use automatic logged-in Windows session token (SSPI/Active Directory)
        win_http.SetAutoLogonPolicy(0)

        # Inject transmission headers cleanly
        for key, value in request.headers.items():
            if key.lower() not in ['host', 'content-length']:
                win_http.SetRequestHeader(key, value)

        # Send payload body if present
        body = request.read()
        win_http.Send(body if body else None)

        # Parse Windows CRLF response header block into standard dictionary structure
        raw_headers = win_http.GetAllResponseHeaders().strip().split("\r\n")
        parsed_headers = {}
        for line in raw_headers:
            if ":" in line:
                key, val = line.split(":", 1)
                parsed_headers[key.strip()] = val.strip()

        # Package and return optimized streaming response
        return httpx.Response(
            status_code=win_http.Status,
            headers=httpx.Headers(parsed_headers),
            stream=WinHttpStream(win_http.ResponseBody),
            request=request
        )


def get_authenticated_clients(proxy_url, timeout=60.0):
    """
    Utility helper that returns pre-configured sync and async HTTPX clients
    ready to be bound to LangChain / LangGraph.
    """
    transport_engine = WinHttpTransport(proxy_url=proxy_url, default_timeout=timeout)

    sync_client = httpx.Client(
        transport=transport_engine,
        verify=False
    )

    async_client = httpx.AsyncClient(
        transport=transport_engine,
        verify=False
    )

    return sync_client, async_client
