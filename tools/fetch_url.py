"""Fetch content from a URL tool."""

import httpx


def fetch_url(url: str) -> str:
    """Fetch content from a URL."""
    try:
        resp = httpx.get(url, timeout=10, follow_redirects=True)
        return resp.text[:2000]
    except Exception as e:
        return f"Error: {e}"
