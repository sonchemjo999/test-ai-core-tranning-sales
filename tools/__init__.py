"""
Tool registry for the Anthropic agent.
Each tool lives in its own module; this __init__ wires them into the registry.
"""

from tools.search_web import search_web
from tools.calculate import calculate
from tools.fetch_url import fetch_url

# Tool registry - the agent uses this dict
TOOLS = {
    "search_web": {
        "fn": search_web,
        "description": "Search for information on the web",
        "parameters": {"query": "string"},
    },
    "calculate": {
        "fn": calculate,
        "description": "Evaluate a math expression",
        "parameters": {"expression": "string"},
    },
    "fetch_url": {
        "fn": fetch_url,
        "description": "Fetch content from a URL",
        "parameters": {"url": "string"},
    },
}


def get_tool_schemas() -> list[dict]:
    """Return tool schemas in Anthropic API format."""
    schemas = []
    for name, tool in TOOLS.items():
        schemas.append({
            "name": name,
            "description": tool["description"],
            "input_schema": {
                "type": "object",
                "properties": {
                    k: {"type": v, "description": k}
                    for k, v in tool["parameters"].items()
                },
                "required": list(tool["parameters"].keys()),
            },
        })
    return schemas


def execute_tool(name: str, args: dict) -> str:
    """Execute a tool by name."""
    tool = TOOLS.get(name)
    if not tool:
        return f"Tool '{name}' does not exist"
    return tool["fn"](**args)
