# AI Agent Tools Integration

This module provides a comprehensive integration layer for autonomous AI agents (such as `AIyaru`) to natively use Sandalwood for Differential Algebra (DA) and Truncated Power Series Algebra (TPSA) operations.

## Architecture

The `agent_tools` module is designed with a decoupled architecture that conforms to standard agentic best practices:

1. **Stateful Session Registry**: Differential algebra objects (like `MultivariateTaylorFunction` and `TaylorMap`) can contain thousands of coefficients. To prevent LLM context window bloat, all tools return lightweight JSON references (e.g., `{"ref": "mtf_0"}`). Agents pass these reference strings in subsequent tool calls.
2. **Core Logic**: Contains the raw Python functions, type hints, robust error handling, and docstrings.
3. **LangChain Tools**: Exposes the core functions wrapped with LangChain's `@tool` decorator in `langchain_tools.py`.
4. **MCP Server**: Exposes the core functions as Model Context Protocol (MCP) tools and provides the `@mcp.prompt("sandalwood-da-expert")` instruction prompt in `mcp_server.py`.

## Usage

### LangChain / LangGraph

```python
from sandalwood.agent_tools.langchain_tools import evaluate_taylor_map, invert_taylor_map

# Add tools to your LangGraph nodes
tools = [evaluate_taylor_map, invert_taylor_map]
# llm.bind_tools(tools)
```

### Model Context Protocol (MCP)

To run the server and expose the tools and prompts to compatible MCP clients (like Claude Desktop or an AI IDE):

```bash
uv run python -m sandalwood.agent_tools.mcp_server
```

You can instruct your MCP client to load the expert instructions by invoking the prompt:

```json
{
  "method": "prompts/get",
  "params": {
    "name": "sandalwood-da-expert"
  }
}
```

Or configure it directly in your Claude Desktop configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "sandalwood": {
      "command": "uv",
      "args": [
        "run",
        "python",
        "-m",
        "sandalwood.agent_tools.mcp_server"
      ]
    }
  }
}
```

## API Reference

The following sections provide the auto-generated documentation for the core agent tools and the stateful registry.

### Core Agent Functions

```{eval-rst}
.. automodule:: sandalwood.agent_tools.functions
   :members:
   :undoc-members:
   :show-inheritance:
```

### Stateful Registry

```{eval-rst}
.. automodule:: sandalwood.agent_tools.registry
   :members:
   :undoc-members:
   :show-inheritance:
```
