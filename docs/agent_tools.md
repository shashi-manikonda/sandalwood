# AI Agent Tools Integration

This module provides a comprehensive integration layer for autonomous AI agents (such as `AIyaru`) to natively use Sandalwood for Differential Algebra (DA) and Truncated Power Series Algebra (TPSA) operations.

## Architecture

The `agent_tools` module is designed with a decoupled architecture that conforms to standard agentic best practices:

1. **Stateful Session Registry**: Differential algebra objects (like `MultivariateTaylorFunction` and `TaylorMap`) can contain thousands of coefficients. To prevent LLM context window bloat, all tools return lightweight JSON references (e.g., `{"ref": "mtf_0"}`). Agents pass these reference strings in subsequent tool calls.
2. **Core Logic**: Contains the raw Python functions, type hints, robust error handling, and docstrings.
3. **LangChain Tools**: Exposes the core functions wrapped with LangChain's `@tool` decorator in `langchain_tools.py`.
4. **MCP Server**: Exposes the core functions as Model Context Protocol (MCP) tools and provides the `@mcp.prompt("sandalwood-da-expert")` instruction prompt in `mcp_server.py`.

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
