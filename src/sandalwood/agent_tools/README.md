# Sandalwood Agent Tools

This module provides an integration layer for autonomous AI agents (such as `AIyaru`) to natively use Sandalwood for Differential Algebra and Taylor Map operations.

## LangChain / LangGraph Tools
The `langchain_tools.py` module exposes tools decorated with LangChain's `@tool` decorator. 

```python
from sandalwood.agent_tools import evaluate_taylor_map, invert_taylor_map

# Add tools to your LangGraph nodes
tools = [evaluate_taylor_map, invert_taylor_map]
# llm.bind_tools(tools)
```

## MCP Server
The `mcp_server.py` implements a Model Context Protocol (MCP) server using `mcp.server.fastmcp.FastMCP`.

To run the server:
```bash
uv run python -m sandalwood.agent_tools.mcp_server
```

## SymPy Parser
LLMs often struggle to generate raw exponent arrays for Taylor Functions. The `expression_to_mtf` helper function allows agents to supply a math expression string and automatically convert it to a `MultivariateTaylorFunction`.

```python
from sandalwood.agent_tools import expression_to_mtf

mtf_obj = expression_to_mtf("x1**2 + sin(x2)", dimension=2, max_order=4)
print(mtf_obj.get_tabular_dataframe())
```
