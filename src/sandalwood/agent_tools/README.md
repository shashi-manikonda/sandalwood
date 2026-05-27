# Sandalwood Agent Tools

This module provides a comprehensive integration layer for autonomous AI agents (such as `AIyaru`) to natively use Sandalwood for Differential Algebra (DA) and Truncated Power Series Algebra (TPSA) operations.

## Architecture

The `agent_tools` module is designed with a decoupled architecture that conforms to standard agentic best practices:

1. **Stateful Session Registry (`registry.py`)**: Differential algebra objects (like `MultivariateTaylorFunction` and `TaylorMap`) can contain thousands of coefficients. To prevent LLM context window bloat, all tools return lightweight JSON references (e.g., `{"ref": "mtf_0", ...}`) inside a standardized response envelope. Agents pass these reference strings in subsequent tool calls instead of large serialized coefficient datasets.
2. **Core Logic (`functions.py`)**: Contains the raw Python functions, type hints, robust error handling, and docstrings. Errors are caught internally and returned as descriptive strings to allow the LLM to observe and auto-correct.
3. **LangChain Tools (`langchain_tools.py`)**: Exposes the core functions wrapped with LangChain's `@tool` decorator.
4. **MCP Server (`mcp_server.py`)**: Exposes the core functions as Model Context Protocol (MCP) tools and exposes the registry state via read-only resources (`registry://variables`). Also provides the `@mcp.prompt("sandalwood-da-expert")` instruction prompt.

## Core Capabilities Exposed

The agent tools expose 21 functions representing the full breadth of the Sandalwood library capabilities.

### 1. Initialization and Setup
* **`initialize_sandalwood`**: Configures the global `max_order`, `max_dimension`, and `implementation` (e.g., `cosy` or `python`) for the Sandalwood engine.

### 2. Construction and Parsing
LLMs often struggle to generate the precise nested coordinate arrays required to instantiate Taylor Functions directly.
* **`parse_expression_to_mtf`**: Parses a mathematical string expression into a `MultivariateTaylorFunction`. Supports common functions (`sin`, `cos`, `exp`, etc.) and specialty functions (`gaussian`, `isqrt`, `inv_cbrt`, `inv_pow_3_2`, `asin`, `acos`, `atan`).
* **`create_taylor_map`**: Accepts a list of mathematical expressions to construct a vector-valued `TaylorMap`.

### 3. Evaluation
* **`evaluate_mtf` / `evaluate_taylor_map`**: Evaluates functions and maps at a single phase space coordinate.
* **`evaluate_mtf_batch` / `evaluate_taylor_map_batch`**: High-performance vectorized evaluation across a batch of coordinates using Sandalwood's `neval` backend.

### 4. Calculus & Differential Algebra
* **`compute_partial_derivative`**: Computes the partial derivative of an MTF with respect to a given variable.
* **`integrate_mtf`**: Performs indefinite integration, or definite integration if numerical limits are provided.
* **`compute_poisson_bracket`**: Computes the Poisson bracket $[F, G]$ of two MTFs, a fundamental operation for Hamiltonian tracking.

### 5. Arithmetic & Composition
* **`perform_mtf_arithmetic`**: Performs basic arithmetic (`+`, `-`, `*`, `/`, `**`) between MTFs or an MTF and a scalar.
* **`perform_complex_operation`**: Extracts the real/imaginary parts or computes the conjugate of a Complex MTF (CMTF).
* **`compose_taylor_maps`**: Composes two maps, computing $M_1 \circ M_2$.
* **`compose_mtfs`**: Substitutes a set of MTFs into the variables of another MTF.

### 6. Variable Substitution and Truncation
* **`substitute_variable_in_mtf` / `substitute_in_taylor_map`**: Partially or fully evaluates expressions by fixing specific variable coordinates.
* **`truncate_object`**: Lowers the maximum truncation order of an existing MTF or TaylorMap in place.
* **`extract_map_component`**: Extracts an individual `MultivariateTaylorFunction` from a vector-valued `TaylorMap` and saves it to the registry.
* **`get_mtf_coefficient`**: Retrieves the scalar coefficient for a specific exponent array.

### 7. Diagnostics and Sensitivity
* **`analyze_taylor_map`**: Analyzes a map to check if it is invertible and computes the trace of its linear part (Jacobian matrix).
* **`invert_taylor_map`**: Computes the algebraic inverse of an invertible Taylor Map.
* **`compute_map_sensitivity`**: Scales map coefficients for sensitivity analysis given a list of scaling factors.
* **`analyze_mtf_diagnostics`**: Retrieves function norms, weighted norms (COSY backend), and estimates stability/decay properties of coefficients.
* **`mtf_info` / `taylor_map_info`**: Retrieves human-readable, tabular string representations of the objects to inspect the terms.

## Usage

### LangChain / LangGraph
```python
from sandalwood.agent_tools import evaluate_taylor_map, invert_taylor_map

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
