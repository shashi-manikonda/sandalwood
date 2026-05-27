from mcp.server.fastmcp import FastMCP
from sandalwood.agent_tools import functions
from sandalwood.agent_tools.registry import _registry

# Create the MCP Server
mcp = FastMCP("sandalwood-agent-tools")

# Register all functions as MCP tools
mcp.tool()(functions.initialize_sandalwood)
mcp.tool()(functions.parse_expression_to_mtf)
mcp.tool()(functions.create_taylor_map)
mcp.tool()(functions.evaluate_taylor_map)
mcp.tool()(functions.evaluate_mtf)
mcp.tool()(functions.invert_taylor_map)
mcp.tool()(functions.compute_partial_derivative)
mcp.tool()(functions.integrate_mtf)
mcp.tool()(functions.compose_taylor_maps)
mcp.tool()(functions.compose_mtfs)
mcp.tool()(functions.perform_mtf_arithmetic)
mcp.tool()(functions.perform_complex_operation)
mcp.tool()(functions.mtf_info)
mcp.tool()(functions.taylor_map_info)
mcp.tool()(functions.substitute_variable_in_mtf)
mcp.tool()(functions.substitute_in_taylor_map)
mcp.tool()(functions.get_mtf_coefficient)
mcp.tool()(functions.truncate_object)
mcp.tool()(functions.analyze_taylor_map)
mcp.tool()(functions.evaluate_mtf_batch)
mcp.tool()(functions.evaluate_taylor_map_batch)
mcp.tool()(functions.analyze_mtf_diagnostics)
mcp.tool()(functions.compute_poisson_bracket)
mcp.tool()(functions.compute_map_sensitivity)
mcp.tool()(functions.extract_map_component)

@mcp.prompt("sandalwood-da-expert")
def sandalwood_da_expert() -> str:
    """Provides expert system instructions on utilizing Sandalwood for differential algebra tasks."""
    return (
        "You are a Differential Algebra (DA) expert assisting the user with Sandalwood.\n"
        "Sandalwood is a high-performance library for Differential Algebra and Truncated Power Series Algebra (TPSA).\n"
        "It has two core classes:\n"
        "1. MultivariateTaylorFunction (MTF): Represents a multivariate Taylor series truncated to a maximum order.\n"
        "2. TaylorMap: Represents a vector-valued mapping (a list of MTFs with the same dimension and order).\n\n"
        "Guidelines for solving tasks:\n"
        "- Initialization: Always call `initialize_sandalwood` before performing any operations if settings (dimension, order, backend implementation) need to be adjusted. By default, Sandalwood operates on the selected backend (e.g. COSY backend supports advanced diagnostics).\n"
        "- Expression Parsing: Use `parse_expression_to_mtf` to convert mathematical expression strings (e.g. 'x1**2 + sin(x2)') into MTFs. You can use common math functions like sin, cos, tan, exp, log, sinh, cosh, tanh, sqrt, arcsin (asin), arccos (acos), arctan (atan), erf, and specialty functions: gaussian, isqrt, inv_cbrt, inv_pow_3_2.\n"
        "- Taylor Maps: Combine multiple expressions or components using `create_taylor_map`. Extract individual components back out using `extract_map_component`.\n"
        "- Evaluation:\n"
        "  - For single points, use `evaluate_mtf` or `evaluate_taylor_map`.\n"
        "  - For high-performance vectorized evaluation on multiple points, use `evaluate_mtf_batch` or `evaluate_taylor_map_batch`.\n"
        "- Arithmetic & Operations: Perform MTF algebra with `perform_mtf_arithmetic` or complex number functions with `perform_complex_operation`.\n"
        "- Calculus & DA: Use `compute_partial_derivative`, `integrate_mtf`, and `compute_poisson_bracket`.\n"
        "- Diagnostics & Sensitivity: Use `analyze_mtf_diagnostics` to get functions' norm, weighted norm, and stability estimates. Use `analyze_taylor_map` to check for invertibility and compute traces. Use `compute_map_sensitivity` to scale map coefficients for sensitivity analysis.\n"
        "- Composition: Compose maps or functions using `compose_taylor_maps` and `compose_mtfs`.\n"
        "- Substitutions: Substitute variables using `substitute_variable_in_mtf` or `substitute_in_taylor_map`.\n"
        "- Truncation: Restrict maximum order using `truncate_object`.\n\n"
        "When referencing objects in tool calls, use their registry reference name (e.g., 'mtf_0', 'map_1') or pass their full JSON representation. Always format mathematical formulas cleanly."
    )


# Expose session registry variables as read-only resources
@mcp.resource("registry://variables")
def list_variables() -> str:
    """List all registered MultivariateTaylorFunctions and TaylorMaps in the current session."""
    import json
    vars_info = {}
    for name, obj in _registry.items():
        if hasattr(obj, "components") and len(obj.components) > 0:
            dim = obj.components[0].dimension
        elif hasattr(obj, "dimension"):
            dim = obj.dimension
        else:
            dim = None
            
        vars_info[name] = {
            "type": type(obj).__name__,
            "dimension": dim
        }
    return json.dumps(vars_info, indent=2)

@mcp.resource("registry://variable/{name}")
def get_variable(name: str) -> str:
    """Get the detailed tabular representation of a registered variable by name."""
    from sandalwood import MultivariateTaylorFunction, TaylorMap
    if name not in _registry:
        return f"Variable '{name}' not found in session registry."
    obj = _registry[name]
    if isinstance(obj, MultivariateTaylorFunction):
        return obj.get_tabular_dataframe().to_string(index=False)
    elif isinstance(obj, TaylorMap):
        return str(obj)
    else:
        return str(obj)

def main():
    mcp.run()

if __name__ == "__main__":
    main()
