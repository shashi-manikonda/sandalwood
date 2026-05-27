from langchain_core.tools import tool

from sandalwood.agent_tools import functions, schemas

initialize_sandalwood = tool(
    "initialize_sandalwood",
    args_schema=schemas.InitializeSandalwoodInput,
    description="Explicitly initializes or resets the global Sandalwood settings (maximum truncation order and maximum dimension). This must be called before using Taylor operations if you want to override the default limits."
)(functions.initialize_sandalwood)

parse_expression_to_mtf = tool(
    "parse_expression_to_mtf",
    args_schema=schemas.ParseExpressionInput,
    description="Parses a natural math string (e.g., 'x1**2 + sin(x2)') into an MTF and registers it. Supports trig functions, exp, log, and specialty roots like isqrt."
)(functions.parse_expression_to_mtf)

create_taylor_map = tool(
    "create_taylor_map",
    args_schema=schemas.CreateTaylorMapInput,
    description="Creates a vector-valued TaylorMap from a list of mathematical expressions, representing a transformation."
)(functions.create_taylor_map)

evaluate_taylor_map = tool(
    "evaluate_taylor_map",
    args_schema=schemas.EvaluateTaylorMapInput,
    description="Evaluates a TaylorMap at a specific single point in phase space. For batch evaluations, use evaluate_taylor_map_batch."
)(functions.evaluate_taylor_map)

evaluate_mtf = tool(
    "evaluate_mtf",
    args_schema=schemas.EvaluateMtfInput,
    description="Evaluates a single MultivariateTaylorFunction at a specific single point. For batch evaluations, use evaluate_mtf_batch."
)(functions.evaluate_mtf)

invert_taylor_map = tool(
    "invert_taylor_map",
    args_schema=schemas.InvertTaylorMapInput,
    description="Inverts a square TaylorMap using fixed-point iteration, provided it has no constant terms and an invertible linear Jacobian."
)(functions.invert_taylor_map)

compute_partial_derivative = tool(
    "compute_partial_derivative",
    args_schema=schemas.ComputePartialDerivativeInput,
    description="Computes the partial derivative of an MTF with respect to a given variable index (1-indexed)."
)(functions.compute_partial_derivative)

integrate_mtf = tool(
    "integrate_mtf",
    args_schema=schemas.IntegrateMtfInput,
    description="Integrates an MTF with respect to a specific variable (1-based index). Optionally evaluates definite limits."
)(functions.integrate_mtf)

compose_taylor_maps = tool(
    "compose_taylor_maps",
    args_schema=schemas.ComposeTaylorMapsInput,
    description="Composes two TaylorMaps, calculating map1(map2(x)). The output dimension of map2 must match the input dimension of map1."
)(functions.compose_taylor_maps)

compose_mtfs = tool(
    "compose_mtfs",
    args_schema=schemas.ComposeMtfsInput,
    description="Composes a MultivariateTaylorFunction with other MultivariateTaylorFunctions."
)(functions.compose_mtfs)

perform_mtf_arithmetic = tool(
    "perform_mtf_arithmetic",
    args_schema=schemas.PerformMtfArithmeticInput,
    description="Performs basic arithmetic (+, -, *, /, **) between two MTFs, or between an MTF and a scalar."
)(functions.perform_mtf_arithmetic)

perform_complex_operation = tool(
    "perform_complex_operation",
    args_schema=schemas.PerformComplexOperationInput,
    description="Performs complex operations (conjugate, real_part, imag_part) on a ComplexMultivariateTaylorFunction."
)(functions.perform_complex_operation)

mtf_info = tool(
    "mtf_info",
    args_schema=schemas.MtfInfoInput,
    description="Returns a tabular representation of a MultivariateTaylorFunction, showing terms, orders, and coefficients."
)(functions.mtf_info)

taylor_map_info = tool(
    "taylor_map_info",
    args_schema=schemas.TaylorMapInfoInput,
    description="Returns a detailed representation of a TaylorMap's components."
)(functions.taylor_map_info)

substitute_variable_in_mtf = tool(
    "substitute_variable_in_mtf",
    args_schema=schemas.SubstituteVariableInMtfInput,
    description="Substitutes a single variable (1-based index) in an MTF with a numeric value."
)(functions.substitute_variable_in_mtf)

substitute_in_taylor_map = tool(
    "substitute_in_taylor_map",
    args_schema=schemas.SubstituteInTaylorMapInput,
    description="Substitutes a single variable in all components of a TaylorMap with a numeric value."
)(functions.substitute_in_taylor_map)

get_mtf_coefficient = tool(
    "get_mtf_coefficient",
    args_schema=schemas.GetMtfCoefficientInput,
    description="Extracts a specific coefficient from an MTF given an exponents list (e.g., [1, 0, 2])."
)(functions.get_mtf_coefficient)

truncate_object = tool(
    "truncate_object",
    args_schema=schemas.TruncateObjectInput,
    description="Truncates an MTF or TaylorMap to a lower maximum order."
)(functions.truncate_object)

analyze_taylor_map = tool(
    "analyze_taylor_map",
    args_schema=schemas.AnalyzeTaylorMapInput,
    description="Analyzes a TaylorMap for invertibility, computes traces, and eigenvalues of the linear part."
)(functions.analyze_taylor_map)

evaluate_mtf_batch = tool(
    "evaluate_mtf_batch",
    args_schema=schemas.EvaluateMtfBatchInput,
    description="Vectorized evaluation of an MTF at multiple coordinate points simultaneously."
)(functions.evaluate_mtf_batch)

evaluate_taylor_map_batch = tool(
    "evaluate_taylor_map_batch",
    args_schema=schemas.EvaluateTaylorMapBatchInput,
    description="Vectorized evaluation of a TaylorMap at multiple coordinate points simultaneously."
)(functions.evaluate_taylor_map_batch)

analyze_mtf_diagnostics = tool(
    "analyze_mtf_diagnostics",
    args_schema=schemas.AnalyzeMtfDiagnosticsInput,
    description="Computes norm, weighted norm, and stability estimates for an MTF."
)(functions.analyze_mtf_diagnostics)

compute_poisson_bracket = tool(
    "compute_poisson_bracket",
    args_schema=schemas.ComputePoissonBracketInput,
    description="Computes the Poisson Bracket of two Hamiltonian MTFs."
)(functions.compute_poisson_bracket)

compute_map_sensitivity = tool(
    "compute_map_sensitivity",
    args_schema=schemas.ComputeMapSensitivityInput,
    description="Scales map coefficients relative to an evaluation point for sensitivity analysis."
)(functions.compute_map_sensitivity)

extract_map_component = tool(
    "extract_map_component",
    args_schema=schemas.ExtractMapComponentInput,
    description="Extracts a specific component MTF (0-indexed) from a TaylorMap."
)(functions.extract_map_component)
