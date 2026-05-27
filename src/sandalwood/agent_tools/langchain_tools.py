from langchain_core.tools import tool

from sandalwood.agent_tools import functions, schemas

initialize_sandalwood = tool(
    "initialize_sandalwood",
    args_schema=schemas.InitializeSandalwoodInput,
    description="Explicitly initializes or resets the global Sandalwood Differential Algebra environment. Accepts 'max_order' (int), 'max_dimension' (int), and an optional 'implementation' string ('cosy' or 'python'). This tool MUST be called before any other operations to define the phase space limits. Returns an envelope payload containing the status.",
)(functions.initialize_sandalwood)

parse_expression_to_mtf = tool(
    "parse_expression_to_mtf",
    args_schema=schemas.ParseExpressionInput,
    description="Parses a human-readable mathematical formula string (e.g., 'x1**2 + sin(x2)') and compiles it into a MultivariateTaylorFunction. Supports basic arithmetic, trigonometric functions (sin, cos, tan), exponential (exp, log), and roots. Automatically stores the compiled MTF in the registry. Returns the MTF registry reference and coefficient JSON data.",
)(functions.parse_expression_to_mtf)

create_taylor_map = tool(
    "create_taylor_map",
    args_schema=schemas.CreateTaylorMapInput,
    description="Constructs a vector-valued TaylorMap (a collection of MTFs representing a coordinate transformation) from a list of natural math expression strings. Compiles and registers the resulting TaylorMap. Returns the TaylorMap registry reference and component JSON data.",
)(functions.create_taylor_map)

evaluate_taylor_map = tool(
    "evaluate_taylor_map",
    args_schema=schemas.EvaluateTaylorMapInput,
    description="Evaluates an existing TaylorMap at a single numerical point in phase space. The input point must be a list of floats matching the domain dimension of the map. Returns a list of floating-point evaluation results. For batch evaluations over multiple points, use evaluate_taylor_map_batch instead.",
)(functions.evaluate_taylor_map)

evaluate_mtf = tool(
    "evaluate_mtf",
    args_schema=schemas.EvaluateMtfInput,
    description="Evaluates a single MultivariateTaylorFunction at a specific single point in phase space. The input point must be a list of floats matching the dimension of the MTF. Returns a single floating-point scalar result. For evaluating multiple points simultaneously, use evaluate_mtf_batch.",
)(functions.evaluate_mtf)

invert_taylor_map = tool(
    "invert_taylor_map",
    args_schema=schemas.InvertTaylorMapInput,
    description="Inverts a square TaylorMap (where domain dimension equals codomain dimension) using iterative fixed-point methods. The map must not have any constant order terms and must possess an invertible linear Jacobian. Registers the inverted map and returns its reference.",
)(functions.invert_taylor_map)

compute_partial_derivative = tool(
    "compute_partial_derivative",
    args_schema=schemas.ComputePartialDerivativeInput,
    description="Computes the exact analytical partial derivative of an MTF with respect to a specified variable. The variable index is 1-based (e.g., index 1 corresponds to x1). Registers the resulting derivative MTF and returns its reference.",
)(functions.compute_partial_derivative)

integrate_mtf = tool(
    "integrate_mtf",
    args_schema=schemas.IntegrateMtfInput,
    description="Computes the exact analytical integral of an MTF with respect to a specific variable (1-based index). Supports both indefinite integration (yielding a new MTF with integration constant 0) and definite integration (by providing lower_limit and upper_limit floats, yielding a constant MTF). Registers and returns the result.",
)(functions.integrate_mtf)

compose_taylor_maps = tool(
    "compose_taylor_maps",
    args_schema=schemas.ComposeTaylorMapsInput,
    description="Composes two TaylorMaps mathematically, computing the combined transformation map1(map2(x)). The output dimension of map2 must exactly match the input domain dimension of map1. Registers the newly composed TaylorMap and returns its reference.",
)(functions.compose_taylor_maps)

compose_mtfs = tool(
    "compose_mtfs",
    args_schema=schemas.ComposeMtfsInput,
    description="Composes a scalar MultivariateTaylorFunction with a list of inner MultivariateTaylorFunctions. The number of inner MTFs must match the dimension of the outer MTF. Evaluates outer(inner_1(x), inner_2(x), ...). Registers the resulting MTF and returns its reference.",
)(functions.compose_mtfs)

perform_mtf_arithmetic = tool(
    "perform_mtf_arithmetic",
    args_schema=schemas.PerformMtfArithmeticInput,
    description="Executes fundamental algebraic operations ('add', 'subtract', 'multiply', 'divide', 'power') between two MTFs, or between an MTF and a numerical scalar. The operands are resolved from the registry or parsed as scalars. Registers the arithmetic result and returns its reference.",
)(functions.perform_mtf_arithmetic)

perform_complex_operation = tool(
    "perform_complex_operation",
    args_schema=schemas.PerformComplexOperationInput,
    description="Executes complex-specific algebraic operations ('conjugate', 'real_part', 'imag_part') on a ComplexMultivariateTaylorFunction. Returns the real or imaginary components as standard MTFs. Registers the result and returns its reference.",
)(functions.perform_complex_operation)

mtf_info = tool(
    "mtf_info",
    args_schema=schemas.MtfInfoInput,
    description="Retrieves a human-readable tabular representation of an MTF, displaying its active expansion terms, their corresponding polynomial orders, and specific numerical coefficients. Useful for inspecting the internal algebraic structure of an MTF.",
)(functions.mtf_info)

taylor_map_info = tool(
    "taylor_map_info",
    args_schema=schemas.TaylorMapInfoInput,
    description="Retrieves a comprehensive text representation of a TaylorMap, detailing the individual MTF components that make up the vector field. Useful for inspecting the coordinate transformations element-by-element.",
)(functions.taylor_map_info)

substitute_variable_in_mtf = tool(
    "substitute_variable_in_mtf",
    args_schema=schemas.SubstituteVariableInMtfInput,
    description="Performs a partial evaluation by substituting a single phase space variable (identified by a 1-based index) with a specific numerical constant inside an MTF. Reduces the effective dimensionality of the MTF. Registers the resulting MTF and returns its reference.",
)(functions.substitute_variable_in_mtf)

substitute_in_taylor_map = tool(
    "substitute_in_taylor_map",
    args_schema=schemas.SubstituteInTaylorMapInput,
    description="Performs a partial evaluation across all components of a TaylorMap by substituting a single variable (1-based index) with a numerical constant. Registers the modified TaylorMap and returns its reference.",
)(functions.substitute_in_taylor_map)

get_mtf_coefficient = tool(
    "get_mtf_coefficient",
    args_schema=schemas.GetMtfCoefficientInput,
    description="Extracts the precise numerical coefficient of a specific polynomial term from an MTF. The term is identified by an integer list of exponents (e.g., [1, 0, 2] for x1 * x3**2). Returns a dictionary containing the coefficient float.",
)(functions.get_mtf_coefficient)

truncate_object = tool(
    "truncate_object",
    args_schema=schemas.TruncateObjectInput,
    description="Truncates an existing MTF or TaylorMap by aggressively removing all polynomial terms that exceed a specified maximum order. This forces the object into a lower-order approximation. Registers the truncated result and returns its reference.",
)(functions.truncate_object)

analyze_taylor_map = tool(
    "analyze_taylor_map",
    args_schema=schemas.AnalyzeTaylorMapInput,
    description="Performs a deep structural analysis of a TaylorMap, computing the trace, the eigenvalues of its linear Jacobian matrix, and a boolean flag indicating if the map is analytically invertible at the origin. Returns a dictionary of these analytical metrics.",
)(functions.analyze_taylor_map)

evaluate_mtf_batch = tool(
    "evaluate_mtf_batch",
    args_schema=schemas.EvaluateMtfBatchInput,
    description="Performs highly optimized, vectorized batch evaluation of an MTF at multiple coordinate points simultaneously. The input must be a 2D list of floats (list of points). Returns a 1D list containing the evaluation scalar for each respective point.",
)(functions.evaluate_mtf_batch)

evaluate_taylor_map_batch = tool(
    "evaluate_taylor_map_batch",
    args_schema=schemas.EvaluateTaylorMapBatchInput,
    description="Performs highly optimized, vectorized batch evaluation of a TaylorMap at multiple coordinate points simultaneously. The input must be a 2D list of floats. Returns a 2D list representing the transformed coordinates for each respective point.",
)(functions.evaluate_taylor_map_batch)

analyze_mtf_diagnostics = tool(
    "analyze_mtf_diagnostics",
    args_schema=schemas.AnalyzeMtfDiagnosticsInput,
    description="Computes advanced mathematical diagnostics for an MTF, including its standard norm, weighted norm (if weight array is provided), and stability/order decay estimates. Returns a dictionary of these structural metrics.",
)(functions.analyze_mtf_diagnostics)

compute_poisson_bracket = tool(
    "compute_poisson_bracket",
    args_schema=schemas.ComputePoissonBracketInput,
    description="Computes the classical Poisson Bracket {A, B} between two scalar MTFs acting as Hamiltonian observables. Assumes the canonical symplectic structure where variables are ordered as [q1, p1, q2, p2, ...]. Registers the resulting bracket MTF.",
)(functions.compute_poisson_bracket)

compute_map_sensitivity = tool(
    "compute_map_sensitivity",
    args_schema=schemas.ComputeMapSensitivityInput,
    description="Calculates the parameter sensitivity of a TaylorMap relative to a specific evaluation point. Conceptually similar to a normalized gradient matrix or scaling transformation. Registers the sensitivity map and returns its reference.",
)(functions.compute_map_sensitivity)

extract_map_component = tool(
    "extract_map_component",
    args_schema=schemas.ExtractMapComponentInput,
    description="Extracts a single scalar MTF representing a specific coordinate component from a vector-valued TaylorMap. The component index is 0-based. Registers the extracted MTF and returns its reference.",
)(functions.extract_map_component)
