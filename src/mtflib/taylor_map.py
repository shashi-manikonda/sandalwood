r"""
Represents a vector-valued function using Taylor series.

This module defines the `TaylorMap` class, which encapsulates a list of
`MultivariateTaylorFunction` (DA vector) objects. It represents a function
from :math:`\mathbb{R}^n` to :math:`\mathbb{R}^m`, where each component of
the output vector is a Taylor series. This class provides methods for
vector arithmetic, composition, and inversion of such maps.
"""

import warnings

import numpy as np
from numpy.exceptions import ComplexWarning

from .taylor_function import MultivariateTaylorFunction


class TaylorMap:
    """
    Represents a function from R^n to R^m using Taylor series components.

    This class models a vector-valued function where each component is a
    `MultivariateTaylorFunction` (a DA vector). It is the primary tool for
    studying systems of equations and coordinate transformations.

    Attributes
    ----------
    components : np.ndarray
        A NumPy array of `MultivariateTaylorFunction` objects that form the
        components of the map.
    map_dim : int
        The dimension of the output space (m), which is the number of
        components.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction, TaylorMap
    >>> mtf.initialize_mtf(max_order=2, max_dimension=2) # doctest: +ELLIPSIS
    Initializing...
    >>>
    >>> # Create variables
    >>> x1 = MultivariateTaylorFunction.var(1, 2)
    >>> x2 = MultivariateTaylorFunction.var(2, 2)
    >>>
    >>> # Create a map F(x1, x2) = [x1 + x2, x1 - x2]
    >>> f1 = x1 + x2
    >>> f2 = x1 - x2
    >>> F = TaylorMap([f1, f2])
    >>>
    >>> # Create another map G(y1, y2) = [y1*y2, y1]
    >>> y1 = MultivariateTaylorFunction.var(1, 2)
    >>> y2 = MultivariateTaylorFunction.var(2, 2)
    >>> g1 = y1 * y2
    >>> g2 = y1
    >>> G = TaylorMap([g1, g2])
    >>>
    >>> # Compose the maps: H(x1, x2) = G(F(x1, x2))
    >>> H = G.compose(F)
    >>>
    >>> print(H) # doctest: +ELLIPSIS
    TaylorMap with 2 components...
    --- Component 1 ---
    ...
    --- Component 2 ---
    ...
    """

    def __init__(self, components: list[MultivariateTaylorFunction]):
        """
        Initializes a TaylorMap object.

        Parameters
        ----------
        components : list[MultivariateTaylorFunction]
            A list of `MultivariateTaylorFunction` objects that define the
            components of the map.
        """
        self.components = np.array(components)
        self.map_dim = len(components)

    def __add__(self, other):
        """
        Performs element-wise addition with another TaylorMap.

        Parameters
        ----------
        other : TaylorMap
            The TaylorMap to add. It must have the same `map_dim`.

        Returns
        -------
        TaylorMap
            A new TaylorMap representing the element-wise sum.
        """
        if not isinstance(other, TaylorMap):
            raise TypeError("Can only add a TaylorMap to another TaylorMap.")
        if self.map_dim != other.map_dim:
            raise ValueError("TaylorMap dimensions must match for addition.")

        new_components = self.components + other.components
        return TaylorMap(list(new_components))

    def __sub__(self, other):
        """
        Performs element-wise subtraction with another TaylorMap.

        Parameters
        ----------
        other : TaylorMap
            The TaylorMap to subtract. It must have the same `map_dim`.

        Returns
        -------
        TaylorMap
            A new TaylorMap representing the element-wise difference.
        """
        if not isinstance(other, TaylorMap):
            raise TypeError("Can only subtract a TaylorMap from another TaylorMap.")
        if self.map_dim != other.map_dim:
            raise ValueError("TaylorMap dimensions must match for subtraction.")

        new_components = self.components - other.components
        return TaylorMap(list(new_components))

    def __mul__(self, other):
        """
        Performs element-wise multiplication with another TaylorMap.

        This is the Hadamard product, not matrix multiplication.

        Parameters
        ----------
        other : TaylorMap
            The TaylorMap to multiply by. It must have the same `map_dim`.

        Returns
        -------
        TaylorMap
            A new TaylorMap representing the element-wise product.
        """
        if not isinstance(other, TaylorMap):
            raise TypeError("Can only multiply a TaylorMap by another TaylorMap.")
        if self.map_dim != other.map_dim:
            raise ValueError("TaylorMap dimensions must match for multiplication.")

        new_components = self.components * other.components
        return TaylorMap(list(new_components))

    def compose(self, other):
        r"""
        Composes this map with another, calculating `self(other(x))`.

        If `self` is a map from :math:`\mathbb{R}^n \\to \mathbb{R}^m` and
        `other` is a map from :math:`\mathbb{R}^k \\to \mathbb{R}^n`, the
        resulting composition is a map from :math:`\mathbb{R}^k \\to \mathbb{R}^m`.

        Parameters
        ----------
        other : TaylorMap
            The inner map in the composition. The output dimension of `other`
            must match the input dimension of `self`.

        Returns
        -------
        TaylorMap
            A new TaylorMap representing the composed function.

        Raises
        ------
        TypeError
            If `other` is not a TaylorMap.
        ValueError
            If the dimensions are incompatible for composition.
        """
        if not isinstance(other, TaylorMap):
            raise TypeError(
                "Composition is only defined between two TaylorMap objects."
            )

        if self.map_dim == 0:
            return TaylorMap([])

        self_input_dim = self.components[0].dimension

        if self_input_dim != other.map_dim:
            raise ValueError(
                f"Cannot compose maps: self input dimension ({self_input_dim}) "
                f"must equal other output dimension ({other.map_dim})."
            )

        new_components = []
        # The new dimension will be the input dimension of the 'other' map.
        new_dimension = other.components[0].dimension if other.map_dim > 0 else 0

        # Cache for powers of the input map components to avoid re-computation.
        # Structure: cache[var_idx][power] -> MultivariateTaylorFunction
        # We initialize it as a list of dictionaries.
        component_powers_cache = [{} for _ in range(self_input_dim)]

        for component_mtf in self.components:
            composed_component = MultivariateTaylorFunction.from_constant(
                0.0, dimension=new_dimension
            )
            for i in range(len(component_mtf.coeffs)):
                exponent = component_mtf.exponents[i]
                coeff = component_mtf.coeffs[i]

                # Start term with the scalar coefficient
                # Optimization: Initialize with coeff directly if it's the first
                # multiplication. But here we handle dimensionality.
                # Let's create a term_mtf initialized to 1.0 (identity)
                # Optimization: if coeff is zero, skip? (Assuming sparse handling, but
                # check for explicit zeros)
                if abs(coeff) < 1e-16:
                    # Simple check, though MTF handles this internally usually
                    continue

                term_mtf = None

                for var_idx, power in enumerate(exponent):
                    if power > 0:
                        # Retrieve or compute the power of the map component
                        if power not in component_powers_cache[var_idx]:
                            component_powers_cache[var_idx][power] = (
                                other.components[var_idx] ** power
                            )

                        factor = component_powers_cache[var_idx][power]

                        if term_mtf is None:
                            term_mtf = factor
                        else:
                            term_mtf *= factor

                if term_mtf is None:
                    # Constant term (all powers 0)
                    term_mtf = MultivariateTaylorFunction.from_constant(
                        1.0, dimension=new_dimension
                    )

                composed_component += term_mtf * coeff

            new_components.append(composed_component)

        return TaylorMap(new_components).truncate(
            MultivariateTaylorFunction.get_max_order()
        )

    def get_component(self, index: int) -> MultivariateTaylorFunction:
        """
        Retrieves a component function from the map.

        Parameters
        ----------
        index : int
            The 0-based index of the component to retrieve.

        Returns
        -------
        MultivariateTaylorFunction
            The component at the specified index.
        """
        return self.components[index]

    def get_coefficient(self, component_index: int, exponent_array: np.ndarray):
        """
        Gets the coefficient of a specific term in a component function.

        Parameters
        ----------
        component_index : int
            The 0-based index of the component function.
        exponent_array : np.ndarray
            The multi-index exponent of the term.

        Returns
        -------
        float or complex
            The value of the coefficient.
        """
        return (
            self.components[component_index]
            .extract_coefficient(tuple(exponent_array))
            .item()
        )

    def set_coefficient(
        self, component_index: int, exponent_array: np.ndarray, new_value
    ):
        """
        Sets the coefficient of a specific term in a component function.

        Parameters
        ----------
        component_index : int
            The 0-based index of the component function.
        exponent_array : np.ndarray
            The multi-index exponent of the term.
        new_value : float or complex
            The new value for the coefficient.
        """
        self.components[component_index].set_coefficient(
            tuple(exponent_array), new_value
        )

    def add_component(self, new_component: MultivariateTaylorFunction):
        """
        Adds a new component function to the map.

        Parameters
        ----------
        new_component : MultivariateTaylorFunction
            The new component to add to the end of the map.
        """
        self.components = np.append(self.components, new_component)
        self.map_dim = len(self.components)

    def remove_component(self, index: int):
        """
        Removes a component function from the map by its index.

        Parameters
        ----------
        index : int
            The 0-based index of the component to remove.
        """
        self.components = np.delete(self.components, index)
        self.map_dim = len(self.components)

    def truncate(self, order: int):
        """
        Truncates all component functions to a specified order.

        Parameters
        ----------
        order : int
            The maximum order to which all components will be truncated.

        Returns
        -------
        TaylorMap
            A new TaylorMap with the truncated components.
        """
        new_components = [c.truncate(order) for c in self.components]
        return TaylorMap(new_components)

    def trace(self):
        """
        Calculates the trace of the first-order part of the map.
        This is only defined for maps from N-dim to N-dim space.
        """
        if self.map_dim == 0:
            return 0.0

        if self.map_dim != self.components[0].dimension:
            raise ValueError(
                "Trace is only defined for maps from N-dim to N-dim space."
            )

        trace_val = 0.0j
        for i in range(self.map_dim):
            exponent = [0] * self.map_dim
            exponent[i] = 1
            trace_val += self.get_coefficient(i, np.array(exponent))

        return trace_val

    def map_sensitivity(self, scaling_factors: list[float]):
        """
        Returns a new TaylorMap with coefficients scaled for sensitivity analysis.
        """
        if self.map_dim > 0 and len(scaling_factors) != self.components[0].dimension:
            raise ValueError("Number of scaling factors must match input dimension.")

        new_map_components = []
        for component in self.components:
            new_coeffs = component.coeffs.copy()
            for i in range(len(component.coeffs)):
                exponent = component.exponents[i]
                scaling_factor = np.prod(np.array(scaling_factors) ** exponent)
                new_coeffs[i] *= scaling_factor

            new_component = MultivariateTaylorFunction(
                (component.exponents.copy(), new_coeffs),
                dimension=component.dimension,
            )
            new_map_components.append(new_component)

        return TaylorMap(new_map_components)

    def substitute(self, variable_map: dict):
        """
        Performs partial or full substitution.

        Args:
            variable_map: A dict of {var_index: value}, where value can be a number.
                          var_index is 1-based.

        Returns:
            A new TaylorMap if the substitution is partial, or a NumPy array of floats
            if the substitution is full.
        """
        if self.map_dim == 0:
            return np.array([])

        is_full_numeric = (
            all(
                isinstance(v, (int, float, complex, np.number))
                for v in variable_map.values()
            )
            and len(variable_map) == self.components[0].dimension
        )

        if is_full_numeric:
            eval_point = [0] * self.components[0].dimension
            for i, val in variable_map.items():
                eval_point[i - 1] = val

            return np.array([c.eval(eval_point).item() for c in self.components])

        new_components = []
        for component in self.components:
            new_component = component.copy()
            for var_index, value in variable_map.items():
                if isinstance(value, (int, float, complex, np.number)):
                    new_component.substitute_variable_inplace(var_index, value)
                else:
                    raise TypeError(
                        f"Unsupported substitution type: {type(value)}. "
                        "Only numeric values are supported in substitute."
                    )
            new_components.append(new_component)

        return TaylorMap(new_components)

    def __repr__(self):
        """
        Returns a concise string representation of the TaylorMap.
        """
        if self.map_dim > 0:
            return (
                f"TaylorMap(map_dim={self.map_dim}, "
                f"input_dim={self.components[0].dimension})"
            )
        return "TaylorMap(map_dim=0, input_dim=N/A)"

    def __str__(self):
        """
        Returns a detailed string representation of the TaylorMap.
        """
        if self.map_dim == 0:
            return "Empty TaylorMap"

        representation = (
            f"TaylorMap with {self.map_dim} components "
            f"(input dim: {self.components[0].dimension}):\n"
        )
        for i, component in enumerate(self.components):
            representation += f"--- Component {i + 1} ---\n"
            representation += str(component) + "\n"
        return representation

    def invert(self):
        r"""
        Computes the inverse of the TaylorMap using fixed-point iteration.

        This method finds the Taylor series for the inverse of a map `F`.
        The map `F` must be a square map (from :math:`\mathbb{R}^n` to
        :math:`\mathbb{R}^n`) with no constant term and an invertible
        linear part (Jacobian).

        The inversion is performed using the iterative formula:
        :math:`F^{-1}_{k+1} = \\beta^{-1} \\circ (I - G \\circ F^{-1}_k)`
        where:
        - :math:`F = \\beta + G` (decomposition into linear and non-linear parts)
        - :math:`\\beta` is the linear part of `F`.
        - :math:`G` is the non-linear part of `F` (orders > 1).
        - :math:`I` is the identity map.
        - :math:`F^{-1}_k` is the k-th order approximation of the inverse.

        The iteration starts with :math:`F^{-1}_1 = \\beta^{-1}` and proceeds
        up to the maximum order defined in the global settings.

        Returns
        -------
        TaylorMap
            A new TaylorMap representing the inverse map :math:`F^{-1}`.

        Raises
        ------
        ValueError
            If the map is not square, contains constant terms, or if its
            linear part (Jacobian) is singular.

        Examples
        --------
        >>> from mtflib import MultivariateTaylorFunction, TaylorMap, mtf
        >>> # The library is already initialized from the class docstring example
        >>> x, y = mtf.var(1), mtf.var(2)
        >>>
        >>> # Define a map F(x, y) = [x + y^2, y - x^2]
        >>> F = TaylorMap([x + y**2, y - x**2])
        >>>
        >>> # Compute the inverse map G = F_inv
        >>> G = F.invert()
        >>>
        >>> # Compose F with its inverse G. The result should be the identity map.
        >>> Identity = F.compose(G)
        >>> print(Identity) # doctest: +ELLIPSIS
        TaylorMap with 2 components...
        --- Component 1 ---
        ...
        --- Component 2 ---
        ...
        """
        # --- Pre-condition Checks ---
        if self.map_dim == 0:
            return TaylorMap([])  # Inverse of an empty map is an empty map

        dim = self.components[0].dimension
        if self.map_dim != dim:
            raise ValueError(
                f"Map must be square to be invertible (input_dim={dim}, "
                f"output_dim={self.map_dim})."
            )

        # Check for constant terms
        zero_exp = tuple([0] * dim)
        for i, component in enumerate(self.components):
            const_term = component.extract_coefficient(zero_exp).item()
            if abs(const_term) > 1e-14:  # Use a tolerance for floating point
                raise ValueError(
                    f"Map must have no constant terms to be invertible. "
                    f"Component {i} has constant term {const_term}."
                )

        # --- Algorithm Step 1: Extract Linear Part (β) and Jacobian ---
        jacobian = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(dim):
            for j in range(dim):
                exp = tuple([1 if k == j else 0 for k in range(dim)])
                coeff = self.get_coefficient(i, np.array(exp))
                jacobian[i, j] = coeff

        # Check for invertible linear part
        if abs(np.linalg.det(jacobian)) < 1e-14:
            raise ValueError(
                "The linear part of the map is not invertible (Jacobian is singular)."
            )

        # --- Algorithm Step 4 (early): Invert Linear Part (β⁻¹) ---
        inv_jacobian = np.linalg.inv(jacobian)
        inv_linear_components = []
        for i in range(dim):
            comp_mtf = MultivariateTaylorFunction.from_constant(0.0, dimension=dim)
            for j in range(dim):
                if abs(inv_jacobian[i, j]) > 1e-14:
                    var_mtf = MultivariateTaylorFunction.var(j + 1, dim)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ComplexWarning)
                        comp_mtf += float(inv_jacobian[i, j]) * var_mtf
            inv_linear_components.append(comp_mtf)
        beta_inv = TaylorMap(inv_linear_components)

        # --- Algorithm Step 2 & 3: Extract G and create Identity ---
        # We don't need to create beta explicitly. G = F - beta
        # The non-linear part G can be calculated on the fly.
        # Let's get the full non-linear part first.

        non_linear_components = []
        for component in self.components:
            # Create a new MTF with only non-linear terms (order > 1)
            nl_coeffs = {}
            for k in range(len(component.coeffs)):
                exp = tuple(component.exponents[k])
                if sum(exp) > 1:
                    nl_coeffs[exp] = component.coeffs[k]
            non_linear_components.append(
                MultivariateTaylorFunction(nl_coeffs, dimension=dim)
            )
        G = TaylorMap(non_linear_components)

        identity_components = [
            MultivariateTaylorFunction.var(i + 1, dim) for i in range(dim)
        ]
        identity_map = TaylorMap(identity_components)

        # --- Algorithm Step 5: Fixed-Point Iteration ---
        F_inv = beta_inv  # Initial guess

        max_order = MultivariateTaylorFunction.get_max_order()
        for _ in range(max_order - 1):
            composition_G_F_inv = G.compose(F_inv).truncate(max_order)
            inner_map = identity_map - composition_G_F_inv
            F_inv = beta_inv.compose(inner_map).truncate(max_order)

        return F_inv
