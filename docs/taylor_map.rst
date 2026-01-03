.. _taylor_map:

Working with Taylor Maps
========================

The ``TaylorMap`` class is a powerful tool for representing and manipulating
vector-valued functions where each component is a Taylor series. It is the
primary object for studying systems of equations, coordinate transformations,
and for composing complex functions.

A ``TaylorMap`` can be thought of as a function from :math:`\mathbb{R}^n` to
:math:`\mathbb{R}^m`, where `n` is the number of variables and `m` is the
number of component functions.

Creating a TaylorMap
--------------------

A ``TaylorMap`` is created from a list of ``MultivariateTaylorFunction``
objects. Each function in the list becomes a component of the map.

.. code-block:: python

   from mtflib import MultivariateTaylorFunction, TaylorMap, Var

   # Initialize for 2 variables, up to order 3
   MultivariateTaylorFunction.initialize_mtf(max_order=3, max_dimension=2)

   # Define variables
   x, y = Var(1), Var(2)

   # Define component functions
   f1 = x + y**2
   f2 = y - x**2

   # Create the TaylorMap
   F = TaylorMap([f1, f2])

   print(F)

This creates a map :math:`F(x, y) = [x + y^2, y - x^2]`. The output of the
print statement will be:

.. code-block:: text

   TaylorMap with 2 components in 2 variables
   Component 1:
   MultivariateTaylorFunction in 2 variables up to order 3
   Number of non-zero terms: 2
      Coefficient  Order Exponents
   0          1.0      1    (1, 0)
   1          1.0      2    (0, 2)
   Component 2:
   MultivariateTaylorFunction in 2 variables up to order 3
   Number of non-zero terms: 2
      Coefficient  Order Exponents
   0          1.0      1    (0, 1)
   1         -1.0      2    (2, 0)

Accessing Components
--------------------

You can access individual components of a ``TaylorMap`` using standard list
indexing:

.. code-block:: python

    # Get the first component of the map
    first_component = F[0]
    print(first_component)

    # Get the last component of the map
    last_component = F[-1]
    print(last_component)

Composing Taylor Maps
---------------------

One of the most powerful features of ``TaylorMap`` is the ability to compose
maps. If you have two maps, `F` and `G`, you can compute the composition
:math:`H(x) = G(F(x))` using the ``compose`` method.

.. code-block:: python

   # Create another map G(a, b) = [a*b, a]
   a, b = Var(1), Var(2)
   g1 = a * b
   g2 = a
   G = TaylorMap([g1, g2])

   # Compose the maps: H(x, y) = G(F(x, y))
   H = G.compose(F)

   print(H)

The resulting map `H` will have the Taylor series expansion of the composed
function.

Inverting Taylor Maps
---------------------

The ``invert`` method allows you to compute the inverse of a square map
(i.e., a map from :math:`\mathbb{R}^n` to :math:`\mathbb{R}^n`). The map must
have no constant term and an invertible linear part (Jacobian).

.. code-block:: python

   # Define a map F(x, y) = [x + y^2, y - x^2]
   F = TaylorMap([x + y**2, y - x**2])

   # Compute the inverse map G = F_inv
   G = F.invert()

   # To verify, compose F with its inverse G.
   # The result should be the identity map [x, y].
   Identity = F.compose(G)

   print(Identity)

This is particularly useful for solving systems of non-linear equations and
for changing coordinate systems.

Connection to Differential Equations
------------------------------------

Taylor maps are also deeply connected to the study of ordinary differential
equations (ODEs). The solution of an ODE can be represented as a Taylor map
that propagates the initial conditions forward in time. This map, often
called the "flow" of the ODE, provides a high-order approximation of the
system's state at a future time. This is a more advanced topic and is
covered in the "Advanced Topics" section of the documentation.

See Also
--------

- :class:`~mtflib.taylor_function.MultivariateTaylorFunction`
