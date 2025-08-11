import firedrake as fd
from ufl.core.expr import Expr

from perphil.models.dpp.parameters import DPPParameters


def exact_expressions(mesh: fd.Mesh, dpp_params: DPPParameters) -> tuple[Expr, Expr, Expr, Expr]:
    """
    Build analytic UFL expressions for u1, p1, u2, p2.

    :param mesh:
        A Firedrake Mesh instance.

    :param dpp_params:
        DPPParameters container with model constants.

    :return:
        Tuple of UFL expressions (u1_expr, p1_expr, u2_expr, p2_expr).
    """
    # Get DOFs coordinate points
    x, y = fd.SpatialCoordinate(mesh)

    # Get the model parameters
    k1 = dpp_params.k1
    assert isinstance(k1, fd.Constant)
    k2 = dpp_params.k2
    assert isinstance(k2, fd.Constant)
    beta = dpp_params.beta
    mu = dpp_params.mu
    eta = dpp_params.eta

    # Exact solution expressions
    u1_expr = fd.as_vector(
        [
            -k1 * (fd.exp(fd.pi * x) * fd.sin(fd.pi * y)),
            -k1 * (fd.exp(fd.pi * x) * fd.cos(fd.pi * y) - (eta / (beta * k1)) * fd.exp(eta * y)),
        ]
    )
    p1_expr = (mu / fd.pi) * fd.exp(fd.pi * x) * fd.sin(fd.pi * y) - (mu / (beta * k1)) * fd.exp(
        eta * y
    )

    u2_expr = fd.as_vector(
        [
            -k2 * (fd.exp(fd.pi * x) * fd.sin(fd.pi * y)),
            -k2 * (fd.exp(fd.pi * x) * fd.cos(fd.pi * y) + (eta / (beta * k2)) * fd.exp(eta * y)),
        ]
    )
    p2_expr = (mu / fd.pi) * fd.exp(fd.pi * x) * fd.sin(fd.pi * y) + (mu / (beta * k2)) * fd.exp(
        eta * y
    )

    return u1_expr, p1_expr, u2_expr, p2_expr


def exact_expressions_3d(mesh: fd.Mesh, dpp_params: DPPParameters) -> tuple[Expr, Expr, Expr, Expr]:
    """
    3D manufactured analytic expressions (u1, p1, u2, p2) based on the paper's Eq. (6.3),
    corrected so that p2 uses k2 in the denominator.

    Unknowns follow the primal two-pressure DPP model with Darcy velocities:
        u1 = -(k1/mu) grad p1,  u2 = -(k2/mu) grad p2

    p1(x,y,z) = (mu/pi) e^{pi x} (sin(pi y) + sin(pi z)) - (mu/(beta k1)) (e^{eta y} + e^{eta z})
    p2(x,y,z) = (mu/pi) e^{pi x} (sin(pi y) + sin(pi z)) + (mu/(beta k2)) (e^{eta y} + e^{eta z})

    :param mesh: Firedrake Mesh (3D)
    :param dpp_params: DPP parameters (k1, k2, beta, mu). eta is taken from dpp_params.eta
    :return: Tuple of UFL expressions (u1_expr, p1_expr, u2_expr, p2_expr)
    """
    # Coordinates (expects 3D mesh)
    x, y, z = fd.SpatialCoordinate(mesh)

    # Parameters
    k1: fd.Constant = dpp_params.k1  # type: ignore[assignment]
    k2: fd.Constant = dpp_params.k2  # type: ignore[assignment]
    beta: fd.Constant = dpp_params.beta  # type: ignore[assignment]
    mu: fd.Constant = dpp_params.mu  # type: ignore[assignment]
    eta: fd.Constant = dpp_params.eta  # derived from (k1,k2,beta)

    # Common factors
    common_xy = fd.exp(fd.pi * x)
    s = fd.sin(fd.pi * y) + fd.sin(fd.pi * z)
    e_y = fd.exp(eta * y)
    e_z = fd.exp(eta * z)

    p1_expr = (mu / fd.pi) * common_xy * s - (mu / (beta * k1)) * (e_y + e_z)
    p2_expr = (mu / fd.pi) * common_xy * s + (mu / (beta * k2)) * (e_y + e_z)

    # Velocities from Darcy's law: u = -(k/mu) grad p
    u1_expr = -(k1 / mu) * fd.grad(p1_expr)
    u2_expr = -(k2 / mu) * fd.grad(p2_expr)

    return u1_expr, p1_expr, u2_expr, p2_expr


def interpolate_exact(
    mesh: fd.Mesh,
    velocity_space: fd.FunctionSpace,
    pressure_space: fd.FunctionSpace,
    dpp_params: DPPParameters,
) -> tuple[fd.Function, fd.Function, fd.Function, fd.Function]:
    """
    Interpolate analytic expressions into Firedrake Functions.

    :param mesh:
        A Firedrake Mesh instance.

    :param velocity_space:
        VectorFunctionSpace for u1 and u2.

    :param pressure_space:
        FunctionSpace for p1 and p2.

    :param params:
        DPPParameters container with model constants.

    :return:
        Tuple of Firedrake Functions (u1_exact, p1_exact, u2_exact, p2_exact).
    """
    u1_e, p1_e, u2_e, p2_e = exact_expressions(mesh, dpp_params)

    u1_exact = fd.Function(velocity_space, name="u1_exact")
    u1_exact.interpolate(u1_e)

    p1_exact = fd.Function(pressure_space, name="p1_exact")
    p1_exact.interpolate(p1_e)

    u2_exact = fd.Function(velocity_space, name="u2_exact")
    u2_exact.interpolate(u2_e)

    p2_exact = fd.Function(pressure_space, name="p2_exact")
    p2_exact.interpolate(p2_e)

    return u1_exact, p1_exact, u2_exact, p2_exact
