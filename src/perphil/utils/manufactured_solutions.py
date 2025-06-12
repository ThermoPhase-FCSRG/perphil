import firedrake as fd

from perphil.models.dpp.parameters import DPPParameters


def exact_expressions(
    mesh: fd.Mesh, dpp_params: DPPParameters
) -> tuple[fd.Expr, fd.Expr, fd.Expr, fd.Expr]:
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
