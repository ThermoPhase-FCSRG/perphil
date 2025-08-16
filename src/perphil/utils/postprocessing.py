from typing import Tuple, Optional, Any
import numpy as np
import firedrake as fd


def split_dpp_solution(dpp_solution: fd.Function) -> Tuple[fd.Function, fd.Function]:
    """
    Extract macro and micro pressure fields from a mixed DPP solution.

    :param dpp_solution:
        A Firedrake Function defined on a MixedFunctionSpace W = V x V.

    :return:
        A tuple (p1_h, p2_h) as Functions on the corresponding pressure subspaces.

    :raises ValueError:
        If the solution's function space is not a MixedFunctionSpace.
    """
    W = dpp_solution.function_space()
    if not hasattr(W, "num_sub_spaces") or W.num_sub_spaces() != 2:
        raise ValueError(f"Expected a 2-field MixedFunctionSpace, got {type(W)}")

    V1 = W.sub(0)
    V2 = W.sub(1)

    pressure_macro = fd.Function(V1, name="p1_h")
    pressure_micro = fd.Function(V2, name="p2_h")
    pressure_macro.assign(dpp_solution.sub(0))
    pressure_micro.assign(dpp_solution.sub(1))

    return pressure_macro, pressure_micro


def calculate_darcy_velocity_from_pressure(
    pressure_field: fd.Function,
    conductivity: fd.Constant | fd.Function,
    velocity_space: Optional[fd.FunctionSpace] = None,
    degree: int = 1,
) -> fd.Function:
    """
    Project the Darcy velocity u = -k * grad(p_h) into a VectorFunctionSpace.

    :param pressure_field:
        Scalar pressure Function defined on a CG space.

    :param conductivity:
        Conductivity associated with Darcy's Law.

    :param velocity_space:
        Optional pre-built VectorFunctionSpace. If None, a default CG space
        of given degree on the same mesh will be used.

    :param degree:
        Polynomial degree for the velocity interpolation if velocity_space is None.

    :return:
        A Firedrake Function representing the velocity field.
    """
    if velocity_space is None:
        mesh = pressure_field.function_space().mesh()
        velocity_space = fd.VectorFunctionSpace(mesh, "CG", degree)

    return fd.project(-conductivity * fd.grad(pressure_field), velocity_space)


def slice_along_x(scalar_field: fd.Function, x_value: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a scalar field along a vertical line x = constant.

    :param scalar_field:
        Scalar pressure Function defined on a CG space.

    :param x_value:
        The x-coordinate at which to slice the solution.

    :return:
        A tuple (y_points, values) where y_points is an array of y-coordinates
        and values are the sampled pressure values.
    """
    mesh = scalar_field.function_space().mesh()
    mesh_coordinates = fd.SpatialCoordinate(mesh)
    y_field = fd.Function(scalar_field.function_space()).interpolate(mesh_coordinates[1])
    y_points = np.unique(y_field.dat.data[:])

    values = np.array([scalar_field.at((x_value, y)) for y in y_points])
    return y_points, values


def l2_error(numerical: fd.Function, exact_expr: Any) -> float:
    """
    Compute the L2 norm of the error ||numerical - exact||_{L2} on the mesh.

    :param numerical:
        Scalar Function on a CG space.

    :param exact_expr:
        UFL expression or Function representing the exact solution.

    :return:
        L2 error as a Python float.
    """
    mesh = numerical.function_space().mesh()
    diff = numerical - exact_expr
    err_sq = fd.assemble(fd.inner(diff, diff) * fd.dx(domain=mesh))
    return float(err_sq**0.5)


def h1_seminorm_error(numerical: fd.Function, exact_expr: Any) -> float:
    """
    Compute the H1 seminorm of the error |numerical - exact|_{H1} on the mesh.

    :param numerical:
        Scalar Function on a CG space.

    :param exact_expr:
        UFL expression or Function representing the exact solution.

    :return:
        H1-seminorm error as a Python float.
    """
    mesh = numerical.function_space().mesh()
    grad_diff = fd.grad(numerical) - fd.grad(exact_expr)
    err_sq = fd.assemble(fd.inner(grad_diff, grad_diff) * fd.dx(domain=mesh))
    return float(err_sq**0.5)
