# src/dpp/postprocess.py

from typing import Tuple, Optional
import numpy as np
import firedrake as fd


def split_solution(solution: fd.Function) -> Tuple[fd.Function, fd.Function]:
    """
    Extract macro and micro pressure fields from a mixed DPP solution.

    :param solution:
        A Firedrake Function defined on a MixedFunctionSpace W = V x V.

    :return:
        A tuple (p1_h, p2_h) as Functions on the corresponding pressure subspaces.

    :raises ValueError:
        If the solution's function space is not a MixedFunctionSpace.
    """
    W = solution.function_space()
    if not isinstance(W, fd.MixedFunctionSpace):
        raise ValueError(f"Expected solution on MixedFunctionSpace, got {type(W)}")

    V1 = W.sub(0)
    V2 = W.sub(1)

    p1_h = fd.Function(V1, name="p1_h")
    p2_h = fd.Function(V2, name="p2_h")
    p1_h.assign(solution.sub(0))
    p2_h.assign(solution.sub(1))

    return p1_h, p2_h


def project_velocity(
    p_h: fd.Function, velocity_space: Optional[fd.FunctionSpace] = None, degree: int = 1
) -> fd.Function:
    """
    Project the Darcy velocity u = -grad(p_h) into a VectorFunctionSpace.

    :param p_h:
        Scalar pressure Function defined on a CG space.

    :param velocity_space:
        Optional pre-built VectorFunctionSpace. If None, a default CG space
        of given degree on the same mesh will be used.

    :param degree:
        Polynomial degree for the velocity interpolation if velocity_space is None.

    :return:
        A Firedrake Function representing the velocity field.
    """
    if velocity_space is None:
        mesh = p_h.function_space().mesh()
        velocity_space = fd.VectorFunctionSpace(mesh, "CG", degree)

    return fd.project(-fd.grad(p_h), velocity_space)


def slice_along_x(p_h: fd.Function, x_value: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a scalar field along a vertical line x = constant.

    :param p_h:
        Scalar pressure Function defined on a CG space.

    :param x_value:
        The x-coordinate at which to slice the solution.

    :return:
        A tuple (y_points, values) where y_points is an array of y-coordinates
        and values are the sampled pressure values.
    """
    mesh = p_h.function_space().mesh()
    coord = fd.SpatialCoordinate(mesh)
    y_field = fd.Function(p_h.function_space()).interpolate(coord[1])
    y_points = np.unique(y_field.dat.data[:])

    values = np.array([p_h.at((x_value, y)) for y in y_points])
    return y_points, values
