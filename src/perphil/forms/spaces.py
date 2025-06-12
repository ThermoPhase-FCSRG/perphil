import firedrake as fd
from typing import Tuple


def create_function_spaces(
    mesh: fd.Mesh,
    velocity_deg: int = 1,
    pressure_deg: int = 1,
    velocity_family: str = "CG",
    pressure_family: str = "CG",
) -> Tuple[fd.FunctionSpace, fd.FunctionSpace]:
    """
    Build velocity and pressure function spaces on the given mesh.

    :param mesh:
        A Firedrake Mesh instance on which to build spaces.

    :param velocity_deg:
        Polynomial degree for velocity space.

    :param pressure_deg:
        Polynomial degree for pressure space.

    :param velocity_family:
        UFL family for velocity (e.g., "CG").

    :param pressure_family:
        UFL family for pressure (e.g., "CG").

    :return:
        A tuple (U, V) where U is the VectorFunctionSpace for velocity
        and V is the scalar FunctionSpace for pressure.
    """
    U = fd.VectorFunctionSpace(mesh, velocity_family, velocity_deg)
    V = fd.FunctionSpace(mesh, pressure_family, pressure_deg)
    return U, V
