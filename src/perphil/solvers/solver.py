from typing import List, Dict
import firedrake as fd

from perphil.models.dpp.parameters import DPPParameters
from perphil.forms.dpp import dpp_form, dpp_splitted_form


def solve_dpp(
    W: fd.FunctionSpace,
    model_params: DPPParameters,
    bcs: List[fd.DirichletBC],
    solver_parameters: Dict,
) -> fd.Function:
    """
    Solve the monolithic double-porosity/permeability linear system.

    :param W:
        MixedFunctionSpace for (p1, p2) pressures.

    :param params:
        DPPParameters container with model constants.

    :param bcs:
        List of DirichletBC objects applied to W.

    :param solver_parameters:
        PETSc solver parameter dictionary.

    :return:
        A Function on W containing the solution (p1, p2).

    :raises ValueError:
        If W is not a MixedFunctionSpace.
    """
    if not isinstance(W, fd.MixedFunctionSpace):
        raise ValueError(f"Expected a MixedFunctionSpace for W, got {type(W)}")

    a, L = dpp_form(W, model_params)
    solution = fd.Function(W)
    problem = fd.LinearVariationalProblem(a, L, solution, bcs=bcs)
    solver = fd.LinearVariationalSolver(problem, solver_parameters=solver_parameters)
    solver.solve()
    return solution


def solve_dpp_splitted(
    W: fd.FunctionSpace,
    model_params: DPPParameters,
    bcs: List[fd.DirichletBC],
    solver_parameters: Dict,
    options_prefix: str = "dpp",
) -> fd.Function:
    """
    Solve the double-porosity/permeability system using Picard (fixed-point) iterations.

    :param W:
        MixedFunctionSpace for (p1, p2) unknowns.

    :param params:
        DPPParameters container with model constants.

    :param bcs:
        List of DirichletBC objects applied to W.

    :param solver_parameters:
        PETSc solver parameter dictionary for the nonlinear solver.

    :param options_prefix:
        Prefix for solver options (default: "dpp").

    :return:
        A Function on W containing the converged solution (p1, p2).

    :raises ValueError:
        If W is not a MixedFunctionSpace.
    """
    if not isinstance(W, fd.MixedFunctionSpace):
        raise ValueError(f"Expected a MixedFunctionSpace for W, got {type(W)}")

    F, fields = dpp_splitted_form(W, model_params)
    problem = fd.NonlinearVariationalProblem(F, fields, bcs=bcs)
    solver = fd.NonlinearVariationalSolver(
        problem, solver_parameters=solver_parameters, options_prefix=options_prefix
    )
    solver.solve()
    return fields
