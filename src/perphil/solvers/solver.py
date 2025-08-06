from typing import List, Dict, Tuple
import attr
import firedrake as fd
import logging

import numpy as np

from perphil.models.dpp.parameters import DPPParameters
from perphil.forms.dpp import dpp_form, dpp_splitted_form

logger = logging.getLogger(__name__)


@attr.define(frozen=True)
class Solution:
    """
    Represents the result of a solver computation.

    Attributes:
        solution (fd.Function | Tuple[fd.Function, fd.Function]): The computed solution, which may be a single function or a tuple of functions depending on the problem.
        iteration_number (int): The number of iterations performed by the solver.
        residual_error (float | np.float64): The final residual error after the solver completes.
    """

    solution: fd.Function | Tuple[fd.Function, fd.Function]
    iteration_number: int
    residual_error: float | np.float64


def solve_dpp(
    W: fd.FunctionSpace,
    model_params: DPPParameters,
    bcs: List[fd.DirichletBC],
    solver_parameters: Dict = {},
    options_prefix: str = "dpp",
) -> Solution:
    """
    Solve the monolithic/preconditioned double-porosity/permeability linear system.

    :param W:
        MixedFunctionSpace for (p1, p2) pressures.

    :param model_params:
        DPPParameters container with model constants.

    :param bcs:
        List of DirichletBC objects applied to W.

    :param solver_parameters:
        PETSc solver parameters to be employed.

    :param options_prefix:
        Prefix for solver options (default: "dpp").

    :return:
        A Function on W containing the solution (p1, p2).

    :raises ValueError:
        If W is not a MixedFunctionSpace.
    """
    if not hasattr(W, "num_sub_spaces") or W.num_sub_spaces() != 2:
        raise ValueError(f"Expected a 2-field MixedFunctionSpace, got {type(W)}")

    a, L = dpp_form(W, model_params)
    solution = fd.Function(W)
    problem = fd.LinearVariationalProblem(a, L, solution, bcs=bcs)
    solver = fd.LinearVariationalSolver(
        problem, solver_parameters=solver_parameters, options_prefix=options_prefix
    )

    solver.solve()

    num_iterations = solver.snes.ksp.getIterationNumber()
    residual_error = solver.snes.ksp.getResidualNorm()
    solution_data = Solution(solution, num_iterations, residual_error)
    return solution_data


def solve_dpp_nonlinear(
    W: fd.FunctionSpace,
    model_params: DPPParameters,
    bcs: List[fd.DirichletBC],
    solver_parameters: Dict = {},
    options_prefix: str = "dpp_nonlinear",
) -> Solution:
    """
    Solve the double-porosity/permeability system using nonlinear (SNES) PETSc infrastructure.

    This solver allows the use of Richardson/Picard methods as defined and managed internally by PETSc.

    :param W:
        MixedFunctionSpace for (p1, p2) unknowns.

    :param model_params:
        DPPParameters container with model constants.

    :param bcs:
        List of DirichletBC objects applied to W.

    :param solver_parameters:
        PETSc solver parameter dictionary options.

    :param options_prefix:
        Prefix for solver options (default: "dpp_nonlinear").

    :return:
        A Function on W containing the converged solution (p1, p2).

    :raises ValueError:
        If W is not a MixedFunctionSpace.
    """
    if not hasattr(W, "num_sub_spaces") or W.num_sub_spaces() != 2:
        raise ValueError(f"Expected a 2-field MixedFunctionSpace, got {type(W)}")

    F, fields = dpp_splitted_form(W, model_params)
    problem = fd.NonlinearVariationalProblem(F, fields, bcs=bcs)
    solver = fd.NonlinearVariationalSolver(
        problem, solver_parameters=solver_parameters, options_prefix=options_prefix
    )

    solver.solve()

    num_iterations = solver.snes.getIterationNumber()
    residual_error = solver.snes.getFunctionNorm()
    solution_data = Solution(fields, num_iterations, residual_error)
    return solution_data
