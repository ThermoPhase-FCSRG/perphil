from typing import List, Dict, Tuple
import attr
import firedrake as fd
from petsc4py import PETSc
import logging

import numpy as np

from perphil.models.dpp.parameters import DPPParameters
from perphil.forms.dpp import dpp_delayed_form, dpp_form, dpp_splitted_form

logger = logging.getLogger(__name__)


@attr.define(frozen=True)
class Solution:
    """
    TODO.
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


def solve_dpp_picard(
    macro_function_space: fd.FunctionSpace,
    micro_function_space: fd.FunctionSpace,
    model_params: DPPParameters,
    bcs_macro: List[fd.DirichletBC],
    bcs_micro: List[fd.DirichletBC],
    macro_solver_parameters: Dict = {},
    micro_solver_parameters: Dict = {},
    picard_damping_parameter: fd.Constant = fd.Constant(1),
    picard_rel_error: float = 1e-12,
    picard_max_iteration_number: int = 10000,
    picard_enable_logging: bool = True,
) -> Solution:
    """
    TODO.

    :param macro_function_space: _description_
    :type macro_function_space: fd.FunctionSpace
    :param micro_function_space: _description_
    :type micro_function_space: fd.FunctionSpace
    :param model_params: _description_
    :type model_params: DPPParameters
    :param bcs_macro: _description_
    :type bcs_macro: List[fd.DirichletBC]
    :param bcs_micro: _description_
    :type bcs_micro: List[fd.DirichletBC]
    :param macro_solver_parameters: _description_, defaults to {}
    :type macro_solver_parameters: Dict, optional
    :param micro_solver_parameters: _description_, defaults to {}
    :type micro_solver_parameters: Dict, optional
    :param picard_damping_parameter: _description_, defaults to fd.Constant(1)
    :type picard_damping_parameter: fd.Constant, optional
    :param picard_rel_error: _description_, defaults to 1e-12
    :type picard_rel_error: float, optional
    :param picard_max_iteration_number: _description_, defaults to 10000
    :type picard_max_iteration_number: int, optional
    :param picard_enable_logging: _description_, defaults to True
    :type picard_enable_logging: bool, optional
    :return: _description_
    :rtype: Tuple[fd.Function, fd.Function]
    """
    # Alias for the function spaces
    V_macro = macro_function_space
    V_micro = micro_function_space

    # Initial guess for pressure fields to be used in Picard iterations
    p1_old = fd.interpolate(fd.Constant(0), V_macro)
    p2_old = fd.interpolate(fd.Constant(0), V_micro)

    # Retrieve variation forms by scale
    forms_macro, forms_micro = dpp_delayed_form(V_macro, V_micro, model_params, p1_old, p2_old)
    a_macro, L_macro = forms_macro
    a_micro, L_micro = forms_micro

    ## Macro
    solution_macro = fd.Function(V_macro)
    problem_macro = fd.LinearVariationalProblem(
        a_macro,
        L_macro,
        solution_macro,
        bcs=bcs_macro,
    )
    solver_macro = fd.LinearVariationalSolver(
        problem_macro, solver_parameters=macro_solver_parameters
    )

    ## Micro
    solution_micro = fd.Function(V_micro)
    problem_micro = fd.LinearVariationalProblem(
        a_micro,
        L_micro,
        solution_micro,
        bcs=bcs_micro,
    )
    solver_micro = fd.LinearVariationalSolver(
        problem_micro, solver_parameters=micro_solver_parameters
    )

    ## Picard loop
    solution_data = _run_picard_iterations_for_dpp(
        (a_macro, L_macro),
        (a_micro, L_micro),
        bcs_macro,
        bcs_micro,
        solver_macro,
        solver_micro,
        solution_macro,
        solution_micro,
        p1_old,
        p2_old,
        picard_damping_parameter=picard_damping_parameter,
        picard_rtol=picard_rel_error,
        picard_max_iteration_number=picard_max_iteration_number,
        enable_logging=picard_enable_logging,
    )
    return solution_data


def _run_picard_iterations_for_dpp(
    forms_macro: Tuple[fd.Form, fd.Form],
    forms_micro: Tuple[fd.Form, fd.Form],
    bcs_macro: List[fd.DirichletBC],
    bcs_micro: List[fd.DirichletBC],
    solver_macro: fd.LinearVariationalSolver,
    solver_micro: fd.LinearVariationalSolver,
    macro_pressure_field: fd.Function,
    micro_pressure_field: fd.Function,
    macro_pressure_initial_solution: fd.Function,
    micro_pressure_initial_solution: fd.Function,
    picard_damping_parameter: fd.Constant = fd.Constant(1),
    picard_rtol: float = 1e-5,  # To match PETSc default
    picard_atol: float = 1e-12,  # To match PETSc default (which is 0.0, actually)
    picard_max_iteration_number: int = 10000,
    enable_logging: bool = True,
) -> Solution:
    """
    TODO.

    :param solver_macro: _description_
    :type solver_macro: fd.LinearVariationalSolver
    :param solver_micro: _description_
    :type solver_micro: fd.LinearVariationalSolver
    :param macro_pressure_field: _description_
    :type macro_pressure_field: fd.Function
    :param micro_pressure_field: _description_
    :type micro_pressure_field: fd.Function
    :param macro_pressure_initial_solution: _description_
    :type macro_pressure_initial_solution: fd.Function
    :param micro_pressure_initial_solution: _description_
    :type micro_pressure_initial_solution: fd.Function
    :param picard_damping_parameter: _description_, defaults to fd.Constant(1)
    :type picard_damping_parameter: fd.Constant, optional
    :param picard_rel_error: _description_, defaults to 1e-12
    :type picard_rel_error: float, optional
    :param picard_max_iteration_number: _description_, defaults to 10000
    :type picard_max_iteration_number: int, optional
    :param enable_logging: _description_, defaults to True
    :type enable_logging: bool, optional
    :raises RuntimeError: _description_
    :return: _description_
    :rtype: Tuple[fd.Function, fd.Function]
    """
    solution_macro = macro_pressure_field
    solution_micro = micro_pressure_field
    p1_old = macro_pressure_initial_solution
    p2_old = micro_pressure_initial_solution

    # Retrieve the bilinear/linear forms
    ## Macro
    a_macro_form, L_macro_form = forms_macro

    ## Micro
    a_micro_form, L_micro_form = forms_micro

    # Assemble PETSc matrices & RHS once
    A_macro = fd.assemble(a_macro_form, bcs=bcs_macro, mat_type="aij").M.handle
    b_macro = fd.assemble(L_macro_form, bcs=bcs_macro)
    bvec_macro = b_macro.vector()

    A_micro = fd.assemble(a_micro_form, bcs=bcs_micro, mat_type="aij").M.handle
    b_micro = fd.assemble(L_micro_form, bcs=bcs_micro)
    bvec_micro = b_micro.vector()

    # Initial guess = zero
    solution_macro.vector().set(0.0)
    solution_micro.vector().set(0.0)

    # Compute initial residual norm r0 = || [A x0 – b] ||
    r0_m = bvec_macro.duplicate()
    A_macro.mult(solution_macro.vector(), r0_m)
    r0_m.axpy(-1.0, bvec_macro)
    r0_μ = bvec_micro.duplicate()
    A_micro.mult(solution_micro.vector(), r0_μ)
    r0_μ.axpy(-1.0, bvec_micro)
    r0 = np.hypot(r0_m.norm(PETSc._PETSc.NORM_2), r0_μ.norm(PETSc._PETSc.NORM_2))

    rtol, maxit = picard_rtol, picard_max_iteration_number
    damping_parameter = picard_damping_parameter
    solution_residual_error = r0
    solution_iteration_number = 0

    for i in range(1, picard_max_iteration_number + 1):
        # 1) Solve for the *new* macro pressure
        solver_macro.solve()
        # 2) Update p1_old ← (1–α)*p1_old + α*p1_new
        p1_old.assign(
            (1 - picard_damping_parameter) * p1_old + picard_damping_parameter * solution_macro
        )

        # 3) Solve for the *new* micro pressure (with updated p1_old in the forms)
        solver_micro.solve()
        # 4) Update p2_old ← (1–α)*p2_old + α*p2_new
        p2_old.assign(
            (1 - picard_damping_parameter) * p2_old + picard_damping_parameter * solution_micro
        )

        # 5) Compute algebraic residuals using the *current* p1_old/p2_old
        xvec_macro = solution_macro.vector()
        xvec_micro = solution_micro.vector()

        r_m = bvec_macro.copy()
        A_macro.mult(xvec_macro, r_m)
        r_m.axpy(-1.0, bvec_macro)
        norm_m = r_m.norm(PETSc._PETSc.NORM_2)

        r_μ = bvec_micro.copy()
        A_micro.mult(xvec_micro, r_μ)
        r_μ.axpy(-1.0, bvec_micro)
        norm_μ = r_μ.norm(PETSc._PETSc.NORM_2)

        r = np.hypot(norm_m, norm_μ)
        if enable_logging:
            logger.info(f"Picard it {i}, ||res|| = {r:e}")

        if r <= picard_atol + picard_rtol * r0:
            return Solution((solution_macro, solution_micro), i, r)

    # for i in range(1, maxit + 1):
    #     # Macro sub-system
    #     solver_macro.solve()  # macro with micro fixed
    #     p1_old.assign((1 - damping_parameter) * p1_old + damping_parameter * solution_macro)

    #     # Micro sub-system
    #     solver_micro.solve()  # micro with macro fixed
    #     p2_old.assign((1 - damping_parameter) * p2_old + damping_parameter * solution_micro)

    #     # Update errors and residuals
    #     R_macro_iteration = fd.assemble(Rf_macro)
    #     R_micro_iteration = fd.assemble(Rf_micro)
    #     norm_residual  = np.sqrt(R_macro_iteration.dat.norm()**2 + R_micro_iteration.dat.norm()**2)

    #     # Update tracking vars
    #     solution_iteration_number = i
    #     solution_residual_error = norm_residual

    #     if enable_logging:
    #         logger.info(
    #             f"Picard iteration: {i}; residual = {norm_residual}"
    #         )

    #     # Convergence check
    #     if norm_residual <= picard_atol + picard_rtol * r0:
    #         logger.info(f"Converged in {i} Picard steps")
    #         break

    #     # Update fully staggered pressures according to Picard's fixed-point method
    #     # p1_old.assign((1 - damping_parameter) * p1_old + damping_parameter * solution_macro)
    #     # p2_old.assign((1 - damping_parameter) * p2_old + damping_parameter * solution_micro)
    # else:
    #     logger.error("Picard did not converge in", maxit)
    #     raise RuntimeError("DPP Picard iterations diverged.")

    # solution = (solution_macro, solution_micro)
    # solution_data = Solution(solution, solution_iteration_number, solution_residual_error)
    return solution_data
