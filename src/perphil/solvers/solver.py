from typing import List, Dict, Tuple
import firedrake as fd
import logging

from perphil.models.dpp.parameters import DPPParameters
from perphil.forms.dpp import dpp_delayed_form, dpp_form, dpp_splitted_form

logger = logging.getLogger(__name__)


def solve_dpp(
    W: fd.FunctionSpace,
    model_params: DPPParameters,
    bcs: List[fd.DirichletBC],
    solver_parameters: Dict = {},
    options_prefix: str = "dpp",
) -> fd.Function:
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
    return solution


def solve_dpp_nonlinear(
    W: fd.FunctionSpace,
    model_params: DPPParameters,
    bcs: List[fd.DirichletBC],
    solver_parameters: Dict = {},
    options_prefix: str = "dpp_nonlinear",
) -> fd.Function:
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
    return fields


def solve_dpp_picard(
    W: fd.FunctionSpace,
    model_params: DPPParameters,
    bcs_macro: List[fd.DirichletBC],
    bcs_micro: List[fd.DirichletBC],
    macro_solver_parameters: Dict = {},
    micro_solver_parameters: Dict = {},
    picard_damping_parameter: fd.Constant = fd.Constant(1),
    picard_rel_error: float = 1e-12,
    picard_max_iteration_number: int = 10000,
    picard_enable_logging: bool = True,
) -> Tuple[fd.Function, fd.Function]:
    """
    TODO.

    :param W: _description_
    :type W: fd.FunctionSpace
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
    # Initial guess for pressure fields to be used in Picard iterations
    p1_old = fd.interpolate(fd.Constant(0), W.sub(0))
    p2_old = fd.interpolate(fd.Constant(0), W.sub(1))

    # Retrieve variation forms by scale
    forms_macro, forms_micro = dpp_delayed_form(W, model_params, p1_old, p2_old)
    a_macro, L_macro = forms_macro
    a_micro, L_micro = forms_micro

    ## Macro
    solution_macro = fd.Function(W.sub(0))
    problem_macro = fd.LinearVariationalProblem(
        a_macro, L_macro, solution_macro, bcs=[bcs_macro], constant_jacobian=True
    )
    solver_macro = fd.LinearVariationalSolver(
        problem_macro, solver_parameters=macro_solver_parameters
    )

    ## Micro
    solution_micro = fd.Function(W.sub(1))
    problem_micro = fd.LinearVariationalProblem(
        a_micro, L_micro, solution_micro, bcs=[bcs_micro], constant_jacobian=True
    )
    solver_micro = fd.LinearVariationalSolver(
        problem_micro, solver_parameters=micro_solver_parameters
    )

    ## Picard loop
    solution_macro, solution_micro = _run_picard_iterations_for_dpp(
        solver_macro,
        solver_micro,
        solution_macro,
        solution_micro,
        p1_old,
        p2_old,
        picard_damping_parameter=picard_damping_parameter,
        picard_rel_error=picard_rel_error,
        picard_max_iteration_number=picard_max_iteration_number,
        enable_logging=picard_enable_logging,
    )
    return solution_macro, solution_micro


def _run_picard_iterations_for_dpp(
    solver_macro: fd.LinearVariationalSolver,
    solver_micro: fd.LinearVariationalSolver,
    macro_pressure_field: fd.Function,
    micro_pressure_field: fd.Function,
    macro_pressure_initial_solution: fd.Function,
    micro_pressure_initial_solution: fd.Function,
    picard_damping_parameter: fd.Constant = fd.Constant(1),
    picard_rel_error: float = 1e-12,
    picard_max_iteration_number: int = 10000,
    enable_logging: bool = True,
) -> Tuple[fd.Function, fd.Function]:
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
    rtol, maxit = picard_rel_error, picard_max_iteration_number
    damping_parameter = picard_damping_parameter
    for i in range(1, maxit + 1):
        # Macro sub-system
        solver_macro.solve()  # macro with micro fixed
        p1_old_k = p1_old.copy(deepcopy=True)
        p1_old.assign((1 - damping_parameter) * p1_old + damping_parameter * solution_macro)

        # Micro sub-system
        solver_micro.solve()  # micro with macro fixed
        p2_old_k = p2_old.copy(deepcopy=True)
        p2_old.assign((1 - damping_parameter) * p2_old + damping_parameter * solution_micro)

        # Errors and residuals
        p1_residual = fd.norm(solution_macro - p1_old_k)
        p2_residual = fd.norm(solution_micro - p2_old_k)
        p1_rel_error = p1_residual / fd.norm(solution_macro)
        p2_rel_error = p2_residual / fd.norm(solution_micro)
        if enable_logging:
            logger.info(
                f"Picard iteration: {i}; p1 rel error = {p1_rel_error}; p2 rel error = {p2_rel_error}"
            )

        # Convergence check
        if max(p1_rel_error, p2_rel_error) < rtol:
            logger.info(f"Converged in {i} Picard steps")
            break

        # Update fully staggered pressures according to Picard's fixed-point method
        # p1_old.assign((1 - damping_parameter) * p1_old + damping_parameter * solution_macro)
        # p2_old.assign((1 - damping_parameter) * p2_old + damping_parameter * solution_micro)
    else:
        logger.error("Picard did not converge in", maxit)
        raise RuntimeError("DPP Picard iterations diverged.")
    return solution_macro, solution_micro
