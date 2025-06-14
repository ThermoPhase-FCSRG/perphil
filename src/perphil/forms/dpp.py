from typing import Tuple
import firedrake as fd

from perphil.models.dpp.parameters import DPPParameters


def _calculate_mass_transfer_term(
    p_macro: fd.Function,
    p_micro: fd.Function,
    beta: fd.Constant,
    mu: fd.Constant,
) -> fd.Function:
    """
    TODO.

    :param p_macro: _description_
    :type p_macro: fd.Function
    :param p_micro: _description_
    :type p_micro: fd.Function
    :param beta: _description_
    :type beta: fd.Constant
    :param mu: _description_
    :type mu: fd.Constant
    :return: _description_
    :rtype: fd.Function
    """
    xi = -beta / mu * (p_macro - p_micro)
    return xi


def _macro_scale_form(
    trial_and_test_functions_macro: Tuple[fd.Function, fd.Function],
    pressure_micro: fd.Function,
    k1: fd.Constant,
    mu: fd.Constant,
    beta: fd.Constant,
) -> Tuple[fd.Form, fd.Form]:
    """
    TODO.

    :param trial_and_test_functions_macro: _description_
    :type trial_and_test_functions_macro: Tuple[fd.Function, fd.Function]
    :param pressure_micro: _description_
    :type pressure_micro: fd.Function
    :param k1: _description_
    :type k1: fd.Constant
    :param mu: _description_
    :type mu: fd.Constant
    :param beta: _description_
    :type beta: fd.Constant
    :return: _description_
    :rtype: Tuple[fd.Form, fd.Form]
    """
    p1, q1 = trial_and_test_functions_macro
    p2 = pressure_micro
    xi = _calculate_mass_transfer_term(p1, p2, beta, mu)
    a_macro = (k1 / mu) * fd.inner(fd.grad(p1), fd.grad(q1)) * fd.dx - xi * q1 * fd.dx
    L_macro = fd.Constant(0.0) * q1 * fd.dx

    return a_macro, L_macro


def _micro_scale_form(
    trial_and_test_functions_micro: Tuple[fd.Function, fd.Function],
    pressure_macro: fd.Function,
    k2: fd.Constant,
    mu: fd.Constant,
    beta: fd.Constant,
) -> Tuple[fd.Form, fd.Form]:
    """
    TODO.

    :param trial_and_test_functions_micro: _description_
    :type trial_and_test_functions_micro: Tuple[fd.Function, fd.Function]
    :param pressure_macro: _description_
    :type pressure_macro: fd.Function
    :param k2: _description_
    :type k2: fd.Constant
    :param mu: _description_
    :type mu: fd.Constant
    :param beta: _description_
    :type beta: fd.Constant
    :return: _description_
    :rtype: Tuple[fd.Form, fd.Form]
    """
    p1 = pressure_macro
    p2, q2 = trial_and_test_functions_micro
    xi = _calculate_mass_transfer_term(p1, p2, beta, mu)
    a_micro = (k2 / mu) * fd.inner(fd.grad(p2), fd.grad(q2)) * fd.dx + xi * q2 * fd.dx
    L_micro = fd.Constant(0.0) * q2 * fd.dx

    return a_micro, L_micro


def dpp_form(W: fd.FunctionSpace, model_params: DPPParameters) -> Tuple[fd.Form, fd.Form]:
    """
    Build the bilinear and linear forms for the double-porosity/permeability system.

    :param W:
        A MixedFunctionSpace for (p1, p2) pressures.

    :param model_params:
        DPPParameters container with model constants.

    :return:
        Tuple of (a, L), where:
        - a is the bilinear form for the coupled system.
        - L is the linear form (currently zero forcing).

    :raises ValueError:
        If W is not a MixedFunctionSpace.
    """
    if not hasattr(W, "num_sub_spaces") or W.num_sub_spaces() != 2:
        raise ValueError(f"Expected a 2-field MixedFunctionSpace, got {type(W)}")

    p1, p2 = fd.TrialFunctions(W)
    q1, q2 = fd.TestFunctions(W)

    k1 = model_params.k1
    k2 = model_params.k2
    assert isinstance(k2, fd.Constant)
    beta = model_params.beta
    mu = model_params.mu

    a_macro, L_macro = _macro_scale_form((p1, q1), p2, k1, mu, beta)

    a_micro, L_micro = _micro_scale_form((p2, q2), p1, k2, mu, beta)

    a = a_macro + a_micro
    L = L_macro + L_micro

    return a, L


def dpp_delayed_form(
    macro_function_space: fd.FunctionSpace,
    micro_function_space: fd.FunctionSpace,
    model_params: DPPParameters,
    macro_pressure_initial_values: fd.Function,
    micro_pressure_initial_values: fd.Function,
) -> Tuple[Tuple[fd.Form, fd.Form], Tuple[fd.Form, fd.Form]]:
    """
    Build the bilinear and linear forms for the double-porosity/permeability system with delayed
    pressure fields.

    This function is to be used only with the Picard approach.

    :param macro_function_space:
        A FunctionSpace for macro pressures.

    :param macro_function_space:
        A FunctionSpace for micro pressures.

    :param model_params:
        DPPParameters container with model constants.

    :param macro_pressure_initial_values:
        A macro pressure field used as initial guess for the Picard iterations.

    :param micro_pressure_initial_values:
        A micro pressure field used as initial guess for the Picard iterations.

    :return:
        Tuple of ((a_macro, L_macro), (a_micro, L_micro)), where:

        - a_macro is the bilinear form for the macro system.
        - a_micro is the bilinear form for the micro system.
        - L_macro is the linear form associated with the macro scale.
        - L_micro is the linear form associated with the micro scale.

    :raises ValueError:
        If W is not a MixedFunctionSpace.
    """
    V_macro = macro_function_space
    V_micro = micro_function_space

    # Trial and test functions
    p1 = fd.TrialFunction(V_macro)
    p2 = fd.TrialFunction(V_micro)
    q1 = fd.TestFunction(V_macro)
    q2 = fd.TestFunction(V_micro)

    # Model parameters
    k1 = model_params.k1
    k2 = model_params.k2
    assert isinstance(k2, fd.Constant)
    beta = model_params.beta
    mu = model_params.mu

    # Initial delayed pressures (assumed zero)
    p1_old = macro_pressure_initial_values
    p2_old = micro_pressure_initial_values

    ## Macro terms
    forms_macro = _macro_scale_form((p1, q1), p2_old, k1, mu, beta)
    a_macro = fd.lhs(forms_macro[0])
    L_macro = fd.rhs(forms_macro[0]) + forms_macro[1]

    ## Micro terms
    forms_micro = _micro_scale_form((p2, q2), p1_old, k2, mu, beta)
    a_micro = fd.lhs(forms_micro[0])
    L_micro = fd.rhs(forms_micro[0]) + forms_micro[1]

    return (a_macro, L_macro), (a_micro, L_micro)


def dpp_splitted_form(
    W: fd.FunctionSpace, model_params: DPPParameters
) -> Tuple[fd.Form, fd.Function]:
    """
    Build the nonlinear residual form for Picard (fixed-point) iterations.

    :param W:
        A MixedFunctionSpace for (p1, p2) unknowns.

    :param model_params:
        DPPParameters container with model constants.

    :return:
        Tuple of (F, fields), where:
        - F is the UFL residual form.
        - fields is the Function holding (p1, p2).

    :raises ValueError:
        If W is not a MixedFunctionSpace.
    """
    if not hasattr(W, "num_sub_spaces") or W.num_sub_spaces() != 2:
        raise ValueError(f"Expected a 2-field MixedFunctionSpace, got {type(W)}")

    fields = fd.Function(W)
    p1, p2 = fd.split(fields)
    q1, q2 = fd.TestFunctions(W)

    k1 = model_params.k1
    k2 = model_params.k2
    assert isinstance(k2, fd.Constant)
    beta = model_params.beta
    mu = model_params.mu

    xi = -beta / mu * (p1 - p2)

    F1 = (k1 / mu) * fd.inner(fd.grad(p1), fd.grad(q1)) * fd.dx - xi * q1 * fd.dx
    F2 = (k2 / mu) * fd.inner(fd.grad(p2), fd.grad(q2)) * fd.dx + xi * q2 * fd.dx

    F = F1 + F2
    return F, fields
