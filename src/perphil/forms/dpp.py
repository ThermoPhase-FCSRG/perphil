from typing import Tuple
import firedrake as fd

from perphil.models.dpp.parameters import DPPParameters


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
    if not isinstance(W, fd.MixedFunctionSpace):
        raise ValueError(f"Expected a MixedFunctionSpace for W, got {type(W)}")

    p1, p2 = fd.TrialFunctions(W)
    q1, q2 = fd.TestFunctions(W)

    k1 = model_params.k1
    k2 = model_params.k2
    assert isinstance(k2, fd.Constant)
    beta = model_params.beta
    mu = model_params.mu

    xi = -beta / mu * (p1 - p2)

    a_macro = (k1 / mu) * fd.inner(fd.grad(p1), fd.grad(q1)) * fd.dx - xi * q1 * fd.dx
    L_macro = fd.Constant(0.0) * q1 * fd.dx

    a_micro = (k2 / mu) * fd.inner(fd.grad(p2), fd.grad(q2)) * fd.dx + xi * q2 * fd.dx
    L_micro = fd.Constant(0.0) * q2 * fd.dx

    a = a_macro + a_micro
    L = L_macro + L_micro

    return a, L


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
    if not isinstance(W, fd.MixedFunctionSpace):
        raise ValueError(f"Expected a MixedFunctionSpace for W, got {type(W)}")

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
