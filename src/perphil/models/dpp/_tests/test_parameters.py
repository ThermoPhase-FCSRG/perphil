import pytest

try:
    import firedrake as fd
    from perphil.models.dpp.parameters import DPPParameters
except Exception:  # pragma: no cover - environment-dependent
    pytest.skip("Firedrake not available", allow_module_level=True)


def test_parameters_auto_constant_coercion_and_defaults():
    params = DPPParameters(k1=2.0, k2=None, beta=3.0, mu=4.0)
    assert isinstance(params.k1, fd.Constant)
    assert isinstance(params.k2, fd.Constant)
    assert isinstance(params.beta, fd.Constant)
    assert isinstance(params.mu, fd.Constant)
    # default k2 = k1/scale
    assert float(params.k2) == pytest.approx(float(params.k1) / params.scale_contrast)


def test_eta_computed_property():
    params = DPPParameters(k1=1.0, k2=0.01, beta=1.0, mu=1.0)
    # eta is a scalar UFL expression (built from Constants); don't require Constant type
    assert hasattr(params.eta, "ufl_shape") and params.eta.ufl_shape == ()
