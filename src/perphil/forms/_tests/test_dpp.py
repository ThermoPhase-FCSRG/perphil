import pytest

try:
    import firedrake as fd
    from perphil.forms.dpp import dpp_form, dpp_splitted_form
    from perphil.forms.spaces import create_function_spaces
    from perphil.mesh.builtin import create_mesh
    from perphil.models.dpp.parameters import DPPParameters
except Exception:  # pragma: no cover
    pytest.skip("Firedrake not available", allow_module_level=True)

    mesh = create_mesh(2, 2, quadrilateral=True)
    _, V = create_function_spaces(mesh)
    W = fd.MixedFunctionSpace((V, V))

    params = DPPParameters(k1=1.0, k2=0.01, beta=1.0, mu=1.0)
    a, L = dpp_form(W, params)

    assert isinstance(a, fd.Form)
    assert isinstance(L, fd.Form)


def test_dpp_form_raises_on_non_mixed_space():
    mesh = create_mesh(2, 2, quadrilateral=True)
    _, V = create_function_spaces(mesh)
    params = DPPParameters()

    with pytest.raises(ValueError):
        dpp_form(V, params)  # type: ignore[arg-type]


def test_dpp_splitted_form_builds_residual_and_fields():
    mesh = create_mesh(2, 2, quadrilateral=False)
    _, V = create_function_spaces(mesh)
    W = fd.MixedFunctionSpace((V, V))
    params = DPPParameters()

    F, fields = dpp_splitted_form(W, params)
    assert isinstance(F, fd.Form)
    assert isinstance(fields, fd.Function)
