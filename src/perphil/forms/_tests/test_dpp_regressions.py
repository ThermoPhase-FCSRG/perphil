import pytest

try:
    import firedrake as fd
    from perphil.forms.dpp import dpp_form
    from perphil.forms.spaces import create_function_spaces
    from perphil.mesh.builtin import create_mesh
    from perphil.models.dpp.parameters import DPPParameters
except Exception:  # pragma: no cover
    pytest.skip("Firedrake not available", allow_module_level=True)


@pytest.mark.regression
def test_dpp_form_structure_regression(data_regression):
    mesh = create_mesh(2, 2, quadrilateral=True)
    _, V = create_function_spaces(mesh)
    W = fd.MixedFunctionSpace((V, V))
    a, L = dpp_form(W, DPPParameters())
    # Export simple metadata that should be stable across runs
    data = {
        # UFL forms don't expose .rank(); use number of arguments instead
        "rank": len(a.arguments()),
        "integrals": len(a.integrals()),
        "test_rank": len(W),
    }
    data_regression.check(data)
