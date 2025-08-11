import pytest

try:
    import firedrake as fd
    from perphil.mesh.builtin import create_mesh
    from perphil.utils.manufactured_solutions import exact_expressions, interpolate_exact
    from perphil.models.dpp.parameters import DPPParameters
except Exception:  # pragma: no cover
    pytest.skip("Firedrake not available", allow_module_level=True)


def test_exact_expressions_shapes():
    mesh = create_mesh(2, 2, quadrilateral=False)
    params = DPPParameters()
    u1e, p1e, u2e, p2e = exact_expressions(mesh, params)
    assert u1e.ufl_shape == (2,)
    assert u2e.ufl_shape == (2,)


def test_interpolate_exact_creates_functions():
    from perphil.forms.spaces import create_function_spaces

    mesh = create_mesh(2, 2, quadrilateral=True)
    U, V = create_function_spaces(mesh)
    params = DPPParameters()
    u1, p1, u2, p2 = interpolate_exact(mesh, U, V, params)
    assert isinstance(u1, fd.Function)
    assert isinstance(u2, fd.Function)
    assert isinstance(p1, fd.Function)
    assert isinstance(p2, fd.Function)
