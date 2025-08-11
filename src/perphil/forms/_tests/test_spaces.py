import pytest

try:
    import firedrake as fd  # noqa: F401
    from perphil.mesh.builtin import create_mesh
    from perphil.forms.spaces import create_function_spaces
except Exception:  # pragma: no cover
    pytest.skip("Firedrake not available", allow_module_level=True)


def test_create_function_spaces():
    mesh = create_mesh(2, 2, quadrilateral=True)
    U, V = create_function_spaces(mesh)
    assert U.mesh() is mesh
    assert V.mesh() is mesh
    # Firedrake returns 'Lagrange' on triangles and 'Q' on quads; accept either
    assert U.ufl_element().family() in {"Lagrange", "Q"}
    assert V.ufl_element().family() in {"Lagrange", "Q"}
