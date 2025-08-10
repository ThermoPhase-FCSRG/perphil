import pytest

try:
    import firedrake as fd
    from perphil.mesh.builtin import create_mesh
except Exception:  # pragma: no cover
    pytest.skip("Firedrake not available", allow_module_level=True)


def test_create_mesh_triangle():
    mesh = create_mesh(4, 3, quadrilateral=False)
    # fd.Mesh is a function; use MeshGeometry type for isinstance checks
    assert isinstance(mesh, fd.MeshGeometry)
    assert mesh.topological_dimension() == 2


def test_create_mesh_quadrilateral():
    mesh = create_mesh(2, 2, quadrilateral=True)
    assert isinstance(mesh, fd.MeshGeometry)
    assert mesh.coordinates.function_space().ufl_element().family() in {"Q", "DQ", "RTCF", "QCF"}
