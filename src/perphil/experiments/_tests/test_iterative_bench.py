import pytest

try:
    from perphil.experiments.iterative_bench import (
        Approach,
        build_mesh,
        build_spaces,
        default_bcs,
        default_model_params,
        solve_on_mesh,
    )
except Exception:  # pragma: no cover
    pytest.skip("Firedrake not available", allow_module_level=True)


def test_build_mesh_and_spaces():
    mesh = build_mesh(2, 2, quadrilateral=True)
    _, _, W = build_spaces(mesh)
    bcs = default_bcs(W)
    assert len(bcs) == 2
    assert W.num_sub_spaces() == 2


def test_solve_on_mesh_returns_result():
    mesh = build_mesh(2, 2, quadrilateral=False)
    _, _, W = build_spaces(mesh)
    res = solve_on_mesh(W, Approach.PLAIN_GMRES, params=default_model_params(), bcs=default_bcs(W))
    assert res.iteration_number >= -1
    assert res.approach == Approach.PLAIN_GMRES
