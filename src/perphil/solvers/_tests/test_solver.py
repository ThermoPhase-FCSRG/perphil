import pytest

try:
    import firedrake as fd
    from perphil.forms.spaces import create_function_spaces
    from perphil.mesh.builtin import create_mesh
    from perphil.solvers.solver import solve_dpp, Solution, solve_dpp_nonlinear
    from perphil.models.dpp.parameters import DPPParameters
except Exception:  # pragma: no cover
    pytest.skip("Firedrake not available", allow_module_level=True)


def _mixed_space(nx=2, ny=2, quad=True):
    mesh = create_mesh(nx, ny, quadrilateral=quad)
    _, V = create_function_spaces(mesh)
    W = fd.MixedFunctionSpace((V, V))
    bcs = [
        fd.DirichletBC(W.sub(0), fd.Constant(0.0), "on_boundary"),
        fd.DirichletBC(W.sub(1), fd.Constant(0.0), "on_boundary"),
    ]
    return W, bcs


def test_solve_dpp_linear_returns_solution():
    W, bcs = _mixed_space()
    params = DPPParameters()
    sol = solve_dpp(W, params, bcs=bcs)
    assert isinstance(sol, Solution)
    assert (
        hasattr(sol, "solution")
        and hasattr(sol, "iteration_number")
        and hasattr(sol, "residual_error")
    )
    assert sol.iteration_number >= 0


def test_solve_dpp_raises_on_non_mixed_space():
    mesh = create_mesh(2, 2, quadrilateral=True)
    _, V = create_function_spaces(mesh)
    params = DPPParameters()
    with pytest.raises(ValueError):
        solve_dpp(V, params, bcs=[])  # type: ignore[arg-type]


def test_solve_dpp_nonlinear_returns_solution():
    W, bcs = _mixed_space()
    params = DPPParameters()
    sol = solve_dpp_nonlinear(W, params, bcs=bcs)
    assert isinstance(sol, Solution)
    assert sol.iteration_number >= 0
