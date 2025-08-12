"""
2D convergence study for the conforming primal (pressure-only) DPP formulation.

This script reuses:
- Mesh factory: perphil.mesh.builtin.create_mesh
- Spaces: perphil.forms.spaces.create_function_spaces
- Manufactured solutions: perphil.utils.manufactured_solutions.exact_expressions
- Solver: perphil.solvers.solver.solve_dpp
- Solver presets: perphil.solvers.parameters
- Error norms: perphil.utils.postprocessing.{l2_error,h1_seminorm_error}

It runs on a sequence of meshes, solves with one or more solver configurations,
and writes a CSV with L2 and H1-seminorm errors for p1 and p2 along with observed h.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import csv


try:
    import firedrake as fd
except Exception:  # pragma: no cover - keep import-light for environments without Firedrake
    raise SystemExit(
        "This experiment requires Firedrake. Please ensure 'firedrake' is installed and importable."
    )

from perphil.mesh.builtin import create_mesh
from perphil.forms.spaces import create_function_spaces
from perphil.models.dpp.parameters import DPPParameters
from perphil.solvers.solver import solve_dpp
from perphil.solvers.parameters import (
    LINEAR_SOLVER_PARAMS,
    PLAIN_GMRES_PARAMS,
    FIELDSPLIT_LU_PARAMS,
)
from perphil.utils.manufactured_solutions import exact_expressions
from perphil.utils.postprocessing import l2_error, h1_seminorm_error


@dataclass(frozen=True)
class SolverSpec:
    name: str
    params: Dict


def _build_bcs(
    W: fd.MixedFunctionSpace, p1_expr: fd.Expr, p2_expr: fd.Expr
) -> list[fd.DirichletBC]:
    """Dirichlet BCs on full boundary for both p1 and p2 using manufactured exact pressures."""
    return [
        fd.DirichletBC(W.sub(0), p1_expr, "on_boundary"),
        fd.DirichletBC(W.sub(1), p2_expr, "on_boundary"),
    ]


def _errors_for_solution(
    W: fd.MixedFunctionSpace,
    solution: fd.Function,
    p1_exact: fd.Expr,
    p2_exact: fd.Expr,
) -> Tuple[float, float, float, float]:
    """
    Compute L2 and H1-seminorm errors for (p1, p2) against exact expressions.
    """
    p1_h = solution.sub(0)
    p2_h = solution.sub(1)

    e1_l2 = l2_error(p1_h, p1_exact)
    e2_l2 = l2_error(p2_h, p2_exact)
    e1_h1s = h1_seminorm_error(p1_h, p1_exact)
    e2_h1s = h1_seminorm_error(p2_h, p2_exact)
    return float(e1_l2), float(e2_l2), float(e1_h1s), float(e2_h1s)


def _mesh_size_h_from_N(N: int) -> float:
    """For uniform UnitSquare meshes, use h = 1/N."""
    return 1.0 / float(N)


def run_one(N: int, solver: SolverSpec, quad: bool, degree: int, params: DPPParameters) -> dict:
    mesh = create_mesh(N, N, quadrilateral=quad)
    _, V = create_function_spaces(mesh, pressure_deg=degree, pressure_family="CG")
    W = fd.MixedFunctionSpace((V, V))

    # Manufactured exact pressures
    _, p1_expr, _, p2_expr = exact_expressions(mesh, params)

    bcs = _build_bcs(W, p1_expr, p2_expr)

    sol = solve_dpp(
        W, params, bcs=bcs, solver_parameters=solver.params, options_prefix=f"dpp_{solver.name}"
    )

    e1_l2, e2_l2, e1_h1s, e2_h1s = _errors_for_solution(W, sol.solution, p1_expr, p2_expr)

    return {
        "N": N,
        "h": _mesh_size_h_from_N(N),
        "degree": degree,
        "quad": int(quad),
        "solver": solver.name,
        "it": int(sol.iteration_number),
        "res": float(sol.residual_error),
        "e1_L2": e1_l2,
        "e2_L2": e2_l2,
        "e1_H1s": e1_h1s,
        "e2_H1s": e2_h1s,
    }


def _default_solvers(rtols: Iterable[float]) -> List[SolverSpec]:
    specs: List[SolverSpec] = [
        SolverSpec("mumps", LINEAR_SOLVER_PARAMS),
    ]
    for rtol in rtols:
        gmres = dict(PLAIN_GMRES_PARAMS)
        gmres["ksp_rtol"] = rtol
        specs.append(SolverSpec(f"gmres_rtol={rtol:g}", gmres))

        fs = dict(FIELDSPLIT_LU_PARAMS)
        # FIELDSPLIT_LU_PARAMS embeds nested dicts; preserve but allow outer KSP params
        fs["ksp_type"] = "gmres"
        fs["ksp_rtol"] = rtol
        fs["ksp_atol"] = 1.0e-12
        specs.append(SolverSpec(f"fs-lu_gmres_rtol={rtol:g}", fs))
    return specs


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="2D convergence experiment for conforming DPP (two pressures)"
    )
    ap.add_argument(
        "--Ns",
        type=int,
        nargs="+",
        default=[16, 32, 64],
        help="Mesh sizes N for UnitSquareMesh (NxN)",
    )
    ap.add_argument("--degree", type=int, default=1, help="CG polynomial degree for pressures")
    ap.add_argument("--tri", action="store_true", help="Use triangles instead of quads")
    ap.add_argument(
        "--rtols",
        type=float,
        nargs="+",
        default=[1e-8, 1e-10],
        help="KSP relative tolerances to test for GMRES-based solvers",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("notebooks/results-conforming-2d/convergence.csv")
    )
    args = ap.parse_args(argv)

    quad = not args.tri

    params = DPPParameters()  # defaults are Constants inside

    solvers = _default_solvers(args.rtols)
    rows: List[dict] = []

    for N in args.Ns:
        for spec in solvers:
            row = run_one(N=N, solver=spec, quad=quad, degree=args.degree, params=params)
            rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
