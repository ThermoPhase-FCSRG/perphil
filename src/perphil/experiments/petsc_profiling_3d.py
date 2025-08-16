"""
3D PETSc profiling for the DPP model using manufactured 3D solutions (paper Eq. 6.3).

This mirrors perphil.experiments.petsc_profiling but swaps the mesh to 3D and
uses exact_expressions_3d for Dirichlet boundary conditions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import json
import pandas as pd
import firedrake as fd
from firedrake.petsc import PETSc

from perphil.experiments.iterative_bench import Approach, params_for
from perphil.models.dpp.parameters import DPPParameters
from perphil.solvers.solver import solve_dpp, solve_dpp_nonlinear
from perphil.utils.manufactured_solutions import exact_expressions_3d

# Reuse the robust profiling utilities from the 2D module
from perphil.experiments import petsc_profiling as prof2d


# Prefer explicit execution for timing stability
fd.parameters["pyop2_options"]["lazy_evaluation"] = False


def _build_mesh_3d(nx: int) -> fd.Mesh:
    return fd.UnitCubeMesh(nx, nx, nx)


def _build_spaces_3d(mesh: fd.Mesh) -> fd.MixedFunctionSpace:
    V = fd.FunctionSpace(mesh, "CG", 1)
    return V * V


def _default_model_params() -> DPPParameters:
    return DPPParameters(k1=1.0, k2=1.0 / 1e2, beta=1.0, mu=1.0)


def run_perf_once_3d(
    nx: int,
    approach: Approach,
    eager: bool = True,
    logical_events: Optional[List[str]] = None,
    repeats: int = 5,
    backend: str = "auto",
) -> Dict[str, Any]:
    """
    3D analog of prof2d.run_perf_once, using UnitCube meshes and 3D MMS BCs.
    Returns a flattened dict ready for DataFrame assembly.
    """
    prof2d.ensure_petsc_logging()

    mesh = _build_mesh_3d(nx)
    W = _build_spaces_3d(mesh)
    comm = mesh.comm

    params = _default_model_params()

    # Manufactured Dirichlet BCs on boundary from 3D MMS
    _u1e, p1e, _u2e, p2e = exact_expressions_3d(mesh, params)
    bcs = [
        fd.DirichletBC(W.sub(0), p1e, "on_boundary"),
        fd.DirichletBC(W.sub(1), p2e, "on_boundary"),
    ]

    logical_events = list(dict.fromkeys((logical_events or []) + prof2d.DEFAULT_LOGICAL_EVENTS))

    # Warmup
    if eager:
        if approach == Approach.PICARD_MUMPS:
            sol_warm = solve_dpp_nonlinear(
                W, params, bcs=bcs, solver_parameters={**params_for(approach)}
            )
        else:
            sol_warm = solve_dpp(W, params, bcs=bcs, solver_parameters={**params_for(approach)})
        prof2d._enable_convergence_history_if_possible(prof2d._extract_solution_handle(sol_warm))

    def run_once() -> None:
        if approach == Approach.PICARD_MUMPS:
            solve_dpp_nonlinear(W, params, bcs=bcs, solver_parameters={**params_for(approach)})
        else:
            solve_dpp(W, params, bcs=bcs, solver_parameters={**params_for(approach)})

    # Record RSS before profiling this case (per-rank)
    rss_before_kb = prof2d._get_rss_kb()

    # Select backend(s)
    backends = [backend] if backend != "auto" else ["json", "ascii", "events", "stage", "wall"]

    wall_total = 0.0
    for backend_name in backends:
        try:
            if backend_name == "json":
                event_times, event_flops, wall_time = prof2d._profile_with_log_view_json(
                    comm, run_once, logical_events, repeats
                )
            elif backend_name == "ascii":
                event_times, event_flops, wall_time = prof2d._profile_with_log_view_ascii(
                    comm, run_once, logical_events, repeats
                )
            elif backend_name == "events":
                event_times, event_flops, wall_time = prof2d._profile_with_events_api(
                    comm, run_once, logical_events, repeats
                )
            elif backend_name == "stage":
                event_times, event_flops, wall_time = prof2d._profile_with_stage_api(
                    comm, run_once, logical_events, repeats
                )
            else:
                # wall fallback
                import time as _t

                comm.barrier()
                t0 = _t.perf_counter()
                for _ in range(max(1, repeats)):
                    run_once()
                wall_time = _t.perf_counter() - t0
                event_times = {e: 0.0 for e in logical_events}
                event_times["KSPSolve"] = wall_time
                event_flops = {e: 0.0 for e in logical_events}
            if (sum(event_times.values()) > 0.0) or backend_name in ("wall", "events"):
                backend_used = backend_name
                wall_total = wall_time
                break
        except Exception:
            continue
    else:
        # If all backends failed
        import time as _t

        comm.barrier()
        t0 = _t.perf_counter()
        for _ in range(max(1, repeats)):
            run_once()
        wall_time = _t.perf_counter() - t0
        event_times = {e: 0.0 for e in logical_events}
        event_times["KSPSolve"] = wall_time
        event_flops = {e: 0.0 for e in logical_events}
        backend_used = "wall"
        wall_total = wall_time

    # Final solve to collect iterations/residual and build operators for memory stats
    if approach == Approach.PICARD_MUMPS:
        sol = solve_dpp_nonlinear(W, params, bcs=bcs, solver_parameters={**params_for(approach)})
    else:
        sol = solve_dpp(W, params, bcs=bcs, solver_parameters={**params_for(approach)})

    # Metadata and counters
    num_cells = comm.allreduce(mesh.cell_set.size, op=prof2d.MPI.SUM)
    dofs = W.dim()
    solver_handle = prof2d._extract_solution_handle(sol)
    ksp_iters = prof2d._extract_ksp_iters_if_possible(solver_handle)
    iterations = ksp_iters if ksp_iters is not None else getattr(sol, "iteration_number", None)
    residual = float(getattr(sol, "residual_error", float("nan")))

    # Memory
    rss_after_kb = prof2d._get_rss_kb()
    rss_peak_kb = float(comm.allreduce(rss_after_kb, op=prof2d.MPI.MAX))
    rss_delta_kb = float(comm.allreduce(max(0, rss_after_kb - rss_before_kb), op=prof2d.MPI.MAX))
    mat_mem = prof2d._collect_matrix_memory(solver_handle)
    memory: Dict[str, float] = {"rss_peak_kb": rss_peak_kb, "rss_delta_kb": rss_delta_kb, **mat_mem}

    meta: Dict[str, Any] = {
        "petsc_version": PETSc.Sys.getVersion(),
        "firedrake_version": getattr(fd, "__version__", None),
        "backend": backend_used,
        "repeats": repeats,
    }

    # Flatten like prof2d.PerfResult.to_dict()
    row: Dict[str, Any] = {
        "approach": approach.value,
        "nx": nx,
        "ny": nx,
        "dofs": int(dofs),
        "num_cells": int(num_cells),
        "iterations": int(iterations) if iterations is not None else None,
        "residual": residual,
        "time_total": float(wall_total / max(1, repeats)),
        "time_total_repeats": float(wall_total),
    }
    # times/flops
    for k, v in (event_times or {}).items():
        row[f"time_{k}"] = float(v)
    for k, v in (event_flops or {}).items():
        row[f"flops_{k}"] = float(v)
        t = float(event_times.get(k, 0.0))
        row[f"mflops_{k}"] = (float(v) / t / 1e6) if t > 0.0 else 0.0
    row["flops_total"] = float(sum((event_flops or {}).values()))
    # memory
    for k, v in memory.items():
        row[f"mem_{k}"] = float(v)
    # metadata (optional): we can add to row if desired or keep separate
    row["backend"] = meta.get("backend")
    row["repeats"] = meta.get("repeats")
    return row


def run_perf_sweep_3d(
    mesh_sizes: List[int],
    approaches: List[Approach],
    logical_events: Optional[List[str]] = None,
    eager: bool = True,
    repeats: int = 5,
    backend: str = "auto",
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for nx in mesh_sizes:
        for ap in approaches:
            res = run_perf_once_3d(
                nx,
                ap,
                eager=eager,
                logical_events=logical_events,
                repeats=repeats,
                backend=backend,
            )
            rows.append(res)
            PETSc.Sys.Print(
                f"[perf3d] nx={nx} {ap.value}: iters={res.get('iterations')}, "
                f"time_total={res.get('time_total', 0.0):.3e}s, "
                f"KSPSolve={res.get('time_KSPSolve', 0.0):.3e}s, "
                f"flops_total={res.get('flops_total', 0.0):.3e} "
                f"(backend={res.get('backend')}, repeats={res.get('repeats')})"
            )
    return pd.DataFrame(rows)


def save_perf_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def save_perf_json(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)
