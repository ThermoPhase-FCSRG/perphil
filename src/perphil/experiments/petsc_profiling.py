"""
Performance profiling utilities for PETSc-based DPP solvers.

This module provides context managers and functions to:
    - Enable and manage PETSc logging stages.
    - Collect and aggregate PETSc event times across MPI ranks.
    - Extract solver iteration counts and handles.
    - Run performance sweeps over mesh sizes and solver approaches.
    - Save profiling results to CSV or JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import json
import os

import numpy as np
import pandas as pd

import firedrake as fd
from firedrake.petsc import PETSc
from mpi4py import MPI

from perphil.experiments.iterative_bench import (
    Approach,
    build_mesh,
    build_spaces,
    default_bcs,
    default_model_params,
    solve_on_mesh,
)

# Prefer explicit execution for timing stability
fd.parameters["pyop2_options"]["lazy_evaluation"] = False

_PETSC_LOG_STARTED = False


def ensure_petsc_logging() -> None:
    """
    Start PETSc logging once per process.

    :return: None
    """
    global _PETSC_LOG_STARTED
    if not _PETSC_LOG_STARTED:
        PETSc.Log.begin()
        _PETSC_LOG_STARTED = True


class Stage:
    """
    Context manager for a PETSc logging stage.

    :param name: Name of the PETSc logging stage to push/pop.
    """

    def __init__(self, name: str):
        self.stage = PETSc.Log.Stage(name)

    def __enter__(self):
        self.stage.push()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stage.pop()


def _get_event_time(name: str) -> float:
    """
    Get wall-time for a PETSc event.

    :param name: Name of the PETSc event.
    :return: Wall-time in seconds for the event, or 0.0 if not recorded.
    """
    try:
        info = PETSc.Log.Event(name).getPerfInfo()
        return float(info.get("time", 0.0))
    except Exception:
        return 0.0


def collect_event_times(
    comm: MPI.Comm, extra_events: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Collect and average PETSc event times across MPI ranks.

    :param comm: MPI communicator.
    :param extra_events: Optional list of additional PETSc event names to include.
    :return: Dictionary mapping event names to average wall-times (seconds).
    """
    default_events = [
        "SNESSolve",
        "KSPSolve",
        "PCSetUp",
        "PCApply",
        "SNESJacobianEval",
        "SNESFunctionEval",
        "MatMult",
        "MatAssemblyBegin",
        "MatAssemblyEnd",
    ]
    events = list(dict.fromkeys(default_events + (extra_events or [])))

    local = {e: _get_event_time(e) for e in events}
    return {e: comm.allreduce(t, op=MPI.SUM) / comm.size for e, t in local.items()}


def _enable_convergence_history_if_possible(solver_obj: Any) -> None:
    """
    Enable convergence history on SNES/KSP solver handle if available.

    :param solver_obj: Solver object or handle potentially containing 'snes' or 'ksp'.
    :return: None
    """
    try:
        snes = getattr(solver_obj, "snes", None)
        if snes is not None and hasattr(snes, "setConvergenceHistory"):
            snes.setConvergenceHistory()
            try:
                snes.ksp.setConvergenceHistory()
            except Exception:
                pass
            return
        ksp = getattr(solver_obj, "ksp", None)
        if ksp is not None and hasattr(ksp, "setConvergenceHistory"):
            ksp.setConvergenceHistory()
    except Exception:
        pass


def _extract_ksp_iters_if_possible(solver_obj: Any) -> Optional[int]:
    """
    Extract iteration count from SNES/KSP handle if available.

    :param solver_obj: Solver handle with possible 'snes' or 'ksp' attribute.
    :return: Number of iterations, or None if not available.
    """
    try:
        snes = getattr(solver_obj, "snes", None)
        if snes is not None and hasattr(snes, "ksp"):
            return int(snes.ksp.getIterationNumber())
        ksp = getattr(solver_obj, "ksp", None)
        if ksp is not None and hasattr(ksp, "getIterationNumber"):
            return int(ksp.getIterationNumber())
    except Exception:
        return None
    return None


def _extract_solution_handles(sol: Any) -> Any:
    """
    Retrieve underlying PETSc solver handle from Solution wrapper.

    :param sol: Solution object returned by solve_on_mesh.
    :return: PETSc solver handle if found, else original solution.
    """
    for key in ("petsc_solver", "solver", "petsc_snes", "petsc_ksp"):
        if hasattr(sol, key):
            return getattr(sol, key)
    return sol


@dataclass
class PerfResult:
    """
    Dataclass capturing performance results for a single solve.

    :param approach: Name of the solver approach used.
    :param nx: Mesh resolution in x (number of elements).
    :param ny: Mesh resolution in y.
    :param dofs: Number of degrees of freedom.
    :param num_cells: Total number of mesh cells.
    :param iterations: Iteration count from solver, if available.
    :param residual: Final residual error from solver.
    :param times: Dict mapping PETSc event names to wall-times.
    :param metadata: Additional metadata (PETSc and Firedrake versions).
    """

    approach: str
    nx: int
    ny: int
    dofs: int
    num_cells: int
    iterations: Optional[int]
    residual: float
    times: Dict[str, float]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert PerfResult to a flat dictionary, prefixing event times with 'time_'."""
        base = asdict(self)
        for k, v in self.times.items():
            base[f"time_{k}"] = v
        base.pop("times", None)
        return base


def run_perf_once(
    nx: int,
    ny: int,
    approach: Approach,
    eager: bool = True,
    stage_name: Optional[str] = None,
    extra_events: Optional[List[str]] = None,
) -> PerfResult:
    """
    Run a single performance measurement for a given mesh and approach.

    :param nx: Number of elements in x direction.
    :param ny: Number of elements in y direction.
    :param approach: Solver approach to benchmark.
    :param eager: If True, perform a warmup run before timing.
    :param stage_name: Optional custom name for PETSc logging stage.
    :param extra_events: Optional additional PETSc events to record.
    :return: PerfResult with timings, iterations, and metadata.
    """
    ensure_petsc_logging()

    mesh = build_mesh(nx, ny, quadrilateral=True)
    _, _, W = build_spaces(mesh)
    bcs = default_bcs(W)
    params = default_model_params()

    stage = stage_name or f"{approach.value}(nx={nx},ny={ny})"

    # Warmup
    if eager:
        with Stage(stage + " [warmup]"):
            sol_warm = solve_on_mesh(W, approach, params=params, bcs=bcs)
            _enable_convergence_history_if_possible(_extract_solution_handles(sol_warm))

    # Timed
    with Stage(stage):
        sol = solve_on_mesh(W, approach, params=params, bcs=bcs)

    comm = mesh.comm
    num_cells = comm.allreduce(mesh.cell_set.size, op=MPI.SUM)
    dofs = W.dim()

    solver_handle = _extract_solution_handles(sol)
    ksp_iters = _extract_ksp_iters_if_possible(solver_handle)
    iterations = ksp_iters if ksp_iters is not None else getattr(sol, "iteration_number", None)
    residual = float(getattr(sol, "residual_error", np.nan))

    times = collect_event_times(comm, extra_events=extra_events)

    meta: Dict[str, Any] = {
        "petsc_version": PETSc.Sys.getVersion(),
        "firedrake_version": getattr(fd, "__version__", None),
    }

    return PerfResult(
        approach=approach.value,
        nx=nx,
        ny=ny,
        dofs=int(dofs),
        num_cells=int(num_cells),
        iterations=iterations,
        residual=residual,
        times=times,
        metadata=meta,
    )


def run_perf_sweep(
    mesh_sizes: List[int],
    approaches: List[Approach],
    extra_events: Optional[List[str]] = None,
    eager: bool = True,
) -> pd.DataFrame:
    """
    Run performance sweep over mesh sizes and solver approaches.

    :param mesh_sizes: List of mesh resolutions (nx=ny) to test.
    :param approaches: List of Approach enums to benchmark.
    :param extra_events: Optional additional PETSc events to record.
    :param eager: If True, perform warmup runs before each timing.
    :return: pandas DataFrame in tidy format with performance metrics.
    """
    rows: List[Dict[str, Any]] = []
    for nx in mesh_sizes:
        ny = nx
        for ap in approaches:
            res = run_perf_once(nx, ny, ap, eager=eager, extra_events=extra_events)
            rows.append(res.to_dict())
            PETSc.Sys.Print(
                f"[perf] nx={nx} {ap.value}: iters={res.iterations}, "
                f"KSPSolve={res.times.get('KSPSolve', 0):.3e}s, "
                f"PCApply={res.times.get('PCApply', 0):.3e}s"
            )
    return pd.DataFrame(rows)


def save_perf_csv(df: pd.DataFrame, path: str) -> None:
    """
    Save performance DataFrame to CSV file.

    :param df: pandas DataFrame with performance results.
    :param path: File path for output CSV.
    :return: None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def save_perf_json(df: pd.DataFrame, path: str) -> None:
    """
    Save performance DataFrame to JSON file.

    :param df: pandas DataFrame with performance results.
    :param path: File path for output JSON.
    :return: None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)
