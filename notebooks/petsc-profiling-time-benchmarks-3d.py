# %% [markdown]
# # PETSc performance breakdown (3D MMS)
#
# This notebook mirrors the 2D profiling but solves a 3D manufactured case (paper Eq. 6.3) on UnitCube meshes with Dirichlet BCs from the exact pressures.
#
# It produces a CSV and a couple of summary plots.

# %%
import os

os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from perphil.experiments.iterative_bench import Approach
from perphil.experiments.petsc_profiling_3d import run_perf_sweep_3d, save_perf_csv

RESULTS_PATH = Path("results-conforming-3d/petsc_profiling")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Parameters

# %%
mesh_sizes: List[int] = [4, 8, 12, 16, 20, 32, 36, 40]
approaches: List[Approach] = [
    Approach.PLAIN_GMRES,
    Approach.GMRES_ILU,
    Approach.SS_GMRES,
    Approach.SS_GMRES_ILU,
    Approach.MONOLITHIC_MUMPS,
]
repeats = 3

# %% [markdown]
# ## Run and collect

# %%
df = run_perf_sweep_3d(mesh_sizes, approaches, repeats=repeats, backend="events")
df.sort_values(["nx", "approach"])

# %% [markdown]
# ## Save CSV

# %%
save_perf_csv(df, str(RESULTS_PATH / "petsc_perf_breakdown_3d.csv"))
df.head()

# %% [markdown]
# ## Plots

# %%
for metric in ("time_total", "time_KSPSolve"):
    pivot = df.pivot(index="nx", columns="approach", values=metric).sort_index()
    ax = pivot.plot(kind="bar", figsize=(10, 8), title=f"3D {metric}", rot=0, logy=True)
    ax.set_xlabel("nx (=ny=nz)")
    ax.set_ylabel("time [s]")
    ax.grid(axis="y", ls=":")
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / f"petsc_{metric}_3d_bar.png")
    plt.show()

# %%
# Additional plots matching 2D notebook, adapted for 3D
for metric in ("time_total", "time_PCApply", "time_PCSetUp"):
    pivot = df.pivot(index="nx", columns="approach", values=metric).sort_index()
    ax = pivot.plot(kind="bar", figsize=(10, 8), title=f"3D {metric}", logy=True, rot=0)
    ax.set_xlabel("nx (=ny=nz)")
    ax.set_ylabel("time [s]")
    ax.grid(axis="y", ls=":")
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / f"petsc_{metric}_3d_bar.png")
    plt.show()

pivot = df.pivot(index="nx", columns="approach", values="flops_total").sort_index()
ax = pivot.plot(kind="bar", figsize=(10, 8), title="3D flops_total", logy=True, rot=0)
ax.set_xlabel("nx (=ny=nz)")
ax.set_ylabel("FLOPS [-]")
ax.grid(axis="y", ls=":")
plt.tight_layout()
plt.savefig(RESULTS_PATH / "petsc_flops_3d_bar.png")
plt.show()

pivot = df.pivot(index="nx", columns="approach", values="iterations").sort_index()
ax = pivot.plot(kind="bar", figsize=(10, 8), title="3D iterations", logy=True, rot=0)
ax.set_xlabel("nx (=ny=nz)")
ax.set_ylabel("Iterations [-]")
ax.grid(axis="y", ls=":")
plt.tight_layout()
plt.savefig(RESULTS_PATH / "petsc_iterations_3d_bar.png")
plt.show()

# Picard per-iteration diagnostics (may be empty if no Picard runs)
df_picard = df[df["approach"].str.contains("Picard", na=False)]
if not df_picard.empty:
    df_picard_per_iteration = pd.DataFrame(
        {
            "approach": df_picard["approach"],
            "num_cells": df_picard["num_cells"],
            "solve_per_iteration": df_picard.get("time_SNESSolve", 0.0)
            / df_picard["iterations"],
            "time_per_iteration": df_picard["time_total"] / df_picard["iterations"],
        }
    )
    # display(df_picard_per_iteration)

# GMRES per-iteration and shares
df_gmres = df[df["approach"].str.contains("GMRES", na=False)]
df_gmres_per_iteration = pd.DataFrame(
    {
        "approach": df_gmres["approach"],
        "num_cells": df_gmres["num_cells"],
        "solve_per_iteration": df_gmres.get("time_KSPSolve", 0.0)
        / df_gmres["iterations"],
        "time_per_iteration": df_gmres["time_total"] / df_gmres["iterations"],
        "pc_factorization_per_time": df_gmres.get("time_PCSetUp", 0.0)
        / df_gmres["time_total"],
        "pc_application_per_time": df_gmres.get("time_PCApply", 0.0)
        / df_gmres["time_total"],
    }
)
# display(df_gmres_per_iteration)

for metric in ("solve_per_iteration", "time_per_iteration"):
    pivot = df_gmres_per_iteration.pivot(
        index="num_cells", columns="approach", values=metric
    ).sort_index()
    ax = pivot.plot(
        marker="o",
        figsize=(10, 8),
        title=f"3D {metric} vs num_cells",
        logx=True,
        logy=True,
    )
    ax.set_xlabel("num_cells")
    ax.set_ylabel("time [s]")
    ax.grid(which="both", ls=":")
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / f"{metric}_3d_scaling.png")
    plt.show()

# Time breakdown at largest nx for GMRES approaches
nx_target = int(pd.to_numeric(df["nx"], errors="coerce").max())
dft = df[
    (pd.to_numeric(df["nx"], errors="coerce") == nx_target)
    & (df["approach"].astype(str).str.contains("GMRES"))
].copy()


def _assembly_row(r):
    return (
        float(r.get("time_SNESFunctionEval", 0.0))
        + float(r.get("time_SNESJacobianEval", 0.0))
        + float(r.get("time_MatAssemblyBegin", 0.0))
        + float(r.get("time_MatAssemblyEnd", 0.0))
    )


def _partition_row_wall(r):
    tot = max(float(r.get("time_total", 0.0)), 0.0)
    ksp = max(float(r.get("time_KSPSolve", 0.0)), 0.0)
    pcsetup = max(float(r.get("time_PCSetUp", 0.0)), 0.0)
    pcapply = max(float(r.get("time_PCApply", 0.0)), 0.0)
    matmult_all = max(float(r.get("time_MatMult", 0.0)), 0.0)
    gmres_orth_all = max(
        float(r.get("time_KSPGMRESOrthogonalization", 0.0)), 0.0
    ) + max(float(r.get("time_KSPGMRESBuildBasis", 0.0)), 0.0)
    assembly_all = max(_assembly_row(r), 0.0)

    rem_ksp = ksp
    pcapply_ex = min(pcapply, rem_ksp)
    rem_ksp -= pcapply_ex
    gmres_orth_ex = min(gmres_orth_all, rem_ksp)
    rem_ksp -= gmres_orth_ex
    matmult_ex = min(matmult_all, rem_ksp)
    rem_ksp -= matmult_ex
    ksp_other = max(rem_ksp, 0.0)

    ksp_sum = pcapply_ex + gmres_orth_ex + matmult_ex + ksp_other
    if tot > 0.0 and ksp_sum > tot:
        scale = tot / ksp_sum
        pcapply_ex *= scale
        gmres_orth_ex *= scale
        matmult_ex *= scale
        ksp_other *= scale
        ksp_sum = pcapply_ex + gmres_orth_ex + matmult_ex + ksp_other

    rem_out = max(tot - ksp_sum, 0.0)
    pcsetup_ex = min(pcsetup, rem_out)
    rem_out -= pcsetup_ex
    assembly_ex = min(assembly_all, rem_out)
    rem_out -= assembly_ex
    unattributed = max(rem_out, 0.0)

    return pd.Series(
        {
            "PC setup": pcsetup_ex,
            "PC apply": pcapply_ex,
            "GMRES orth": gmres_orth_ex,
            "MatMult": matmult_ex,
            "KSP other": ksp_other,
            "Assembly": assembly_ex,
            "Unattributed": unattributed,
        }
    )


if not dft.empty:
    parts = dft.apply(_partition_row_wall, axis=1)
    err = (parts.sum(axis=1) - dft["time_total"].astype(float)).abs()
    assert (err < 1e-9).all() or (
        err / dft["time_total"].replace(0, np.nan) < 1e-6
    ).all()

    # Absolute seconds (stack sums to time_total)
    parts.index = dft["approach"].astype(str).values
    ax = parts.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 8),
        title=f"Runtime composition (absolute, nx={nx_target})",
        rot=45,
        logy=True,
    )
    ax.set_ylabel("Time [s]")
    ax.legend(loc="upper right", ncols=2)
    ax.grid(axis="y", ls=":")
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / f"time_breakdown_wall_absolute_3d_nx{nx_target}.png")
    plt.show()

    # Percent of total runtime
    percent = 100.0 * parts.div(dft["time_total"].values, axis=0).clip(lower=0.0)
    ax = percent.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 8),
        title=f"Runtime composition (percent of wall, nx={nx_target})",
        rot=45,
    )
    ax.set_ylabel("Percent of wall time [%]")
    ax.legend(loc="upper right", ncols=2)
    ax.grid(axis="y", ls=":")
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / f"time_breakdown_wall_percent_3d_nx{nx_target}.png")
    plt.show()

    # Wall time split: KSP inclusive vs outside
    ksp_incl = dft["time_KSPSolve"].astype(float).clip(lower=0.0)
    outside = (dft["time_total"].astype(float) - ksp_incl).clip(lower=0.0)
    total_percent = 100.0 * pd.DataFrame(
        {"KSP (inclusive)": ksp_incl, "Outside KSP": outside}, index=dft["approach"]
    ).div(dft["time_total"].values, axis=0)
    ax = total_percent.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 8),
        title=f"Wall time split (nx={nx_target})",
        rot=45,
    )
    ax.set_ylabel("Percent of wall time [%]")
    ax.grid(axis="y", ls=":")
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / f"time_split_ksp_vs_outside_3d_nx{nx_target}.png")
    plt.show()

    # Inside-KSP breakdown (percent of KSPSolve)
    def _ksp_breakdown_row(r):
        ksp = float(r.get("time_KSPSolve", 0.0))
        rem = max(ksp, 0.0)
        pcapply = float(r.get("time_PCApply", 0.0))
        gmres_orth = float(r.get("time_KSPGMRESOrthogonalization", 0.0)) + float(
            r.get("time_KSPGMRESBuildBasis", 0.0)
        )
        matmult = float(r.get("time_MatMult", 0.0))
        pcapply_ex = min(pcapply, rem)
        rem -= pcapply_ex
        gmres_orth_ex = min(gmres_orth, rem)
        rem -= gmres_orth_ex
        matmult_ex = min(matmult, rem)
        rem -= matmult_ex
        ksp_other = max(rem, 0.0)
        return pd.Series(
            {
                "PC apply": pcapply_ex,
                "GMRES orth": gmres_orth_ex,
                "MatMult": matmult_ex,
                "KSP other": ksp_other,
            }
        )

    kparts = dft.apply(_ksp_breakdown_row, axis=1)
    den = dft["time_KSPSolve"].replace(0, np.nan).values
    kpercent = 100.0 * kparts.div(den, axis=0)
    ax = kpercent.set_index(dft["approach"]).plot(
        kind="bar",
        stacked=True,
        figsize=(10, 8),
        title=f"KSP time composition (nx={nx_target})",
        rot=45,
    )
    ax.set_ylabel("Percent of KSPSolve [%]")
    ax.grid(axis="y", ls=":")
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / f"ksp_composition_percent_3d_nx{nx_target}.png")
    plt.show()

    # Absolute comps side by side
    def _num_col(df_, name):
        if name in df_:
            s = pd.to_numeric(df_[name], errors="coerce")
            return s.fillna(0.0).astype(float).clip(lower=0.0)
        return pd.Series(0.0, index=df_.index, dtype=float)

    pcsetup = _num_col(dft, "time_PCSetUp")
    pcapply = _num_col(dft, "time_PCApply")
    ksp_incl = _num_col(dft, "time_KSPSolve")
    abs_df = pd.DataFrame(
        {
            "approach": dft["approach"].astype(str).values,
            "PC setup": pcsetup.values,
            "PC apply": pcapply.values,
            "KSP (inclusive)": ksp_incl.values,
        }
    ).set_index("approach")
    ax = abs_df.plot(
        kind="bar",
        stacked=False,
        rot=45,
        figsize=(10, 8),
        title=f"Absolute times (nx={nx_target})",
        logy=True,
    )
    ax.set_ylabel("Time [s]")
    ax.grid(axis="y", ls=":")
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / f"pc_setup_apply_ksp_absolute_3d_nx{nx_target}.png")
    plt.show()

    # Inside-KSP absolute breakdown
    pcapply = _num_col(dft, "time_PCApply")
    matmult = _num_col(dft, "time_MatMult")
    gmres_orth = _num_col(dft, "time_KSPGMRESOrthogonalization") + _num_col(
        dft, "time_KSPGMRESBuildBasis"
    )
    ksp = _num_col(dft, "time_KSPSolve")
    pcapply_ex = pd.concat([pcapply, ksp], axis=1).min(axis=1)
    rem = (ksp - pcapply_ex).clip(lower=0.0)
    gmres_orth_ex = pd.concat([gmres_orth, rem], axis=1).min(axis=1)
    rem = (rem - gmres_orth_ex).clip(lower=0.0)
    matmult_ex = pd.concat([matmult, rem], axis=1).min(axis=1)
    ksp_other = (rem - matmult_ex).clip(lower=0.0)
    kparts_abs = pd.DataFrame(
        {
            "approach": dft["approach"].astype(str).values,
            "PC apply": pcapply_ex.values,
            "GMRES orth": gmres_orth_ex.values,
            "MatMult": matmult_ex.values,
            "KSP other": ksp_other.values,
        }
    ).set_index("approach")
    ax = kparts_abs.plot(
        kind="bar",
        stacked=True,
        rot=45,
        figsize=(10, 8),
        title=f"Inside KSP (absolute times, nx={nx_target})",
    )
    ax.set_ylabel("Time in KSPSolve [s]")
    ax.grid(axis="y", ls=":")
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / f"ksp_internal_absolute_3d_nx{nx_target}.png")
    plt.show()

# Time vs Memory plot
df_plot = df.assign(mem_mb=(df.get("mem_rss_peak_kb", np.nan) / 1024.0))
markers = ["o", "s", "^", "D", "P", "X", "*", "v", "<", ">", "h"]
approaches_in_order = list(dict.fromkeys(df_plot["approach"]))
marker_map = {ap: markers[i % len(markers)] for i, ap in enumerate(approaches_in_order)}
fig, ax = plt.subplots(figsize=(10, 8))
for ap, sub in df_plot.groupby("approach", sort=False):
    sub = sub.sort_values("mem_mb")
    ax.plot(
        sub["mem_mb"],
        sub["time_total"],
        marker=marker_map[ap],
        markersize=7,
        markerfacecolor="none",
        linewidth=1.5,
        label=ap,
        alpha=0.9,
    )
ax.set_yscale("log")
ax.set_xlabel("RSS peak [MB]")
ax.set_ylabel("time_total [s]")
ax.grid(which="both", ls=":")
ax.legend(loc="best", title="Approach")
plt.tight_layout()
plt.savefig(RESULTS_PATH / "time_vs_memory_3d_lines.png")
plt.show()

# Show the dataframe last
# df
