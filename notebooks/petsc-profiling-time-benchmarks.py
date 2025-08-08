# %% [markdown]
# # PETSc performance breakdown
#
# This notebook runs timing breakdown using PETSc events:
#
# - Warmup + timed solve per approach and mesh size
# - Event times: SNESSolve, KSPSolve, PCSetUp, PCApply, Jacobian/Residual evals, MatAssembly
# - Outputs a CSV of results
#
# You can tweak the mesh sizes and approaches as needed.
#
# Highly inspired by this work: https://github.com/thomasgibson/tabula-rasa

# %%
import os

os.environ["OMP_NUM_THREADS"] = "1"

from typing import List
import matplotlib.pyplot as plt

from perphil.experiments.iterative_bench import Approach
from perphil.experiments.petsc_profiling import run_perf_sweep, save_perf_csv

# %% [markdown]
# ## Parameters

# %%
mesh_sizes: List[int] = [8, 16, 32, 64]
approaches: List[Approach] = [
    Approach.PLAIN_GMRES,
    Approach.GMRES_ILU,
    # Approach.SS_GMRES,
    # Approach.SS_GMRES_ILU,
    # Approach.PICARD_MUMPS,
    # Approach.MONOLITHIC_MUMPS,
]
extra_events: List[str] = []

# %% [markdown]
# ## Run and collect

# %%
df = run_perf_sweep(mesh_sizes, approaches, repeats=7, backend="events")
df.sort_values(["nx", "approach"])

# %% [markdown]
# ## Save CSV (optional)

# %%
save_perf_csv(df, "results/petsc_perf_breakdown.csv")
df.head()

# %% [markdown]
# ## Plot: KSPSolve and PCApply share

# %%
for metric in ("time_KSPSolve", "time_PCApply", "time_PCSetUp", "flops_total"):
    pivot = df.pivot(index="nx", columns="approach", values=metric).sort_index()
    ax = pivot.plot(kind="bar", figsize=(9, 6), title=metric)
    ax.set_xlabel("nx (=ny)")
    ax.set_ylabel("time [s]")
    ax.grid(axis="y", ls=":")
    plt.tight_layout()
    plt.show()

# %%
df
