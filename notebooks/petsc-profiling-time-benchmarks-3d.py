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
