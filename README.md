# perphil

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests and Coverage](https://github.com/volpatto/perphil/actions/workflows/tests.yml/badge.svg)](https://github.com/volpatto/perphil/actions/workflows/tests.yml)
[![Coverage badge](https://raw.githubusercontent.com/volpatto/perphil/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/volpatto/perphil/blob/python-coverage-comment-action-data/htmlcov/index.html)

Double porosity/permeability (DPP) flows with conforming FEM on Firedrake, plus solver experiments, conditioning studies, and PETSc profiling.

## What’s inside

- Firedrake UFL forms for DPP (monolithic and Picard-splitted)
- Solver presets and wrappers (linear and nonlinear) using PETSc
- Manufactured solutions (2D/3D) and post-processing utilities
- Condition-number analysis (monolithic and block-wise)
- Reproducible notebooks (paired via Jupytext)

## Supported platforms

- Linux (Ubuntu 22.04 recommended). macOS works, but build times may be longer.
- Python 3.10+

## Quick start (managed setup via Invoke)

This repo ships Invoke tasks to set up PETSc/Firedrake and Python deps in a local virtual environment.

1. Create a virtual environment and dev tools
   - Creates `.venv` and installs Python dev dependencies from `pyproject.toml`.
2. Build PETSc matched to Firedrake’s configure
3. Install Firedrake against that PETSc

Run these tasks from the repo root in order (Linux/macOS):

```sh
# 1) Create .venv and install dev deps
python3 -m venv .venv
source .venv/bin/activate
inv install-deps

# 2) Prepare system prerequisites (MPI, compilers, etc.)
inv download-firedrake-configure
inv install-system-packages  # apt (Ubuntu) or brew (macOS)

# 3) Build PETSc and install Firedrake (order matters)
inv install-petsc
inv install-firedrake

# 4) Editable install of perphil
inv dev-install
```

Notes

- Step order 3→4 is required: install_firedrake expects PETSC_DIR/ARCH exported by install_petsc.
- A constraints.txt is used to keep some build-time pins (e.g., Cython).
- You can validate Firedrake with: `firedrake-check`.

## Minimal example (2D, conforming pressures)

Solve the linear monolithic DPP system on a unit square, using the 2D manufactured solution for Dirichlet BCs.

```python
import firedrake as fd
from perphil.mesh.builtin import create_mesh
from perphil.forms.spaces import create_function_spaces
from perphil.forms.dpp import dpp_form
from perphil.models.dpp.parameters import DPPParameters
from perphil.utils.manufactured_solutions import exact_expressions
from perphil.solvers.solver import solve_dpp
from perphil.solvers.parameters import LINEAR_SOLVER_PARAMS

# Mesh and spaces
mesh = create_mesh(16, 16, quadrilateral=True)
_, V = create_function_spaces(mesh)
W = fd.MixedFunctionSpace((V, V))

# Manufactured Dirichlet data
params = DPPParameters(k1=1.0, k2=1e-2, beta=1.0, mu=1.0)
_, p1_exact, _, p2_exact = exact_expressions(mesh, params)
bcs = [
	fd.DirichletBC(W.sub(0), p1_exact, "on_boundary"),
	fd.DirichletBC(W.sub(1), p2_exact, "on_boundary"),
]

# Assemble and solve (linear)
solution = solve_dpp(W, params, bcs=bcs, solver_parameters=LINEAR_SOLVER_PARAMS)
print("iterations:", solution.iteration_number, "residual:", solution.residual_error)
```

For 3D, swap the mesh and exact data:

```python
mesh = fd.UnitCubeMesh(8, 8, 8)
from perphil.utils.manufactured_solutions import exact_expressions_3d
_, p1_exact, _, p2_exact = exact_expressions_3d(mesh, params)
```

## Condition-number studies

See notebooks in `notebooks/`:

- `condition-number-study.ipynb` (2D)
- `condition-number-study-3d.ipynb` (3D) — uses the 3D manufactured solution and assembles the DPP matrix in Firedrake.

Each notebook saves CSVs and figures under `notebooks/results-conforming-{2d|3d}/conditioning/`.

## Notebooks and Jupytext

Notebooks are paired with Python scripts via Jupytext (percent format). Useful tasks:

```sh
# Pair/unpair notebooks and scripts (see jupytext.toml)
inv pair_ipynbs

# Optional: run pre-commit hooks
inv hooks
inv run_hooks --all-files
```

## Developer tips

- Prefer PETSc matrix type `aij` and field-split options in solver presets.
- Mixed space sanity: solvers assume a 2-field pressure MixedFunctionSpace.
- Model parameter scalars are auto-coerced to Firedrake Constants.
- For analysis, you can inspect the assembled matrix with:
	`perphil.solvers.conditioning.get_matrix_data_from_form(form, bcs)` and
	`perphil.solvers.conditioning.calculate_condition_number(...)`.

## Troubleshooting

- “firedrake not found”: confirm you ran `inv install_petsc` then `inv install_firedrake` and that you are in `.venv`.
- Long compile times: prefer Ubuntu 22.04; ensure MPI toolchain and compilers are installed (`inv install_system_packages`).
- Matplotlib mathtext issues: use raw strings for LaTeX-like labels, e.g., `r"$\\log_{10}(h)$"` in notebooks.

## License

MIT — see [LICENSE](LICENSE).

## Contact

Open an issue or email me: [volpatto@lncc.br](mailto:volpatto@lncc.br)
