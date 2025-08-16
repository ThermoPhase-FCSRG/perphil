## Copilot agent instructions for this repo

Purpose: help an AI be productive immediately in this Firedrake-based FEM codebase for double porosity/permeability (DPP) flows.

### Big picture
- Language/toolchain: Python 3.10+, Firedrake (UFL, PETSc), petsc4py, SciPy, NumPy, attrs.
- Core flow: mesh → function spaces → model parameters → UFL forms → Firedrake (Linear/Nonlinear) solver → Solution and postprocessing.
- Two solve modes:
  - Linear monolithic: `perphil.forms.dpp.dpp_form` + `perphil.solvers.solver.solve_dpp`.
  - Nonlinear Picard via SNES: `perphil.forms.dpp.dpp_splitted_form` + `perphil.solvers.solver.solve_dpp_nonlinear`.

### Repo layout (what to read first)
- `src/perphil/forms/` – UFL forms: `dpp.py` (macro/micro coupling, residuals), `spaces.py` (builds U/V spaces).
- `src/perphil/solvers/` – `solver.py` (Firedrake solvers, returns `Solution`), `parameters.py` (PETSc option dicts), `conditioning.py` (matrix assembly, symmetry, condition number).
- `src/perphil/models/dpp/parameters.py` – `DPPParameters` with Firedrake Constants (auto-coercion in `__attrs_post_init__`).
- `src/perphil/mesh/builtin.py` – simple unit-square mesh factory.
- `src/perphil/utils/` – manufactured solutions, plotting, postprocessing.
- `notebooks/` – paired with `.py` via Jupytext percent format (`jupytext.toml`).

### Environment & setup (Linux/macOS)
- Use Invoke tasks (`tasks.py`) to automate everything:
  1) `inv create_venv` → `.venv/` and upgrade pip.
  2) `inv install_deps` → installs `.[dev]` from `pyproject.toml` (ruff, mypy, invoke, jupytext, etc.).
  3) `inv download_firedrake_configure` → fetches `./firedrake-configure` matching latest Firedrake release.
  4) `inv install_system_packages` → apt/brew install prerequisites (adds OpenMPI on Linux).
  5) `inv install_petsc` → clones/builds PETSc with flags from firedrake; sets PETSC_DIR/ARCH for this session.
  6) `inv install_firedrake` → pip-installs `firedrake[check]` against that PETSc (uses `constraints.txt` to pin Cython<3.1).
- Tip: steps 5→6 are order-sensitive; `install_firedrake` expects PETSC_DIR/ARCH from the previous step.

### How to run something minimal
- Build mesh/spaces, pick parameters and solver options, solve:
  - Mesh: `mesh = perphil.mesh.builtin.create_mesh(nx, ny, quadrilateral=True)`.
  - Spaces: `U, V = perphil.forms.spaces.create_function_spaces(mesh)`; for DPP, use a mixed space W=V×V in your driver.
  - Model params: `params = DPPParameters(k1=1.0, k2=None, beta=1.0, mu=1.0)` (floats auto-wrap to `fd.Constant`).
  - Solver params: import from `perphil.solvers.parameters` (e.g., `LINEAR_SOLVER_PARAMS`, `GMRES_PARAMS`, `FIELDSPLIT_LU_PARAMS`).
  - Solve: `solve_dpp(W, params, bcs=[...], solver_parameters=LINEAR_SOLVER_PARAMS)` or `solve_dpp_nonlinear(...)`.
  - Result: a `Solution` (attrs, frozen) with `(solution|fields), iteration_number, residual_error`.

### Project-specific conventions
- Always coerce numeric model parameters to Firedrake `Constant` (handled inside `DPPParameters`). Accessors assume Constants.
- Mixed space sanity: solvers check `W.num_sub_spaces() == 2`; pass a 2-field pressure `MixedFunctionSpace`.
- PETSc options are Python dicts; prefer `mat_type: "aij"` and field-split blocks for (p1,p2). See `solvers/parameters.py` for ready sets:
  - Direct: `LINEAR_SOLVER_PARAMS` (LU via MUMPS).
  - Iterative: `GMRES_PARAMS`, with `GMRES_JACOBI_PARAMS` / `GMRES_ILU_PARAMS`.
  - Field-split: `FIELDSPLIT_LU_PARAMS` with `pc_fieldsplit_{0,1}_fields` mapped to blocks.
  - SNES: `RICHARDSON_SOLVER_PARAMS`, `NGS_SOLVER_PARAMS`, `KSP_PREONLY_PARAMS` (use with `solve_dpp_nonlinear`).
- Forms structure (see `forms/dpp.py`): mass transfer term `xi = -beta/mu * (p1 - p2)`; macro subtracts xi, micro adds xi.

### Debugging & analysis
- Verify Firedrake install: `firedrake-check` (Invoke sets `OMP_NUM_THREADS=1` temporarily during check).
- Inspect matrices: use `conditioning.get_matrix_data_from_form(form, bcs)` → PETSc handle, symmetry, CSR, nnz, DoFs.
- Condition number: `conditioning.calculate_condition_number(csr, num_of_factors, use_sparse=True|False)`.

### Dev workflows
- Dev install: `inv dev_install` (editable `perphil`).
- Pre-commit hooks: `inv hooks` then `inv run_hooks --all-files` (config in `.pre-commit-config.yaml`).
- Notebooks ↔ scripts pairing: `inv pair_ipynbs` (percent format governed by `jupytext.toml`).

### CI & GitHub Actions
- Workflow file: `.github/workflows/tests.yml` automates build, test, and coverage on `push` and `pull_request`.
  - Caches:
    - PETSc build: directory `petsc-<version>` keyed by a hash of configure flags from `firedrake-configure --show-petsc-configure-options`.
    - Python virtualenv (`.venv`) and `pip` cache.
    - `~/.ccache` store for fast recompilation.
  - CI environment sets `CC="ccache mpicc"` and `CXX="ccache mpicxx"`; `tasks.py` strips the `ccache ` prefix before calling PETSc `./configure`, ensuring the raw MPI compilers are used (avoiding "Ignoring environment variable" warnings).
  - After setup, tests run with `pytest --cov=src/perphil --cov-report=xml --cov-report=term-missing`, coverage is uploaded as `coverage.xml`, and `diff-cover` enforces PR coverage thresholds.
