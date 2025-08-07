# %% [markdown]
# # 2D Conforming Primal Pressure-Only DPP: Solver Comparison

# %%
import time
import matplotlib.pyplot as plt
import firedrake as fd
from perphil.forms.spaces import create_function_spaces
from perphil.forms.dpp import dpp_form, dpp_delayed_form
from perphil.mesh.builtin import create_mesh
from perphil.models.dpp.parameters import DPPParameters
from perphil.solvers.solver import solve_dpp
from perphil.solvers.conditioning import (
    get_matrix_data_from_form,
    calculate_condition_number,
)
from perphil.utils.manufactured_solutions import interpolate_exact

# %%
# Problem parameters
dpp_params = DPPParameters(k1=1.0, k2=1e-2, beta=1.0, mu=1.0)
# Mesh resolution
nx, ny = 50, 50
mesh = create_mesh(nx, ny, quadrilateral=True)
# Create function spaces (pressure only)
U, V = create_function_spaces(mesh, velocity_deg=1, pressure_deg=1)
W = V * V

# %%
# Manufactured solutions
u1_exact, p1_exact, u2_exact, p2_exact = interpolate_exact(mesh, U, V, dpp_params)
# Dirichlet BCs
bc1 = fd.DirichletBC(W.sub(0), p1_exact, "on_boundary")
bc2 = fd.DirichletBC(W.sub(1), p2_exact, "on_boundary")
bcs = [bc1, bc2]

# %%
# Assemble forms
a, L = dpp_form(W, dpp_params)
problem = (a, L, bcs, W)

# %%
# Condition number analysis
matrix_data = get_matrix_data_from_form(a, bcs)
cond_mono = calculate_condition_number(
    matrix_data.sparse_csr_data, num_of_factors=matrix_data.number_of_dofs - 1
)
print(f"Monolithic system condition number: {cond_mono}")

# Initial zero fields for delayed form
p1_zero = fd.Function(V)
p1_zero.interpolate(fd.Constant(0.0))
p2_zero = fd.Function(V)
p2_zero.interpolate(fd.Constant(0.0))

# Build bilinear forms for scale-splitting
forms_macro, forms_micro = dpp_delayed_form(V, V, dpp_params, p1_zero, p2_zero)
a_macro, _ = forms_macro
a_micro, _ = forms_micro

# Conditioning for macro and micro systems
matrix_data_macro = get_matrix_data_from_form(a_macro, [bc1])
matrix_data_micro = get_matrix_data_from_form(a_micro, [bc2])
cond_macro = calculate_condition_number(
    matrix_data_macro.sparse_csr_data,
    num_of_factors=matrix_data_macro.number_of_dofs - 1,
)
cond_micro = calculate_condition_number(
    matrix_data_micro.sparse_csr_data,
    num_of_factors=matrix_data_micro.number_of_dofs - 1,
)
print(f"Macro system condition number: {cond_macro}")
print(f"Micro system condition number: {cond_micro}")

# %%
# Solver strategies and timings
strategies = {
    "Direct LU (MUMPS)": {},
    "GMRES scale-split": {...},
    "GMRES no split": {...},
}
results = {}
for name, opts in strategies.items():
    t0 = time.time()
    sol = solve_dpp(*problem, solver_parameters=opts)  # perphil API call
    t1 = time.time()
    results[name] = t1 - t0
results

# %%
# Plot timings
names = list(results.keys())
times = list(results.values())
plt.bar(names, times)
plt.ylabel("Time (s)")
plt.xticks(rotation=45, ha="right")
plt.title("Solver comparison: 2D Conforming Primal DPP")
plt.show()
