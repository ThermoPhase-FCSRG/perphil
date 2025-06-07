# %%
import os
import matplotlib.pyplot as plt

os.environ["OMP_NUM_THREADS"] = "1"

import firedrake as fd
import numpy as np

# %% [markdown]
# For convenience, we define the operators from Firedrake:

# %%
grad = fd.grad
div = fd.div
dx = fd.dx
inner = fd.inner
pi = fd.pi
sin = fd.sin
exp = fd.exp
cos = fd.cos

# %% [markdown]
# ## Exact solutions

# %%
# 1) Create a mesh and function‐spaces.  For example, a unit square:
num_elements_x, num_elements_y = 10, 10
enable_run_on_quads = True
mesh = fd.UnitSquareMesh(
    num_elements_x,
    num_elements_y,
    quadrilateral=enable_run_on_quads,
)

# Vector‐valued H1 space for velocity, and scalar CG1 space for pressure:
V = fd.VectorFunctionSpace(mesh, "CG", 1)
Q = fd.FunctionSpace(mesh, "CG", 1)

# 2) Declare SpatialCoordinate and all parameters:
x, y = fd.SpatialCoordinate(mesh)

# Physical / problem parameters (you can change these as needed):
k1 = fd.Constant(1.0e0)  # example value for k1
k2 = k1 / 1e2  # example value for k2
beta = fd.Constant(1.0e0)  # example value for β
mu = fd.Constant(1.0e0)  # example value for μ

# Define η = sqrt(β (k1 + k2) / (k1 k2))
eta = fd.sqrt(beta * (k1 + k2) / (k1 * k2))

# 3) Build the UFL expressions for u1, p1, u2, p2 exactly as given:
u1_expr = fd.as_vector(
    [
        -k1 * (exp(pi * x) * sin(pi * y)),
        -k1 * (exp(pi * x) * cos(pi * y) - (eta / (beta * k1)) * exp(eta * y)),
    ]
)

p1_expr = (mu / pi) * exp(pi * x) * sin(pi * y) - (mu / (beta * k1)) * exp(eta * y)

u2_expr = fd.as_vector(
    [
        -k2 * (exp(pi * x) * sin(pi * y)),
        -k2 * (exp(pi * x) * cos(pi * y) + (eta / (beta * k2)) * exp(eta * y)),
    ]
)

p2_expr = (mu / pi) * exp(pi * x) * sin(pi * y) + (mu / (beta * k2)) * exp(eta * y)

# 4) Now interpolate each analytic expression into a Firedrake Function:
u1_exact = fd.Function(V, name="u1_analytic")
u1_exact.interpolate(u1_expr)

p1_exact = fd.Function(Q, name="p1_analytic")
p1_exact.interpolate(p1_expr)

u2_exact = fd.Function(V, name="u2_analytic")
u2_exact.interpolate(u2_expr)

p2_exact = fd.Function(Q, name="p2_analytic")
p2_exact.interpolate(p2_expr)

# %%
fig, axes = plt.subplots()
contours = fd.tripcolor(p1_exact, axes=axes, cmap="inferno")
axes.set_aspect("equal")
axes.set_title(r"$p_1$ scalar field")
fig.colorbar(contours)
plt.show()

fig, axes = plt.subplots()
contours = fd.tripcolor(p2_exact, axes=axes, cmap="inferno")
axes.set_aspect("equal")
axes.set_title(r"$p_2$ scalar field")
fig.colorbar(contours)
plt.show()

fig, axes = plt.subplots()
contours = fd.quiver(u1_exact, axes=axes, cmap="inferno")
axes.set_aspect("equal")
axes.set_title(r"$u_1$ vector field")
fig.colorbar(contours)
plt.show()

fig, axes = plt.subplots()
contours = fd.quiver(u2_exact, axes=axes, cmap="inferno")
axes.set_aspect("equal")
axes.set_title(r"$u_2$ vector field")
fig.colorbar(contours)
plt.show()

# %% [markdown]
# ## Conforming Galerkin FEM approximations

# %% [markdown]
# ### Monolithic (fully coupled) approximation

# %%
# Approximation degree
degree = 1

# Function space declaration
pressure_family = "CG"
velocity_family = "CG"
U = fd.VectorFunctionSpace(mesh, velocity_family, degree)
V = fd.FunctionSpace(mesh, pressure_family, degree)
W = V * V

# Trial and test functions
p1, p2 = fd.TrialFunctions(W)
q1, q2 = fd.TestFunctions(W)

# Forcing function
f = fd.Constant(0.0)

# Dirichlet BCs
bc_macro = fd.DirichletBC(W.sub(0), p1_exact, "on_boundary")
bc_micro = fd.DirichletBC(W.sub(1), p2_exact, "on_boundary")
bcs = [bc_macro, bc_micro]

# Variational form
## Mass transfer term
xi = -beta / mu * (p1 - p2)

## Macro terms
a = (k1 / mu) * inner(grad(p1), grad(q1)) * dx - xi * q1 * dx
L = f * q1 * dx

## Micro terms
a += (k2 / mu) * inner(grad(p2), grad(q2)) * dx + xi * q2 * dx
L += f * q2 * dx

# Solving the problem
solver_parameters = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
solution = fd.Function(W)
problem = fd.LinearVariationalProblem(a, L, solution, bcs=bcs, constant_jacobian=True)
solver = fd.LinearVariationalSolver(problem, solver_parameters=solver_parameters)
solver.solve()

# Retrieving the solution
p1_h = fd.Function(V, name="p1_h")
p2_h = fd.Function(V, name="p2_h")
p1_h.assign(solution.sub(0))
p2_h.assign(solution.sub(1))
u1_h = fd.project(-grad(p1_h), U)
u2_h = fd.project(-grad(p2_h), U)

# %%
fig, axes = plt.subplots()
colors = fd.tripcolor(p1_h, axes=axes, cmap="inferno")
fd.triplot(mesh, axes=axes, interior_kw={"edgecolors": "red"}, boundary_kw={"colors": "red"})
axes.set_aspect("equal")
axes.set_title(r"$p_1$ scalar field")
fig.colorbar(colors)
plt.show()

fig, axes = plt.subplots()
colors = fd.tripcolor(p2_h, axes=axes, cmap="inferno")
fd.triplot(mesh, axes=axes, interior_kw={"edgecolors": "red"}, boundary_kw={"colors": "red"})
axes.set_aspect("equal")
axes.set_title(r"$p_2$ scalar field")
fig.colorbar(colors)
plt.show()

fig, axes = plt.subplots()
fd.triplot(mesh, axes=axes, boundary_kw={"colors": "k"})
colors = fd.quiver(u1_h, axes=axes, cmap="inferno")
axes.set_aspect("equal")
axes.set_title(r"$u_1$ vector field")
fig.colorbar(colors)
plt.show()

fig, axes = plt.subplots()
fd.triplot(mesh, axes=axes, boundary_kw={"colors": "k"})
colors = fd.quiver(u2_h, axes=axes, cmap="inferno")
axes.set_aspect("equal")
axes.set_title(r"$u_2$ vector field")
fig.colorbar(colors)
plt.show()


# %%
def get_xy_coordinate_points(function_space, mesh):
    x, y = fd.SpatialCoordinate(mesh)

    xfunc = fd.Function(function_space).interpolate(x)
    x_points = np.unique(np.array(xfunc.dat.data))

    yfunc = fd.Function(function_space).interpolate(y)
    y_points = np.unique(np.array(yfunc.dat.data))

    return x_points, y_points


def retrieve_solution_on_line_fixed_x(solution, function_space, mesh, x_value):
    _, y_points = get_xy_coordinate_points(function_space, mesh)
    solution_on_a_line = [solution.at([x_value, y_point]) for y_point in y_points]
    solution_on_a_line = np.array(solution_on_a_line)
    return solution_on_a_line


# %%
# Fixed x-point coordinate to slice the solution
x_points, y_points = get_xy_coordinate_points(V, mesh)
x_mid_point = (x_points.min() + x_points.max()) / 2

p1_at_x_mid_point = retrieve_solution_on_line_fixed_x(p1_h, V, mesh, x_mid_point)

p2_at_x_mid_point = retrieve_solution_on_line_fixed_x(p2_h, V, mesh, x_mid_point)

p1_exact_at_x_mid_point = retrieve_solution_on_line_fixed_x(p1_exact, V, mesh, x_mid_point)

p2_exact_at_x_mid_point = retrieve_solution_on_line_fixed_x(p2_exact, V, mesh, x_mid_point)

# %%
figsize = (7, 7)
plt.figure(figsize=figsize)
plt.plot(y_points, p1_at_x_mid_point, "x", ms=10, lw=4, c="k", label="CG")
plt.plot(y_points, p1_exact_at_x_mid_point, lw=4, c="k", label="Exact Solution")
plt.legend(frameon=False)
plt.xlabel("y coordinate")
plt.ylabel(r"Macro Pressure $(p_{1,h})$")
plt.title(f"At x= {x_mid_point:.2f}")
plt.show()

plt.figure(figsize=figsize)
plt.plot(y_points, p2_at_x_mid_point, "x", ms=10, lw=4, c="k", label="CG")
plt.plot(y_points, p2_exact_at_x_mid_point, lw=4, c="k", label="Exact Solution")
plt.legend(frameon=False)
plt.xlabel("y coordinate")
plt.ylabel(r"Micro Pressure $(p_{2,h})$")
plt.title(f"At x= {x_mid_point:.2f}")
plt.show()
