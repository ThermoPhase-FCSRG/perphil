import firedrake as fd
import matplotlib.pyplot as plt


def plot_scalar_field(
    scalar: fd.Function, title: str = "Scalar Field", cmap: str = "inferno"
) -> None:
    """
    Plot a scalar field using Firedrake's tripcolor.

    :param scalar:
        Scalar Function to plot.

    :param title:
        Title for the plot.

    :param cmap:
        Colormap name.
    """
    fig, ax = plt.subplots()
    cntr = fd.tripcolor(scalar, axes=ax, cmap=cmap)
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.colorbar(cntr)
    plt.show()


def plot_vector_field(
    vector: fd.Function, title: str = "Vector Field", cmap: str = "inferno"
) -> None:
    """
    Plot a vector field using Firedrake's quiver.

    :param vector:
        Vector Function to plot.

    :param title:
        Title for the plot.

    :param cmap:
        Colormap name.
    """
    fig, ax = plt.subplots()
    q = fd.quiver(vector, axes=ax, cmap=cmap)
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.colorbar(q)
    plt.show()


def plot_2d_mesh(
    mesh: fd.Mesh, title: str = "Mesh", boundary_color: str = "black", edge_color: str = "black"
) -> None:
    """
    Convenient function to plot a 2D mesh using Firedrake's triplot.

    :param mesh: 2D mesh to plot.
    :type mesh: fd.Mesh
    :param title: Title of the plot. Defaults to "Mesh".
    :type title: str, optional
    :param boundary_color: Color of the boundary edges. Defaults to "black".
    :type boundary_color: str, optional
    :param edge_color: Color of the interior edges. Defaults to "black".
    :type edge_color: str, optional
    """
    _, axes = plt.subplots()
    fd.triplot(
        mesh,
        axes=axes,
        interior_kw={"edgecolors": edge_color},
        boundary_kw={"colors": boundary_color},
    )
    axes.set_aspect("equal")
    axes.set_title(title)
    plt.show()
