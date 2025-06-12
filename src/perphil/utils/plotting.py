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
