import firedrake as fd


def create_mesh(num_x: int, num_y: int, quadrilateral: bool = True) -> fd.Mesh:
    """
    Create a 2D unit-square mesh for DPP problems.

    :param num_x:
        Number of elements in the x direction.

    :param num_y:
        Number of elements in the y direction.

    :param quadrilateral:
        Whether to use quadrilateral elements.

    :return:
        A Firedrake Mesh instance representing the unit square.
    """
    return fd.UnitSquareMesh(num_x, num_y, quadrilateral=quadrilateral)
