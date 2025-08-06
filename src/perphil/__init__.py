try:
    import firedrake
except ImportError:
    raise ImportError(
        "This package requires Firedrake."
        "Please install Firedrake and then install `perphil` in the same env."
    )
