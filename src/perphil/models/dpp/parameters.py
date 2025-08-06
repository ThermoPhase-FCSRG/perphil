import firedrake as fd
import attr


@attr.define
class DPPParameters:
    """
    Container for double-porosity/permeability model parameters.

    :param k1:
        Macro permeability constant.

    :param k2:
        Micro permeability constant. If None, defaults to k1/1e2.

    :param beta:
        Mass-transfer coefficient.

    :param mu:
        Viscosity parameter.
        
    :param eta:
        Permeability constrast parameter (computed from k1 and k2 only).
    """

    k1: float | fd.Constant = 1.0
    k2: float | fd.Constant | None = None
    beta: float | fd.Constant = 1.0
    mu: float | fd.Constant = 1.0
    scale_contrast: float = 1e2

    def __attrs_post_init__(self):
        if not isinstance(self.k1, fd.Constant):
            self.k1 = fd.Constant(self.k1)
        if self.k2 is None:
            self.k2 = self.k1 / self.scale_contrast
        if not isinstance(self.k2, fd.Constant):
            self.k2 = fd.Constant(self.k2)
        if not isinstance(self.beta, fd.Constant):
            self.beta = fd.Constant(self.beta)
        if not isinstance(self.mu, fd.Constant):
            self.mu = fd.Constant(self.mu)
        
    @property
    def eta(self) -> fd.Constant:
        """
        :return:
            Computed eta = sqrt(beta*(k1+k2) / (k1*k2)).
        """
        assert type(self.k2) is not None
        assert isinstance(self.k1, fd.Constant)
        eta_expression = fd.sqrt(self.beta * (self.k1 + self.k2) / (self.k1 * self.k2))
        return eta_expression
