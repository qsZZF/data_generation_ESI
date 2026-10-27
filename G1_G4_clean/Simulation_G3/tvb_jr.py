"""TVB Jansen–Rit equations with overflow-safe logistic evaluation.

The model, parameters, initial history, and state ordering are unchanged.
Only 1 / (1 + exp(-x)) is evaluated with SciPy's stable expit.
"""

import numpy as np
from scipy.special import expit
from tvb.simulator.coupling import SigmoidalJansenRit
from tvb.simulator.models.jansen_rit import JansenRit


class StableJansenRit(JansenRit):
    """Preserve TVB's six-state JR model, including regional parameters."""

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        # TVB's Numba model keeps regional parameters as (regions,), not (regions, 1).
        y0, y1, y2, y3, y4, y5 = state_variables[..., 0]
        voltage = y1 - y2
        pyramidal_rate = 2 * self.nu_max * expit(self.r * (voltage - self.v0))
        excitatory_rate = 2 * self.nu_max * expit(self.r * (self.a_1 * self.J * y0 - self.v0))
        inhibitory_rate = 2 * self.nu_max * expit(self.r * (self.a_3 * self.J * y0 - self.v0))
        drive = self.mu + coupling[0, :, 0] + local_coupling * voltage
        derivatives = np.array([
            y3,
            y4,
            y5,
            self.A * self.a * pyramidal_rate - 2 * self.a * y3 - self.a**2 * y0,
            self.A * self.a * (drive + self.a_2 * self.J * excitatory_rate)
            - 2 * self.a * y4 - self.a**2 * y1,
            self.B * self.b * self.a_4 * self.J * inhibitory_rate
            - 2 * self.b * y5 - self.b**2 * y2,
        ])
        return derivatives[..., np.newaxis]


class StableSigmoidalJansenRit(SigmoidalJansenRit):
    """Keep TVB's coupling shape and post-sum gain, with a stable sigmoid."""

    def pre(self, x_i, x_j):
        voltage = x_j[:, 0] - x_j[:, 1]
        rates = self.cmin + (self.cmax - self.cmin) * expit(self.r * (voltage - self.midpoint))
        return rates[:, np.newaxis]


class SparseSigmoidalJansenRit(StableSigmoidalJansenRit):
    """Evaluate the same coupling on existing edges, preserving TVB delays."""

    def __call__(self, step, history):
        # Keep TVB's broadcasting contract for spatial or mode-specific parameters.
        if not hasattr(history, "query_sparse") or any(
            value.size != 1 for value in (self.cmin, self.cmax, self.r, self.midpoint, self.a)
        ):
            return super().__call__(step, history)

        _, delayed = history.query_sparse(step)
        voltage = delayed[0] - delayed[1]
        rates = self.cmin + (self.cmax - self.cmin) * expit(self.r * (voltage - self.midpoint))
        weighted = history.nnz_weights[:, np.newaxis] * rates
        summed = np.empty((1, history.n_node, history.n_mode))
        for mode in range(history.n_mode):
            summed[0, :, mode] = np.bincount(
                history.nnz_row_el_idx, weights=weighted[:, mode], minlength=history.n_node
            )
        # DenseHistory broadcasts its weights over both coupling variables.
        return self.post(np.repeat(summed, history.n_cvar, axis=0))
