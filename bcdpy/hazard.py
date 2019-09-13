import numpy as np


class Hazard:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class ConstantHazard(Hazard):
    def __init__(self, _lambda):
        self._lambda = _lambda

    def __call__(self, r):
        """
        Args:
          r: The length of the current run (np.ndarray or scalar)

        Returns:
          p: Changepoint Probabilities(np.ndarray with shape = r.shape)
        """
        if isinstance(r, np.ndarray):
            shape = r.shape
        else:
            shape = 1

        return np.ones(shape) / self._lambda
