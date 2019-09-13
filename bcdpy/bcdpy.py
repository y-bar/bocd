import numpy as np


class BayesianOnlineChangePointDetection:
    def __init__(self, hazard, distribution):
        self.hazard = hazard
        self.distribution = distribution
        self.T = 0
        self.beliefs = np.zeros((1, 2))
        self.beliefs[0, 0] = 1.0

    def reset_params(self):
        self.T = 0
        self.beliefs = np.zeros((1, 2))
        self.beliefs[0, 0] = 1.0

    def _expand_belief_matrix(self):
        rows = np.zeros((1, 2))
        self.beliefs = np.concatenate((self.beliefs, rows), axis=0)

    def _shift_belief_matrix(self):
        self.beliefs[:, 0] = self.beliefs[:, 1]
        self.beliefs[:, 1] = 0.0

    def update(self, x):
        self._expand_belief_matrix()

        # Evaluate Predictive Probability (3 in Algortihm 1)
        pi_t = self.distribution.pdf(x)

        # Calculate H(r_{t-1})
        h = self.hazard(self.rt)

        # Calculate Growth Probability (4 in Algorithm 1)
        self.beliefs[1 : self.T + 2, 1] = self.beliefs[: self.T + 1, 0] * pi_t * (1 - h)

        # Calculate Changepoint Probabilities (5 in Algorithm 1)
        self.beliefs[0, 1] = (self.beliefs[: self.T + 1, 0] * pi_t * h).sum()

        # Determine Run length Distribution (7 in Algorithm 1)
        self.beliefs[:, 1] = self.beliefs[:, 1] / self.beliefs[:, 1].sum()

        # Update sufficient statistics (8 in Algorithm 8)
        self.distribution.update_params(x)

        # Update internal state
        self._shift_belief_matrix()
        self.T += 1

    @property
    def rt(self):
        return np.where(self.beliefs[:, 0] == self.beliefs[:, 0].max())[0]

    @property
    def belief(self):
        return self.beliefs[:, 0]
