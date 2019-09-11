import numpy as np


class BayesianOnlineChangePointDetection:
    T = 0

    def __init__(self, hazard, distribution):
        self.hazard = hazard
        self.distribution = distribution
        self.beliefs = np.zeros((2, 2))
        self.beliefs[0, 0] = 1.0

    def reset_params(self):
        self.T = 0
        self.beliefs = np.zeros((2, 2))
        self.beliefs[0, 0] = 1.0

    def _expand_belief_matrix(self):
        rows = np.zeros((1, 2))
        self.beliefs = np.concatenate((self.beliefs, rows), axis=0)

    def _shift_belief_matrix(self):
        current_belief = self.beliefs[:, 0]
        self.beliefs[:, 0] = self.beliefs[:, 1]
        self.beliefs[:, 1] = 0
        return current_belief

    def update(self, x):
        self._expand_belief_matrix()

        # Evaluate Predictive Probability (3 in Algortihm 1)
        probs = self.distribution.pdf(x)

        hazard = self.hazard(np.arange(self.T + 1))

        # Calculate Growth Probability (4 in Algorithm 1)
        self.beliefs[1 : self.T + 2, 1] = (
            self.beliefs[: self.T + 1, 0] * probs * (1 - hazard)
        )

        # Calculate Changepoint Probabilities (5 in Algorithm 1)
        self.beliefs[0, 1] = (self.beliefs[: self.T + 1, 0] * probs * hazard).sum()

        # Determine Run length Distribution (7 in Algorithm 1)
        self.beliefs[:, 1] = self.beliefs[:, 1] / self.beliefs[:, 1].sum()

        # Update sufficient statistics (8 in Algorithm 8)
        self.distribution.update_params(x)

        max_belief_idx = np.where(self.beliefs[:, 0] == self.beliefs[:, 0].max())[0]
        current_belief = self._shift_belief_matrix()
        self.T += 1
        return max_belief_idx, current_belief
