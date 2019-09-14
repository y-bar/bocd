import numpy as np
from scipy import stats


class Distribution:
    def reset_params(self):
        raise NotImplementedError()

    def pdf(self, x):
        raise NotImplementedError()

    def update_params(self, x):
        raise NotImplementedError()


class StudentT(Distribution):
    """ Generalized Student t distribution 
    https://en.wikipedia.org/wiki/Student%27s_t-distribution#Generalized_Student's_t-distribution

    This setting corresponds to select
      1: Gaussian distribution as a likelihood
      2: normal-Gamma distribution as a prior for Gaussian
    """

    def __init__(self, mu=0, kappa=1, alpha=1, beta=1):
        self.mu0 = np.array([mu])
        self.kappa0 = np.array([kappa])
        self.alpha0 = np.array([alpha])
        self.beta0 = np.array([beta])
        # We need the following lines to prevent "outside defined warning"
        self.muT = self.mu0.copy()
        self.kappaT = self.kappa0.copy()
        self.alphaT = self.alpha0.copy()
        self.betaT = self.beta0.copy()

    def reset_params(self):
        self.muT = self.mu0.copy()
        self.kappaT = self.kappa0.copy()
        self.alphaT = self.alpha0.copy()
        self.betaT = self.beta0.copy()

    def pdf(self, x):
        """ Probability Density Function
        """
        return stats.t.pdf(
            x,
            loc=self.muT,
            df=2 * self.alphaT,
            scale=np.sqrt(self.betaT * (self.kappaT + 1) / (self.alphaT * self.kappaT)),
        )

    def update_params(self, x):
        """Update Sufficient Statistcs (Parameters)

        To understand why we use this, see e.g.
        Conjugate Bayesian analysis of the Gaussian distribution, Kevin P. Murphy∗
        https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
        3.5 Posterior predictive
        """
        self.betaT = np.concatenate(
            [
                self.beta0,
                (self.kappaT + (self.kappaT * (x - self.muT) ** 2) / (2 * (self.kappaT + 1))),
            ]
        )
        self.muT = np.concatenate([self.mu0, (self.kappaT * self.muT + x) / (self.kappaT + 1)])
        self.kappaT = np.concatenate([self.kappa0, self.kappaT + 1])
        self.alphaT = np.concatenate([self.alpha0, self.alphaT + 0.5])
