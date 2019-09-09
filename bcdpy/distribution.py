import numpy as np
import scipy as sp


class Distribution:
    pass

class StudentT(Distribution):
    """ Generalized Student t distribution 
    https://en.wikipedia.org/wiki/Student%27s_t-distribution#Generalized_Student's_t-distribution
    """
    def __init__(self, mu0=0, kappa0=1, alpha0=1, beta0=1):
        self.mu0    = np.array([mu0])
        self.kappa0 = np.array([kappa0])
        self.alpha0 = np.array([alpha0])
        self.beta0  = np.array([beta0])
        self.reset_params()

    def reset_params(self):
        self.muT    = self.mu0.copy()
        self.kappaT = self.kappa0.copy()
        self.alphaT = self.alpha0.copy()
        self.betaT  = self.beta0.copy()

    def pdf(self, x):
        """ Probability Density Function
        """
        return sp.stats.t.pdf(x - self.muT, np.power(2*self.alphaT, 0.5))

    def update_params(self, x):
        """Update Sufficient Statistcs (Parameters)
        """
        self.muT = np.concatenate([self.mu0, (self.kappaT * self.muT + x) / (self.kappaT + 1) ])
        self.kappaT = np.concatenate([self.kappa0, self.kappaT + 1 ])
        self.alphaT = np.concatenate([self.alpha0, self.alphaT + 0.5 ])
        self.betaT = np.concatenate(
            [self.beta0,
             (self.kappaT
              + (self.kappaT * (x - self.muT)**2) / (2 * (self.kappaT + 1)))])
