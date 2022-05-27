"""
Pruning utilities

Based on: 

Zhu, M., & Gupta, S. (2017). To prune, or not to prune: Exploring the efficacy of pruning for model compression
https://doi.org/10.48550/arXiv.1710.01878
"""

from dataclasses import dataclass
from typing import Optional

# from .utils import logging

# logger = logging.get_logger(__name__)




@dataclass
class Pruning:
    """Pruning class

    calculates sparsitiy level at each time step. 

    Attributes:
        s_i (float): initial sparsity.
        s_f (float): final sparsity.
        dt (int): timesteps per each pruning update.
        t0 (int): initial timestep to increase sparsity. 
        n (int): number of pruning steps (in terms of delta_t). 
    """

    s_i : float
    s_f : float
    dt : int
    t0 : int
    n : int

    def __post_init__(self):
        print(f"Defined a pruner with s_i={self.s_i}, s_f={self.s_f}, dt={self.dt}, t0={self.t0}, n={self.n}")    

    def scheduler(self, t):
        """
        Implementing the scheduler.

        Returns the sparsity
        """
        
        sparse = self.s_i
        
        if t > self.t0:
            if t < (self.t0 + self.n * self.dt):
                sparse = self.s_f + (self.s_i - self.s_f) * (1 - (t - self.t0)/(self.n * self.dt))**3
            else:
                sparse = self.s_f

        return sparse


def main():

    prune = Pruning(s_i=0, s_f=0.5, dt=10, t0=30, n=300)
    
    for i in range(100):
        print(prune.scheduler(i))

    print(prune)

if __name__ == "__main__":
    main()