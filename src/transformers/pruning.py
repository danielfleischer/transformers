"""
Pruning utilities

Based on: 

Zhu, M., & Gupta, S. (2017). To prune, or not to prune: Exploring the efficacy of pruning for model compression
https://doi.org/10.48550/arXiv.1710.01878
"""

from dataclasses import dataclass
from typing import Optional, List
import torch
# from .utils import logging

# logger = logging.get_logger(__name__)




@dataclass
class Pruning:
    """Pruning class

    calculates sparsitiy level at each time step. 

    Attributes:
        layers (List[torch.Tensor]): list of tensors to be pruned. 
        s_i (float): initial sparsity.
        s_f (float): final sparsity.
        dt (int): timesteps per each pruning update.
        t0 (int): initial timestep to increase sparsity. 
        n (int): number of pruning steps (in terms of delta_t). 
    """

    layers : List[torch.Tensor]
    s_i : float
    s_f : float
    dt : int
    t0 : int
    n : int
    _time : int = 0

    def __post_init__(self):
        print(f"Defined a pruner with s_i={self.s_i}, s_f={self.s_f}, dt={self.dt}, t0={self.t0}, n={self.n}")    

    def step(self):
        "Update time stamp"
        self._time += 1

    def scheduler(self):
        """
        Implementing the scheduler.

        Returns the sparsity
        """
        
        sparse = self.s_i
        
        if self._time > self.t0:
            if self._time < (self.t0 + self.n * self.dt):
                sparse = self.s_f + (self.s_i - self.s_f) * (1 - (self._time - self.t0)/(self.n * self.dt))**3
            else:
                sparse = self.s_f

        return sparse


def main():

    prune = Pruning(s_i=0, s_f=0.5, dt=10, t0=30, n=7)
    
    for i in range(100):

        print(prune.scheduler())
        prune.step()

    # print(prune)

if __name__ == "__main__":
    main()