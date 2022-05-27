"""
 * Pruning utilities * 

Algorithm based on: 

Zhu, M., & Gupta, S. (2017). To prune, or not to prune: Exploring the efficacy of pruning for model compression
https://doi.org/10.48550/arXiv.1710.01878
"""

import torch

from .utils import logging
logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)


class Pruning:
    """Pruning class

    calculates sparsitiy level at each time step. 

    Attributes:
        layers (List[Tuple[str, torch.Tensor]]): list of named tensors to be pruned. 
        s_i (float): initial sparsity.
        s_f (float): final sparsity.
        dt (int): timesteps per each pruning update.
        t0 (int): initial timestep to increase sparsity. 
        n (int): number of pruning steps (in terms of delta_t). 
    """
    def __init__(self, layers, s_i, s_f, dt, t0, n):

        self.layers = layers
        self.s_i = s_i
        self.s_f = s_f
        self.dt = dt
        self.t0 = t0
        self.n = n
        self._time = 0

        # Pruning state of every layer is represented
        # by the boolean masks. 0 means a neuron is dead. 
        self.masks = []
        for name, layer in self.layers:
            self.masks.append(torch.ones(layer.shape, dtype=bool))

        logger.info(f"Defined a pruner with s_i={self.s_i}, s_f={self.s_f}, dt={self.dt}, t0={self.t0}, n={self.n}")    


    def prune(self):
        """
        Prune the layers specified when creating a Pruner.
        Calculate sparsity level using current timestamp and the scheduler
        function. 
        """

        sparsity = self.scheduler()

        for (name, layer), mask in zip(self.layers, self.masks):
            
            column = layer.view(-1)
            idx = torch.topk(column.abs(), int(len(column) * sparsity),largest=False).indices

            mask.view(-1)[idx] = 0
            
            with torch.no_grad():
                layer *= mask
        
        
    def step(self):
        "Update time stamp"
        self._time += 1


    def _stats(self, num=-1) -> str:
        msg = f"Sparsity: {self.scheduler()}. "
        msg += f"Non zero weights: {[l[1].count_nonzero().item() for l in self.layers][:num]}"
        return msg

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

    layer = torch.nn.Linear(100, 10)

    prune = Pruning([("layer.weight", layer.weight), ("layer.bias", layer.bias)], 
                    s_i=0, s_f=0.5, dt=10, t0=30, n=7)
    print(prune.masks[0].shape)

    for i in range(100):
        prune.prune()
        print(f"Sparsity: {prune.scheduler()}. Non zero weights", 
              [l[1].count_nonzero() for l in prune.layers])
        prune.step()

    # print(prune)

if __name__ == "__main__":
    main()