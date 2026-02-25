import numpy as np
from .base import BaseSolver

class MVO(BaseSolver):
    """
    Multi-Verse Optimizer (MVO).
    
    A physics-inspired metaheuristic based on three concepts in cosmology: 
    white holes, black holes, and wormholes. High-fitness universes (white holes) 
    transfer objects to low-fitness universes (black holes) through a roulette 
    wheel mechanism (exploration), while wormholes allow for local 
    perturbations around the best universe (exploitation).
    """

    def __init__(self, bounds, pop_size=50, max_iter=100, 
                 wep_min=0.2, wep_max=1.0, p=3, stop_patience=100):
        """
        Args:
            bounds (tuple): (lower_bounds, upper_bounds).
            pop_size (int): Number of universes in the multi-verse.
            max_iter (int): Maximum iterations (used for TDR and WEP adaptation).
            wep_min (float): Minimum Wormhole Existence Probability.
            wep_max (float): Maximum Wormhole Existence Probability.
            p (float): Power parameter that defines the acceleration of the 
                       Traveling Distance Rate (TDR) over iterations.
            stop_patience (int): Iterations to wait before stagnation cutoff.
        """
        
        super().__init__("MVO", pop_size, max_iter, bounds, stop_patience)
        
        # Hyperparameters
        self.wep_min = wep_min
        self.wep_max = wep_max
        self.p = p # Power parameter for TDR
        
        # Internal state
        self.current_iter = 0

    def init_solver(self, obj_func):
        
        self.current_iter = 0
        self.population = self._initialize_population()
        self.fitness = np.array([obj_func(ind) for ind in self.population])
        
        # Initial best universe identification
        idx_best = np.argmin(self.fitness)
        self.best_fitness = self.fitness[idx_best]
        self.best_solution = self.population[idx_best].copy()

    def step(self, obj_func):
        
        self.current_iter += 1
        
        # 1. Update WEP and TDR coefficients
        wep = self.wep_min + self.current_iter * (self.wep_max - self.wep_min) / self.max_iter
        tdr = 1 - (self.current_iter**(1/self.p) / self.max_iter**(1/self.p))
        
        # 2. Sort universes by inflation rate (fitness)
        # Note: Lower fitness = higher inflation rate in minimization
        sorted_idx = np.argsort(self.fitness)
        sorted_pop = self.population[sorted_idx]
        sorted_fit = self.fitness[sorted_idx]
        
        # 3. Calculate probabilities for White Hole selection (Roulette Wheel)
        f_min, f_max = sorted_fit[0], sorted_fit[-1]
        if f_max > f_min:
            # Normalize fitness to get weights
            weights = (f_max - sorted_fit) / (f_max - f_min + np.finfo(float).eps)
        else:
            weights = np.ones(self.pop_size) / self.pop_size
            
        prob = weights / np.sum(weights)
        cdf = np.cumsum(prob)
        
        # 4. Create new universes
        new_pop = np.copy(sorted_pop)
        
        # The best universe (Elite) is kept at index 0
        for i in range(1, self.pop_size):
            for j in range(self.dim):
                # Black/White Hole exchange (Exploration)
                if np.random.rand() < prob[i]:
                    white_hole_idx = np.where(cdf >= np.random.rand())[0][0]
                    new_pop[i, j] = sorted_pop[white_hole_idx, j]
                
                # Wormhole mechanism (Exploitation around best)
                if np.random.rand() < wep:
                    r2 = np.random.rand()
                    # Calculate travel distance around best solution
                    offset = tdr * ((self.ub[j] - self.lb[j]) * np.random.rand() + self.lb[j])
                    if r2 < 0.5:
                        new_pop[i, j] = self.best_solution[j] + offset
                    else:
                        new_pop[i, j] = self.best_solution[j] - offset
        
        # 5. Boundary check and Evaluation
        self.population = self._clip_bounds(new_pop)
        self.fitness = np.array([obj_func(ind) for ind in self.population])
        
        # 6. Update Global Best (Wormhole target)
        idx_best = np.argmin(self.fitness)
        if self.fitness[idx_best] < self.best_fitness:
            self.best_fitness = self.fitness[idx_best]
            self.best_solution = self.population[idx_best].copy()
            return True
            
        return False