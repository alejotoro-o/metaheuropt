import numpy as np
from .base import BaseSolver

class SA(BaseSolver):
    """
    Simulated Annealing (SA).
    
    A trajectory-based metaheuristic inspired by the annealing process in metallurgy. 
    It explores the search space by moving to neighboring solutions, always 
    accepting improvements and occasionally accepting worse solutions with a 
    probability that decreases over time (temperature cooling). This mechanism 
    allows the algorithm to escape local optima.
    """

    def __init__(self, bounds, pop_size=1, max_iter=1000, t0=100, alpha=0.99, step_size=0.1, stop_patience=100):
        """
        Args:
            bounds (tuple): (lower_bounds, upper_bounds).
            pop_size (int): Number of independent SA chains to run in parallel.
            max_iter (int): Total number of cooling steps.
            t0 (float): Initial temperature. Higher values increase early exploration.
            alpha (float): Cooling rate (typically [0.8, 0.99]). Determines how 
                           fast the temperature decays.
            step_size (float): Relative perturbation radius (fraction of search 
                               range) used for generating neighbors.
            stop_patience (int): Iterations to wait before stagnation cutoff.
        """
        
        # SA usually works with pop_size=1, but we support N independent chains
        super().__init__("SA", pop_size, max_iter, bounds, stop_patience)
        
        # Hyperparameters
        self.t0 = t0           # Initial Temperature
        self.alpha = alpha     # Cooling rate (T = T * alpha)
        self.step_size = step_size # Perturbation radius
        
        # Internal state
        self.current_iter = 0
        self.t = t0

    def init_solver(self, obj_func):
        
        self.current_iter = 0
        self.t = self.t0
        self.population = self._initialize_population()
        self.fitness = np.array([obj_func(ind) for ind in self.population])
        
        # Global best tracking
        idx_best = np.argmin(self.fitness)
        self.best_fitness = self.fitness[idx_best]
        self.best_solution = self.population[idx_best].copy()

    def step(self, obj_func):
        
        self.current_iter += 1
        improved_global = False

        for i in range(self.pop_size):
            # 1. Generate neighbor (Perturbation)
            # We perturb the current solution by a random amount proportional to step_size
            neighbor = self.population[i] + self.step_size * (self.ub - self.lb) * np.random.uniform(-1, 1, self.dim)
            neighbor = self._clip_bounds(neighbor)
            
            f_neighbor = obj_func(neighbor)
            
            # 2. Acceptance Logic (Metropolis Criterion)
            delta_e = f_neighbor - self.fitness[i]
            
            # If better, always accept. If worse, accept with probability exp(-delta_e / T)
            if delta_e < 0 or np.random.rand() < np.exp(-delta_e / (self.t + 1e-15)):
                self.population[i] = neighbor
                self.fitness[i] = f_neighbor
                
                # 3. Update Global Best
                if f_neighbor < self.best_fitness:
                    self.best_fitness = f_neighbor
                    self.best_solution = neighbor.copy()
                    improved_global = True

        # 4. Cooling Schedule
        self.t *= self.alpha
        
        return improved_global