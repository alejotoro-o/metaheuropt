import numpy as np
from .base import BaseSolver

class DE(BaseSolver):
    """
    Differential Evolution (DE).
    
    A population-based stochastic search algorithm that optimizes problems by 
    iteratively improving a candidate solution with regard to a given measure 
    of quality. It uses vector differences for mutation and a binomial crossover 
    scheme to maintain population diversity.
    """

    def __init__(self, bounds, pop_size=50, max_iter=100, F=0.5, CR=0.9, stop_patience=100):
        """
        Args:
            bounds (tuple): (lower_bounds, upper_bounds).
            pop_size (int): Number of target vectors in the population.
            max_iter (int): Maximum number of generations.
            F (float): Mutation scale factor (typically in [0, 2]). Controls the 
                       amplification of the differential variation.
            CR (float): Crossover probability (typically in [0, 1]). Controls 
                        the fraction of parameter values copied from the mutant vector.
            stop_patience (int): Iterations to wait before stagnation cutoff.
        """
        
        super().__init__("DE", pop_size, max_iter, bounds, stop_patience)
        
        # Hyperparameters
        self.F = F    # Mutation scale factor
        self.CR = CR  # Crossover probability
        
        # Internal state
        self.current_iter = 0

    def init_solver(self, obj_func):
    
        self.current_iter = 0
        self.population = self._initialize_population()
        self.fitness = np.array([obj_func(ind) for ind in self.population])
        
        # Identify initial best
        idx_best = np.argmin(self.fitness)
        self.best_fitness = self.fitness[idx_best]
        self.best_solution = self.population[idx_best].copy()

    def step(self, obj_func):
   
        self.current_iter += 1
        new_population = np.copy(self.population)
        new_fitness = np.copy(self.fitness)
        improved_global = False

        for i in range(self.pop_size):
            # 1. Mutation: DE/rand/1
            # Select 3 distinct individuals different from 'i'
            candidates = [idx for idx in range(self.pop_size) if idx != i]
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            
            # Mutant vector: v = x_r1 + F * (x_r2 - x_r3)
            vi = self.population[r1] + self.F * (self.population[r2] - self.population[r3])
            
            # 2. Binomial Crossover
            mask = np.random.rand(self.dim) < self.CR
            # Ensure at least one dimension is inherited from the mutant
            j_rand = np.random.randint(self.dim)
            mask[j_rand] = True
            
            # Trial vector ui
            ui = np.where(mask, vi, self.population[i])
            ui = self._clip_bounds(ui)
            
            # 3. Greedy Selection
            f_ui = obj_func(ui)
            if f_ui <= self.fitness[i]:
                new_population[i] = ui
                new_fitness[i] = f_ui
                
                # Check for global improvement
                if f_ui < self.best_fitness:
                    self.best_fitness = f_ui
                    self.best_solution = ui.copy()
                    improved_global = True

        self.population = new_population
        self.fitness = new_fitness
        
        return improved_global