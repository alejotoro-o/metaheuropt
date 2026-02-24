import numpy as np
from .base import BaseSolver

class WOA(BaseSolver):
    def __init__(self, bounds, pop_size=50, max_iter=100, b=1, stop_patience=100):
        super().__init__("WOA", pop_size, max_iter, bounds, stop_patience)
        
        # Hyperparameters
        self.b = b  # Constant for defining the shape of the logarithmic spiral
        self.current_iter = 0

    def init_solver(self, obj_func):
        """Initializes the whale pod and identifies the Leader."""
        self.current_iter = 0
        self.population = self._initialize_population()
        self.fitness = np.array([obj_func(ind) for ind in self.population])
        
        idx_best = np.argmin(self.fitness)
        self.best_fitness = self.fitness[idx_best]
        self.best_solution = self.population[idx_best].copy()

    def step(self, obj_func):
        """Performs one iteration of bubble-net hunting."""
        self.current_iter += 1
        
        # a decreases linearly from 2 to 0
        a = 2 - 2 * (self.current_iter / self.max_iter)
        
        new_population = np.zeros_like(self.population)
        
        for i in range(self.pop_size):
            r1 = np.random.rand()
            r2 = np.random.rand()
            
            A = 2 * a * r1 - a
            C = 2 * r2
            
            p = np.random.rand()
            l = -1 + 2 * np.random.rand() # parameter 'l' in [-1, 1]
            
            if p < 0.5:
                if np.abs(A) < 1:
                    # Shrinking encircling mechanism
                    D = np.abs(C * self.best_solution - self.population[i])
                    new_population[i] = self.best_solution - A * D
                else:
                    # Search for prey (exploration)
                    rand_idx = np.random.randint(self.pop_size)
                    X_rand = self.population[rand_idx]
                    D = np.abs(C * X_rand - self.population[i])
                    new_population[i] = X_rand - A * D
            else:
                # Spiral updating position (exploitation)
                D = np.abs(self.best_solution - self.population[i])
                # Spiral equation
                new_population[i] = D * np.exp(self.b * l) * np.cos(2 * np.pi * l) + self.best_solution
        
        # Boundary Control & Evaluation
        self.population = self._clip_bounds(new_population)
        self.fitness = np.array([obj_func(ind) for ind in self.population])
        
        # Update Leader
        idx_best = np.argmin(self.fitness)
        if self.fitness[idx_best] < self.best_fitness:
            self.best_fitness = self.fitness[idx_best]
            self.best_solution = self.population[idx_best].copy()
            return True
            
        return False