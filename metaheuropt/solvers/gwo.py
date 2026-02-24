import numpy as np
from .base import BaseSolver

class GWO(BaseSolver):
    def __init__(self, bounds, pop_size=50, max_iter=100, stop_patience=100):
        super().__init__("GWO", pop_size, max_iter, bounds, stop_patience)
        
        # Leadership positions
        self.alpha_pos = None
        self.alpha_score = np.inf
        self.beta_pos = None
        self.beta_score = np.inf
        self.delta_pos = None
        self.delta_score = np.inf
        
        self.current_iter = 0

    def init_solver(self, obj_func):
        """Initializes the pack and identifies the initial hierarchy."""
        self.current_iter = 0
        self.population = self._initialize_population()
        self.fitness = np.array([obj_func(ind) for ind in self.population])
        
        # Sort and identify hierarchy
        sorted_indices = np.argsort(self.fitness)
        
        self.alpha_score = self.fitness[sorted_indices[0]]
        self.alpha_pos = self.population[sorted_indices[0]].copy()
        
        self.beta_score = self.fitness[sorted_indices[1]]
        self.beta_pos = self.population[sorted_indices[1]].copy()
        
        self.delta_score = self.fitness[sorted_indices[2]]
        self.delta_pos = self.population[sorted_indices[2]].copy()
        
        self.best_fitness = self.alpha_score
        self.best_solution = self.alpha_pos.copy()

    def step(self, obj_func):
        """Performs one iteration of hunting (position updates)."""
        self.current_iter += 1
        
        # Linearly decreasing 'a' from 2 to 0
        a = 2 - 2 * (self.current_iter / self.max_iter)
        
        new_population = np.zeros_like(self.population)
        
        # Update position of each search agent
        for i in range(self.pop_size):
            # Update based on Alpha
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = np.abs(C1 * self.alpha_pos - self.population[i])
            X1 = self.alpha_pos - A1 * D_alpha
            
            # Update based on Beta
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = np.abs(C2 * self.beta_pos - self.population[i])
            X2 = self.beta_pos - A2 * D_beta
            
            # Update based on Delta
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = np.abs(C3 * self.delta_pos - self.population[i])
            X3 = self.delta_pos - A3 * D_delta
            
            # The new position is the average of the positions calculated by Alpha, Beta, and Delta
            new_population[i] = (X1 + X2 + X3) / 3.0
            
        # Boundary enforcement and Evaluation
        self.population = self._clip_bounds(new_population)
        self.fitness = np.array([obj_func(ind) for ind in self.population])
        
        # Update hierarchy based on new fitness values
        improved = False
        for i in range(self.pop_size):
            if self.fitness[i] < self.alpha_score:
                # Alpha is replaced, push others down
                self.delta_score, self.delta_pos = self.beta_score, self.beta_pos.copy()
                self.beta_score, self.beta_pos = self.alpha_score, self.alpha_pos.copy()
                self.alpha_score, self.alpha_pos = self.fitness[i], self.population[i].copy()
                improved = True
            elif self.fitness[i] < self.beta_score:
                # Beta is replaced
                self.delta_score, self.delta_pos = self.beta_score, self.beta_pos.copy()
                self.beta_score, self.beta_pos = self.fitness[i], self.population[i].copy()
            elif self.fitness[i] < self.delta_score:
                # Delta is replaced
                self.delta_score, self.delta_pos = self.fitness[i], self.population[i].copy()

        self.best_fitness = self.alpha_score
        self.best_solution = self.alpha_pos.copy()
        
        return improved