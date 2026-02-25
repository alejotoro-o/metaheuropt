import numpy as np
from .base import BaseSolver

class GSA(BaseSolver):
    """
    Gravitational Search Algorithm (GSA).
    
    A physics-inspired optimization algorithm based on the law of gravity and 
    mass interactions. Search agents are considered as objects with masses 
    that attract each other through gravitational forces, causing them to 
    accelerate toward objects with higher masses (better solutions).
    """

    def __init__(self, bounds, pop_size=50, max_iter=100, g0=100, alpha=20, stop_patience=100):
        """
        Args:
            bounds (tuple): (lower_bounds, upper_bounds).
            pop_size (int): Number of mass agents.
            max_iter (int): Total iterations (used for gravitational constant decay).
            g0 (float): Initial gravitational constant.
            alpha (float): Gravitational decay constant; controls the reduction 
                           speed of gravity over time.
            stop_patience (int): Iterations to wait before stagnation cutoff.
        """
        
        super().__init__("GSA", pop_size, max_iter, bounds, stop_patience)
        
        # Hyperparameters
        self.g0 = g0        # Initial gravitational constant
        self.alpha = alpha  # Power of the gravitational constant decay
        
        # State variables
        self.velocity = None
        self.current_iter = 0

    def init_solver(self, obj_func):
        self.current_iter = 0
        self.population = self._initialize_population()
        self.velocity = np.zeros((self.pop_size, self.dim))
        self.fitness = np.array([obj_func(ind) for ind in self.population])
        
        idx_best = np.argmin(self.fitness)
        self.best_fitness = self.fitness[idx_best]
        self.best_solution = self.population[idx_best].copy()

    def step(self, obj_func):
        self.current_iter += 1
        
        # 1. Update Gravitational Constant (G)
        g = self.g0 * np.exp(-self.alpha * self.current_iter / self.max_iter)
        
        # 2. Calculate Masses based on fitness
        f_min, f_max = np.min(self.fitness), np.max(self.fitness)
        
        if f_max == f_min:
            masses = np.ones(self.pop_size)
        else:
            # Better fitness = larger mass
            best_val = f_min
            worst_val = f_max
            q = (self.fitness - worst_val) / (best_val - worst_val + 1e-15)
            masses = q / np.sum(q)

        # 3. Calculate Acceleration
        acceleration = np.zeros((self.pop_size, self.dim))
        
        # Kbest: Number of agents that exert force (decreases over time)
        # Standard GSA logic: start with all agents, decrease to 1
        kbest = int(self.pop_size * (1 - self.current_iter / self.max_iter))
        kbest = max(kbest, 1)
        sorted_idx = np.argsort(masses)[::-1] # Indices of heaviest masses
        
        for i in range(self.pop_size):
            force = np.zeros(self.dim)
            for idx in range(kbest):
                j = sorted_idx[idx]
                if i != j:
                    dist = np.linalg.norm(self.population[i] - self.population[j]) + 1e-15
                    # Newton's Law of Gravity: F = G * (Mi * Mj) / R
                    # Note: Original GSA uses R instead of R^2 for better performance
                    force += np.random.rand() * g * (masses[i] * masses[j] / dist) * (self.population[j] - self.population[i])
            
            # Acceleration: a = F / m
            acceleration[i] = force / (masses[i] + 1e-15)

        # 4. Update Velocity and Position
        self.velocity = np.random.rand() * self.velocity + acceleration
        self.population = self._clip_bounds(self.population + self.velocity)
        
        # 5. Evaluate and Update Best
        self.fitness = np.array([obj_func(ind) for ind in self.population])
        
        idx_best = np.argmin(self.fitness)
        if self.fitness[idx_best] < self.best_fitness:
            self.best_fitness = self.fitness[idx_best]
            self.best_solution = self.population[idx_best].copy()
            return True
            
        return False