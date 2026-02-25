import numpy as np
import math
from .base import BaseSolver

class HHO(BaseSolver):
    """
    Harris Hawks Optimization (HHO).
    
    A population-based metaheuristic that simulates the cooperative hunting 
    strategies of Harris's hawks. The algorithm features a dynamic transition 
    between exploration and exploitation based on the escaping energy of the 
    prey, utilizing "Surprise Pounce" maneuvers and Levy flight-based rapid dives.
    """

    def __init__(self, bounds, pop_size=50, max_iter=100, stop_patience=100):
        """
        Args:
            bounds (tuple): (lower_bounds, upper_bounds).
            pop_size (int): Total number of hawks in the population.
            max_iter (int): Maximum iterations (used to calculate prey energy decay).
            stop_patience (int): Iterations to wait before stagnation cutoff.
        """

        super().__init__("HHO", pop_size, max_iter, bounds, stop_patience)
        
        # Internal state
        self.current_iter = 0

    def init_solver(self, obj_func):
        self.current_iter = 0
        self.population = self._initialize_population()
        self.fitness = np.array([obj_func(ind) for ind in self.population])
        
        idx_best = np.argmin(self.fitness)
        self.best_fitness = self.fitness[idx_best]
        self.best_solution = self.population[idx_best].copy()

    def step(self, obj_func):
        self.current_iter += 1
        
        # 1. Update Prey Energy (E) - Decreases from 2 to 0
        # E0 is the initial energy in [-1, 1]
        E0 = 2 * np.random.rand() - 1
        # E is the escaping energy of the prey
        E = 2 * E0 * (1 - (self.current_iter / self.max_iter))
        
        new_population = np.copy(self.population)

        for i in range(self.pop_size):
            # --- Exploration Phase (|E| >= 1) ---
            if np.abs(E) >= 1:
                q = np.random.rand()
                if q < 0.5:
                    # Perch based on random hawk
                    rand_idx = np.random.randint(self.pop_size)
                    X_rand = self.population[rand_idx]
                    new_population[i] = X_rand - np.random.rand() * np.abs(X_rand - 2 * np.random.rand() * self.population[i])
                else:
                    # Perch on random tall tree (within bounds)
                    X_mean = np.mean(self.population, axis=0)
                    new_population[i] = (self.best_solution - X_mean) - np.random.rand() * ((self.ub - self.lb) * np.random.rand() + self.lb)

            # --- Exploitation Phase (|E| < 1) ---
            else:
                r = np.random.rand()
                
                # Case 1: Soft Besiege (r >= 0.5, |E| >= 0.5)
                if r >= 0.5 and np.abs(E) >= 0.5:
                    jump_strength = 2 * (1 - np.random.rand())
                    dist = self.best_solution - self.population[i]
                    new_population[i] = dist - E * np.abs(jump_strength * self.best_solution - self.population[i])
                
                # Case 2: Hard Besiege (r >= 0.5, |E| < 0.5)
                elif r >= 0.5 and np.abs(E) < 0.5:
                    dist = self.best_solution - self.population[i]
                    new_population[i] = self.best_solution - E * np.abs(dist)
                
                # Case 3: Soft Besiege with progressive rapid dives (r < 0.5, |E| >= 0.5)
                elif r < 0.5 and np.abs(E) >= 0.5:
                    new_population[i] = self._rapid_dive(obj_func, i, E, soft=True)
                
                # Case 4: Hard Besiege with progressive rapid dives (r < 0.5, |E| < 0.5)
                elif r < 0.5 and np.abs(E) < 0.5:
                    new_population[i] = self._rapid_dive(obj_func, i, E, soft=False)

        # Update and Evaluate
        self.population = self._clip_bounds(new_population)
        self.fitness = np.array([obj_func(ind) for ind in self.population])
        
        idx_best = np.argmin(self.fitness)
        if self.fitness[idx_best] < self.best_fitness:
            self.best_fitness = self.fitness[idx_best]
            self.best_solution = self.population[idx_best].copy()
            return True
            
        return False

    def _rapid_dive(self, obj_func, i, E, soft=True):
        """
        Executes a rapid-dive maneuver using Levy flight distribution.

        This specialized movement mimics the zigzag behavior of hawks during 
        a chase. It evaluates two candidate positions (Y and Z) and selects 
         the best one via greedy selection.

        Args:
            obj_func (callable): The objective function to minimize.
            i (int): Index of the current hawk.
            E (float): Escaping energy of the prey.
            soft (bool): If True, performs a 'Soft Besiege'; otherwise, performs 
                         a 'Hard Besiege' relative to the population mean.

        Returns:
            np.ndarray: The best resulting position vector from the dive.
        """
        
        beta = 1.5
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        L = 0.01 * u / (np.abs(v)**(1 / beta))
        
        jump_strength = 2 * (1 - np.random.rand())
        
        # Calculate Y
        if soft:
            Y = self.best_solution - E * np.abs(jump_strength * self.best_solution - self.population[i])
        else:
            X_mean = np.mean(self.population, axis=0)
            Y = self.best_solution - E * np.abs(jump_strength * self.best_solution - X_mean)
        
        Y = self._clip_bounds(Y)
        
        # Calculate Z
        Z = Y + np.random.randn(self.dim) * L
        Z = self._clip_bounds(Z)
        
        # Greedy selection between Y and Z
        if obj_func(Y) < self.fitness[i]:
            return Y
        if obj_func(Z) < self.fitness[i]:
            return Z
        
        return self.population[i]