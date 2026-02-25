import numpy as np
from .base import BaseSolver

class TLBO(BaseSolver):
    """
    Teaching-Learning Based Optimization (TLBO).
    
    A population-based algorithm inspired by the teaching-learning process in a 
    classroom. It does not require algorithm-specific hyperparameters, making 
    it highly robust. The search process is divided into the 'Teacher Phase', 
    where learners gain knowledge from the best individual (the teacher), and 
    the 'Learner Phase', where learners improve by interacting with their peers.
    """

    def __init__(self, bounds, pop_size=50, max_iter=100, stop_patience=100):
        """
        Args:
            bounds (tuple): (lower_bounds, upper_bounds).
            pop_size (int): Number of learners in the classroom.
            max_iter (int): Maximum number of teaching cycles.
            stop_patience (int): Iterations to wait before stagnation cutoff.
        """
        
        super().__init__("TLBO", pop_size, max_iter, bounds, stop_patience)
        
    def init_solver(self, obj_func):
        self.population = self._initialize_population()
        self.fitness = np.array([obj_func(ind) for ind in self.population])
        
        idx_best = np.argmin(self.fitness)
        self.best_fitness = self.fitness[idx_best]
        self.best_solution = self.population[idx_best].copy()

    def step(self, obj_func):
        improved_global = False
        
        for i in range(self.pop_size):
            # --- 1. Teacher Phase ---
            # The 'Teacher' is the best solution in the population
            teacher = self.best_solution
            mean_val = np.mean(self.population, axis=0)
            
            # Teaching factor (TF) is either 1 or 2 (randomly)
            tf = np.random.randint(1, 3)
            
            # Difference Mean: Hawks/Learners move toward the teacher
            diff_mean = np.random.rand(self.dim) * (teacher - tf * mean_val)
            
            new_sol = self.population[i] + diff_mean
            new_sol = self._clip_bounds(new_sol)
            new_fit = obj_func(new_sol)
            
            # Greedy Selection for Teacher Phase
            if new_fit < self.fitness[i]:
                self.population[i] = new_sol
                self.fitness[i] = new_fit

            # --- 2. Learner Phase ---
            # Learn by interacting with another random learner 'k'
            k = i
            while k == i:
                k = np.random.randint(self.pop_size)
            
            if self.fitness[i] < self.fitness[k]:
                new_sol = self.population[i] + np.random.rand(self.dim) * (self.population[i] - self.population[k])
            else:
                new_sol = self.population[i] + np.random.rand(self.dim) * (self.population[k] - self.population[i])
            
            new_sol = self._clip_bounds(new_sol)
            new_fit = obj_func(new_sol)
            
            # Greedy Selection for Learner Phase
            if new_fit < self.fitness[i]:
                self.population[i] = new_sol
                self.fitness[i] = new_fit

        # Update Global Best
        idx_best = np.argmin(self.fitness)
        if self.fitness[idx_best] < self.best_fitness:
            self.best_fitness = self.fitness[idx_best]
            self.best_solution = self.population[idx_best].copy()
            improved_global = True
            
        return improved_global