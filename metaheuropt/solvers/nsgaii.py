import numpy as np
from .base import BaseSolver

class NSGAII(BaseSolver):
    def __init__(self, bounds, pop_size=50, max_iter=100, pc=0.9, eta_c=15, 
                 pm=None, eta_m=20, stop_patience=100):
        super().__init__("NSGAII", pop_size, max_iter, bounds, stop_patience)
        
        # Hyperparameters
        self.pc = pc
        self.eta_c = eta_c
        self.pm = pm if pm is not None else 1.0 / self.dim
        self.eta_m = eta_m
        
        self.current_iter = 0

    def _sbx_crossover(self, p1, p2):
        """Simulated Binary Crossover (SBX)."""
        u = np.random.rand(self.dim)
        beta = np.zeros(self.dim)
        
        # Logic for beta calculation
        mask_low = u <= 0.5
        beta[mask_low] = (2 * u[mask_low])**(1.0 / (self.eta_c + 1))
        beta[~mask_low] = (1.0 / (2 * (1 - u[~mask_low])))**(1.0 / (self.eta_c + 1))
        
        c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        
        return self._clip_bounds(c1), self._clip_bounds(c2)

    def _polynomial_mutation(self, x):
        """Polynomial Mutation."""
        y = x.copy()
        for j in range(self.dim):
            if np.random.rand() < self.pm:
                delta1 = (x[j] - self.lb[j]) / (self.ub[j] - self.lb[j] + 1e-15)
                delta2 = (self.ub[j] - x[j]) / (self.ub[j] - self.lb[j] + 1e-15)
                
                u = np.random.rand()
                mut_pow = 1.0 / (self.eta_m + 1)
                
                if u <= 0.5:
                    xy = 1.0 - delta1
                    val = 2 * u + (1 - 2 * u) * (xy**(self.eta_m + 1))
                    delta_q = val**mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2 * (1 - u) + 2 * (u - 0.5) * (xy**(self.eta_m + 1))
                    delta_q = 1.0 - val**mut_pow
                
                y[j] = x[j] + delta_q * (self.ub[j] - self.lb[j])
        
        return self._clip_bounds(y)

    def init_solver(self, obj_func):
        self.current_iter = 0
        self.population = self._initialize_population()
        self.fitness = np.array([obj_func(ind) for ind in self.population])
        
        idx_best = np.argmin(self.fitness)
        self.best_fitness = self.fitness[idx_best]
        self.best_solution = self.population[idx_best].copy()

    def step(self, obj_func):
        self.current_iter += 1
        
        # 1. Binary Tournament Selection
        m_pop = np.zeros_like(self.population)
        for k in range(self.pop_size):
            i1, i2 = np.random.randint(0, self.pop_size, 2)
            m_pop[k] = self.population[i1] if self.fitness[i1] < self.fitness[i2] else self.population[i2]
            
        # 2. SBX Crossover
        offspring = m_pop.copy()
        for k in range(0, self.pop_size - 1, 2):
            if np.random.rand() < self.pc:
                offspring[k], offspring[k+1] = self._sbx_crossover(m_pop[k], m_pop[k+1])
                
        # 3. Mutation
        for k in range(self.pop_size):
            offspring[k] = self._polynomial_mutation(offspring[k])
            
        # 4. Evaluate Offspring
        f_off = np.array([obj_func(ind) for ind in offspring])
        
        # 5. Elitist Selection (Mu + Lambda)
        # Combine Parent and Offspring populations
        u_pop = np.vstack([self.population, offspring])
        u_fit = np.concatenate([self.fitness, f_off])
        
        # Sort and take the best Ni
        sorted_indices = np.argsort(u_fit)
        self.population = u_pop[sorted_indices[:self.pop_size]]
        self.fitness = u_fit[sorted_indices[:self.pop_size]]
        
        # Update Global Best
        if self.fitness[0] < self.best_fitness:
            self.best_fitness = self.fitness[0]
            self.best_solution = self.population[0].copy()
            return True
            
        return False