import numpy as np
from .base import BaseSolver

class GA(BaseSolver):
    def __init__(self, bounds, pop_size=50, max_iter=100, pc=0.9, eta_c=15, eta_m=20, stop_patience=100):
        super().__init__("GA", pop_size, max_iter, bounds, stop_patience)
        self.pc, self.eta_c, self.eta_m = pc, eta_c, eta_m
        self.pm = 1.0 / self.dim

    def init_solver(self, obj_func):
        self.population = self._initialize_population()
        self.fitness = np.array([obj_func(ind) for ind in self.population])
        idx = np.argmin(self.fitness)
        self.best_fitness = self.fitness[idx]
        self.best_solution = self.population[idx].copy()

    def step(self, obj_func):
        # 1. Selection
        idx1 = np.random.randint(0, self.pop_size, self.pop_size)
        idx2 = np.random.randint(0, self.pop_size, self.pop_size)
        winners = np.where(self.fitness[idx1] < self.fitness[idx2], idx1, idx2)
        mating_pool = self.population[winners].copy()

        # 2. Crossover
        offspring = mating_pool.copy()
        for i in range(0, self.pop_size - 1, 2):
            if np.random.rand() < self.pc:
                u = np.random.rand(self.dim)
                beta = np.where(u <= 0.5, (2*u)**(1/(self.eta_c+1)), (2*(1-u))**(-1/(self.eta_c+1)))
                offspring[i] = 0.5*((1+beta)*mating_pool[i] + (1-beta)*mating_pool[i+1])
                offspring[i+1] = 0.5*((1-beta)*mating_pool[i] + (1+beta)*mating_pool[i+1])

        # 3. Mutation & Clip
        for i in range(self.pop_size):
            offspring[i] = self._poly_mutation(offspring[i])
        offspring = self._clip_bounds(offspring)

        # 4. Evaluation & Elitism
        f_off = np.array([obj_func(ind) for ind in offspring])
        comb_p = np.vstack((self.population, offspring))
        comb_f = np.hstack((self.fitness, f_off))
        
        sort_idx = np.argsort(comb_f)[:self.pop_size]
        self.population, self.fitness = comb_p[sort_idx], comb_f[sort_idx]

        # Update best
        if self.fitness[0] < self.best_fitness:
            self.best_fitness = self.fitness[0]
            self.best_solution = self.population[0].copy()
            return True # Improved
        return False # No improvement

    def _poly_mutation(self, x):
        y = x.copy()
        for j in range(self.dim):
            if np.random.rand() < self.pm:
                d1, d2 = (x[j]-self.lb[j])/(self.ub[j]-self.lb[j]), (self.ub[j]-x[j])/(self.ub[j]-self.lb[j])
                u, mut_pow = np.random.rand(), 1/(self.eta_m+1)
                dq = (2*u + (1-2*u)*(1-d1)**(self.eta_m+1))**mut_pow - 1 if u <= 0.5 else 1 - (2*(1-u) + 2*(u-0.5)*(1-d2)**(self.eta_m+1))**mut_pow
                y[j] += dq * (self.ub[j] - self.lb[j])
        return y