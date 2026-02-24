import numpy as np
from .base import BaseSolver

class JADE(BaseSolver):
    def __init__(self, bounds, pop_size=50, max_iter=100, p=0.1, c_adapt=0.1, stop_patience=100):
        super().__init__("JADE", pop_size, max_iter, bounds, stop_patience)
        
        # Hyperparameters
        self.p = p
        self.c_adapt = c_adapt
        
        # Adaptive parameters (means)
        self.mu_f = 0.5
        self.mu_cr = 0.5
        
        # State variables
        self.archive = []
        self.current_iter = 0

    def _lehmer_mean(self, values):
        """Calculates the Lehmer mean for the mutation factor F."""
        if not values:
            return 0
        arr = np.array(values)
        return np.sum(arr**2) / (np.sum(arr) + 1e-15)

    def _cauchy_pos(self, location, scale):
        """Generates a positive value from a Cauchy distribution."""
        val = location + scale * np.tan(np.pi * (np.random.rand() - 0.5))
        while val <= 0:
            val = location + scale * np.tan(np.pi * (np.random.rand() - 0.5))
        return min(val, 1.0)

    def init_solver(self, obj_func):
        """Initializes population and archive."""
        self.current_iter = 0
        self.mu_f = 0.5
        self.mu_cr = 0.5
        self.archive = []
        
        self.population = self._initialize_population()
        self.fitness = np.array([obj_func(ind) for ind in self.population])
        
        idx_best = np.argmin(self.fitness)
        self.best_fitness = self.fitness[idx_best]
        self.best_solution = self.population[idx_best].copy()

    def step(self, obj_func):
        """Performs one iteration of JADE adaptation and evolution."""
        self.current_iter += 1
        
        # 1. Prepare for current-to-pbest mutation
        sorted_indices = np.argsort(self.fitness)
        p_num = max(2, int(np.round(self.p * self.pop_size)))
        top_p_indices = sorted_indices[:p_num]
        
        s_f = []
        s_cr = []
        new_population = np.copy(self.population)
        new_fitness = np.copy(self.fitness)
        
        # Current Archive Pool: Population + External Archive
        pool = np.vstack([self.population] + self.archive) if self.archive else self.population
        pool_size = pool.shape[0]

        # 2. Evolve each individual
        for i in range(self.pop_size):
            # Generate Fi and CRi for this individual
            fi = self._cauchy_pos(self.mu_f, 0.1)
            cri = np.clip(np.random.normal(self.mu_cr, 0.1), 0, 1)
            
            # Select mutation components
            # xpbest: one of the top p% individuals
            xp_idx = np.random.choice(top_p_indices)
            xp_best = self.population[xp_idx]
            
            # xr1: random from population (distinct from i)
            r1 = i
            while r1 == i:
                r1 = np.random.randint(self.pop_size)
            xr1 = self.population[r1]
            
            # xr2: random from pool (distinct from i and r1)
            r2 = i
            while r2 == i or r2 == r1:
                r2 = np.random.randint(pool_size)
            xr2 = pool[r2]
            
            # Mutation: current-to-pbest/1
            vi = self.population[i] + fi * (xp_best - self.population[i]) + fi * (xr1 - xr2)
            
            # Crossover (Binomial)
            mask = np.random.rand(self.dim) < cri
            j_rand = np.random.randint(self.dim)
            mask[j_rand] = True # Ensure at least one dimension changes
            
            ui = np.where(mask, vi, self.population[i])
            ui = self._clip_bounds(ui)
            
            # Selection
            f_ui = obj_func(ui)
            if f_ui <= self.fitness[i]:
                new_population[i] = ui
                new_fitness[i] = f_ui
                s_f.append(fi)
                s_cr.append(cri)
                
                # Add the replaced parent to the archive
                self.archive.append(self.population[i].copy())

        # 3. Archive Maintenance
        if len(self.archive) > self.pop_size:
            # Randomly sub-sample to maintain size Ni
            indices = np.random.choice(len(self.archive), self.pop_size, replace=False)
            self.archive = [self.archive[idx] for idx in indices]

        # 4. Update adaptive means mu_f and mu_cr
        if s_f:
            self.mu_f = (1 - self.c_adapt) * self.mu_f + self.c_adapt * self._lehmer_mean(s_f)
        if s_cr:
            self.mu_cr = (1 - self.c_adapt) * self.mu_cr + self.c_adapt * np.mean(s_cr)

        # 5. Update State
        self.population = new_population
        self.fitness = new_fitness
        
        idx_best = np.argmin(self.fitness)
        if self.fitness[idx_best] < self.best_fitness:
            self.best_fitness = self.fitness[idx_best]
            self.best_solution = self.population[idx_best].copy()
            return True
            
        return False