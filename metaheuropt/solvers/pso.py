import numpy as np
from .base import BaseSolver

class PSO(BaseSolver):
    """
    Particle Swarm Optimization (PSO).
    
    A population-based stochastic optimization technique inspired by the social 
    behavior of bird flocking or fish schooling. Particles move through the 
    search space by following their personal historical best positions and the 
    swarm's collective global best position, balancing momentum, individual 
    intelligence, and social influence.
    """

    def __init__(self, bounds, pop_size=50, max_iter=100, w_max=0.95, w_min=0.35, 
                 c1=1.4, c2=1.4, stop_patience=100):
        """
        Args:
            bounds (tuple): (lower_bounds, upper_bounds).
            pop_size (int): Number of particles in the swarm.
            max_iter (int): Maximum number of iterations (used for linear inertia weight decay).
            w_max (float): Initial (maximum) inertia weight, controlling exploration.
            w_min (float): Final (minimum) inertia weight, favoring exploitation.
            c1 (float): Cognitive coefficient; determines the pull toward a particle's personal best.
            c2 (float): Social coefficient; determines the pull toward the swarm's global best.
            stop_patience (int): Iterations to wait before stagnation cutoff.
        """
        
        super().__init__("PSO", pop_size, max_iter, bounds, stop_patience)
        
        # Hyperparameters
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        
        # Velocity limits (20% of the range as per your MATLAB script)
        self.v_max = 0.2 * (self.ub - self.lb)
        self.v_min = -self.v_max
        
        # State variables
        self.velocity = None
        self.pbest_pos = None
        self.pbest_val = None
        self.current_iter = 0

    def init_solver(self, obj_func):
        
        self.current_iter = 0
        self.population = self._initialize_population()
        self.velocity = np.zeros((self.pop_size, self.dim))
        
        # Initial evaluation
        self.fitness = np.array([obj_func(ind) for ind in self.population])
        
        # Initialize Personal Bests
        self.pbest_pos = self.population.copy()
        self.pbest_val = self.fitness.copy()
        
        # Initialize Global Best
        idx_best = np.argmin(self.pbest_val)
        self.best_fitness = self.pbest_val[idx_best]
        self.best_solution = self.pbest_pos[idx_best].copy()

    def step(self, obj_func):
        
        self.current_iter += 1
        
        # 1. Update Inertia Weight (Linear decay)
        w = self.w_max - (self.w_max - self.w_min) * (self.current_iter / self.max_iter)
        
        # 2. Generate Random Coefficients
        r1 = np.random.rand(self.pop_size, self.dim)
        r2 = np.random.rand(self.pop_size, self.dim)
        
        # 3. Update Velocity
        self.velocity = (w * self.velocity + 
                         self.c1 * r1 * (self.pbest_pos - self.population) + 
                         self.c2 * r2 * (self.best_solution - self.population))
        
        # Apply velocity limits
        self.velocity = np.clip(self.velocity, self.v_min, self.v_max)
        
        # 4. Update Position and Clip Bounds
        self.population = self._clip_bounds(self.population + self.velocity)
        
        # 5. Evaluation
        self.fitness = np.array([obj_func(ind) for ind in self.population])
        
        # 6. Update Personal Bests
        improved_mask = self.fitness < self.pbest_val
        self.pbest_pos[improved_mask] = self.population[improved_mask]
        self.pbest_val[improved_mask] = self.fitness[improved_mask]
        
        # 7. Update Global Best
        idx_best = np.argmin(self.pbest_val)
        if self.pbest_val[idx_best] < self.best_fitness:
            self.best_fitness = self.pbest_val[idx_best]
            self.best_solution = self.pbest_pos[idx_best].copy()
            return True # Improvement found
            
        return False # No improvement