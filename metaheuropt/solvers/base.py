from abc import ABC, abstractmethod
import numpy as np

class BaseSolver(ABC):
    def __init__(self, name, pop_size, max_iter, bounds, stop_patience=None):
        self.name = name
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.stop_patience = stop_patience
        self.lb, self.ub = np.array(bounds[0]), np.array(bounds[1])
        self.dim = len(self.lb)
        
        # State variables
        self.population = None
        self.fitness = None
        self.best_solution = None
        self.best_fitness = np.inf

    @abstractmethod
    def init_solver(self, obj_func):
        """Initialize population and first evaluation."""
        pass

    @abstractmethod
    def step(self, obj_func):
        """Perform one iteration of the algorithm."""
        pass

    def _initialize_population(self):
        return self.lb + (self.ub - self.lb) * np.random.rand(self.pop_size, self.dim)

    def _clip_bounds(self, pop):
        return np.clip(pop, self.lb, self.ub)