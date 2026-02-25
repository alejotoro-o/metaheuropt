from abc import ABC, abstractmethod
import numpy as np

class BaseSolver(ABC):
    """
    Abstract Base Class for all metaheuristic optimization algorithms.

    This class defines the standard interface and common utility methods 
    required by all solvers in the package. It handles population initialization, 
    boundary enforcement, and core state tracking.

    Attributes:
        name (str): The identifier name of the algorithm (e.g., "PSO", "GWO").
        pop_size (int): The number of search agents (population size).
        max_iter (int): Maximum number of iterations allowed per run.
        stop_patience (int, optional): Number of iterations to wait for improvement 
                                       before early stopping.
        lb (np.ndarray): Lower bounds of the search space.
        ub (np.ndarray): Upper bounds of the search space.
        dim (int): Dimensionality of the optimization problem.
        population (np.ndarray): Current positions of all search agents.
        fitness (np.ndarray): Objective values for the current population.
        best_solution (np.ndarray): Best position vector found during the run.
        best_fitness (float): Best objective value found during the run.
    """

    def __init__(self, name, pop_size, max_iter, bounds, stop_patience=None):
        """
        Initializes the base solver with problem and algorithm parameters.

        Args:
            name (str): Algorithm name.
            pop_size (int): Number of individuals in the population.
            max_iter (int): Maximum iteration limit.
            bounds (tuple): A tuple containing (lower_bounds, upper_bounds).
            stop_patience (int, optional): Iterations to allow without improvement.
        """

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
        """
        Initializes the algorithm state before a run.

        Subclasses must implement this to handle algorithm-specific setup,
        initial population generation, and first fitness evaluations.

        Args:
            obj_func (callable): The objective function to minimize.
        """

        pass

    @abstractmethod
    def step(self, obj_func):
        """
        Performs a single iteration of the optimization algorithm.

        Subclasses must implement the logic for position updates, 
        parameter adaptation, and global best tracking.

        Args:
            obj_func (callable): The objective function to minimize.

        Returns:
            bool: True if a new global best was found during this step, 
                  False otherwise.
        """

        pass

    def _initialize_population(self):
        """
        Generates an initial population using uniform random distribution.

        Returns:
            np.ndarray: A matrix of shape (pop_size, dim) within [lb, ub].
        """

        return self.lb + (self.ub - self.lb) * np.random.rand(self.pop_size, self.dim)

    def _clip_bounds(self, pop):
        """
        Enforces search space boundaries by clipping out-of-bounds values.

        Args:
            pop (np.ndarray): Population matrix or individual vector to clip.

        Returns:
            np.ndarray: The clipped data within the range [self.lb, self.ub].
        """
        
        return np.clip(pop, self.lb, self.ub)