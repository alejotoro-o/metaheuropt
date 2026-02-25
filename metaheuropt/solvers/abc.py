import numpy as np
from .base import BaseSolver

class ABC(BaseSolver):
    """
    Artificial Bee Colony (ABC) Optimization.
    
    Mimics the foraging behavior of honey bees. The population is divided into 
    Employed Bees (local search), Onlooker Bees (probability-based search), 
    and Scout Bees (re-initialization of exhausted food sources).
    """

    def __init__(self, bounds, pop_size=50, max_iter=100, limit=20, stop_patience=100):
        """
        Args:
            bounds (tuple): (lower_bounds, upper_bounds).
            pop_size (int): Number of food sources (bees).
            max_iter (int): Maximum foraging cycles.
            limit (int): Stagnation limit; trials allowed before a food source 
                         is abandoned and a bee becomes a scout.
            stop_patience (int): Iterations to wait before global stagnation cutoff.
        """

        super().__init__("ABC", pop_size, max_iter, bounds, stop_patience)
        
        # Hyperparameters
        # limit: how many trials a food source can last without improvement before being abandoned
        self.limit = limit 
        
        # State variables
        self.trials = None # Counter for stagnation of each food source
        self.current_iter = 0

    def init_solver(self, obj_func):
        self.current_iter = 0
        self.population = self._initialize_population()
        self.fitness = np.array([obj_func(ind) for ind in self.population])
        self.trials = np.zeros(self.pop_size)
        
        idx_best = np.argmin(self.fitness)
        self.best_fitness = self.fitness[idx_best]
        self.best_solution = self.population[idx_best].copy()

    def step(self, obj_func):
        self.current_iter += 1
        improved_global = False

        # 1. Employed Bees Phase
        # Every employed bee explores a neighbor of its current food source
        for i in range(self.pop_size):
            self._explore_neighborhood(obj_func, i)

        # 2. Onlooker Bees Phase
        # Probabilistic selection: better sources attract more onlookers
        # Convert fitness to 'profitability' (minimization)
        probs = self._calculate_probabilities()
        
        for _ in range(self.pop_size):
            # Select a source based on probability (Roulette Wheel)
            i = np.random.choice(self.pop_size, p=probs)
            self._explore_neighborhood(obj_func, i)

        # 3. Scout Bees Phase
        # If a food source has exceeded the limit, abandon it and find a new random one
        for i in range(self.pop_size):
            if self.trials[i] > self.limit:
                self.population[i] = self._initialize_population()[0] # New random source
                self.fitness[i] = obj_func(self.population[i])
                self.trials[i] = 0
        
        # Update Global Best
        idx_best = np.argmin(self.fitness)
        if self.fitness[idx_best] < self.best_fitness:
            self.best_fitness = self.fitness[idx_best]
            self.best_solution = self.population[idx_best].copy()
            improved_global = True

        return improved_global

    def _explore_neighborhood(self, obj_func, i):
        """
        Performs a local search around a specific food source.

        Uses a random partner 'k' and a random dimension 'j' to generate 
        a candidate solution. Implements greedy selection between the 
        neighbor and the original food source.

        Args:
            obj_func (callable): The objective function to minimize.
            i (int): Index of the bee/food source to perturb.
        """

        # Choose a random dimension and a random partner k
        j = np.random.randint(self.dim)
        k = i
        while k == i:
            k = np.random.randint(self.pop_size)
        
        phi = np.random.uniform(-1, 1)
        v = self.population[i].copy()
        v[j] = v[j] + phi * (v[j] - self.population[k][j])
        v = np.clip(v, self.lb[j], self.ub[j])
        
        f_v = obj_func(v)
        
        # Greedy Selection
        if f_v < self.fitness[i]:
            self.population[i] = v
            self.fitness[i] = f_v
            self.trials[i] = 0
        else:
            self.trials[i] += 1

    def _calculate_probabilities(self):
        """
        Computes fitness-based selection probabilities for Onlooker Bees.

        Applies a mapping to the raw fitness values so that lower objective 
        values result in higher selection probabilities.

        Returns:
            np.ndarray: Probability distribution for the population.
        """
        
        # For minimization, we transform fitness so that smaller values have higher probs
        # Standard ABC uses: 1/(1+fit) if fit >= 0
        fitness_mapped = np.where(self.fitness >= 0, 
                                  1 / (1 + self.fitness), 
                                  1 + np.abs(self.fitness))
        return fitness_mapped / np.sum(fitness_mapped)