import numpy as np
from .base import BaseSolver

class CMAES(BaseSolver):
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
    
    A state-of-the-art derivative-free optimization algorithm for non-linear, 
    non-convex continuous problems. It adapts the multivariate normal 
    distribution (mean and covariance matrix) to capture the landscape 
    of the objective function.
    """

    def __init__(self, bounds, pop_size=10, max_iter=100, sigma=0.3, stop_patience=100):
        """
        Args:
            bounds (tuple): (lower_bounds, upper_bounds).
            pop_size (int): Number of individuals (lambda) sampled per generation.
            max_iter (int): Maximum number of generations.
            sigma (float): Initial step-size (standard deviation), usually 
                           scaled to ~30% of the search range.
            stop_patience (int): Iterations to wait before stagnation cutoff.
        """

        super().__init__("CMAES", pop_size, max_iter, bounds, stop_patience)
        
        # Hyperparameters
        self.mu = int(np.floor(self.pop_size / 2))
        
        # Strategy weights
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = weights / np.sum(weights)
        self.mu_eff = 1.0 / np.sum(self.w**2)
        
        # Adaptation constants
        self.cc = (4 + self.mu_eff/self.dim) / (self.dim + 4 + 2*self.mu_eff/self.dim)
        self.cs = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1/self.mu_eff) / ((self.dim + 2)**2 + self.mu_eff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1) + self.cs
        
        # Expected value of the norm of a N(0,I) vector
        self.chiN = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
        
        # Initial search state
        self.initial_sigma = sigma
        self.reset_state()

    def reset_state(self):
        r"""
        Initializes or resets the internal evolution state variables.
        
        This includes the mean vector, evolution paths ($p_c, p_s$), 
        the covariance matrix ($C$), and the step-size ($\sigma$).
        """
        
        self.m = (self.lb + self.ub) / 2.0
        self.sigma = self.initial_sigma
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.C = np.eye(self.dim)
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        self.inv_sqrtC = np.eye(self.dim)
        self.current_iter = 0

    def init_solver(self, obj_func):
        self.reset_state()
        # Initial evaluation of the mean
        self.best_fitness = obj_func(self.m)
        self.best_solution = self.m.copy()

    def step(self, obj_func):
        self.current_iter += 1
        
        # 1. Sample new candidates
        BD = self.B @ np.diag(self.D)
        arz = np.random.randn(self.pop_size, self.dim)
        ary = arz @ BD.T
        Xcand = self.m + self.sigma * ary
        
        # 2. Boundary handling (with resampling/clipping)
        for i in range(self.pop_size):
            tries = 0
            while np.any(Xcand[i] < self.lb) or np.any(Xcand[i] > self.ub):
                arz[i] = np.random.randn(self.dim)
                ary[i] = arz[i] @ BD.T
                Xcand[i] = self.m + self.sigma * ary[i]
                tries += 1
                if tries > 20:
                    Xcand[i] = np.clip(Xcand[i], self.lb, self.ub)
                    break

        # 3. Evaluation and Sorting
        fitness_values = np.array([obj_func(x) for x in Xcand])
        indices = np.argsort(fitness_values)
        
        Xcand_sorted = Xcand[indices]
        arz_sorted = arz[indices]
        
        # Update global best
        if fitness_values[indices[0]] < self.best_fitness:
            self.best_fitness = fitness_values[indices[0]]
            self.best_solution = Xcand_sorted[0].copy()
            improved = True
        else:
            improved = False

        # 4. Selection and Adaptation
        m_old = self.m.copy()
        # Update mean
        self.m = self.w @ Xcand_sorted[:self.mu]
        
        # Evolution paths
        ymean = (self.m - m_old) / self.sigma
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * (ymean @ self.inv_sqrtC)
        
        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * self.current_iter)) / self.chiN < (1.4 + 2 / (self.dim + 1))
        
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * ymean
        
        # Update Covariance Matrix C
        artmp = (Xcand_sorted[:self.mu] - m_old) / self.sigma
        rank_mu = (artmp.T * self.w) @ artmp
        
        self.C = ((1 - self.c1 - self.cmu) * self.C + 
                  self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + 
                  self.cmu * rank_mu)
        
        # Ensure symmetry and decompose
        self.C = (self.C + self.C.T) / 2
        
        # 5. Step size control (sigma)
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        
        # 6. Update B and D for next iteration
        # eig returns eigenvalues and eigenvectors
        evals, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(evals, 1e-30))
        self.inv_sqrtC = self.B @ np.diag(1.0 / self.D) @ self.B.T
        
        return improved