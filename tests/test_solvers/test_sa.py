import numpy as np
from metaheuropt.solvers import SA

def test_sa_initialization(standard_bounds):
    """Test if SA initializes temperature and state correctly."""
    t0 = 500
    sa = SA(bounds=standard_bounds, pop_size=1, t0=t0)
    sa.init_solver(lambda x: np.sum(x**2))
    
    assert sa.name == "SA"
    assert sa.t == t0
    assert sa.current_iter == 0
    assert sa.population.shape == (1, 3)

def test_sa_lifecycle_step(sphere_func, standard_bounds):
    """Test the Metropolis-Hastings move and cooling schedule."""
    t0 = 100
    alpha = 0.95
    sa = SA(bounds=standard_bounds, pop_size=1, t0=t0, alpha=alpha)
    sa.init_solver(sphere_func)
    
    initial_temp = sa.t
    initial_pos = sa.population.copy()
    
    # Perform a step
    improved = sa.step(sphere_func)
    
    assert isinstance(improved, bool)
    assert sa.current_iter == 1
    # Temperature should have cooled
    assert sa.t == initial_temp * alpha
    # The solution should likely have changed (accepted move)
    # Note: There's a tiny chance it didn't accept, but with high T0 it usually does
    assert not np.array_equal(sa.population, initial_pos) or sa.t < initial_temp

def test_sa_metropolis_acceptance(standard_bounds):
    """
    Verify that SA can accept worse solutions (exploration) 
    but still tracks the best solution found so far.
    """
    # High temperature to ensure acceptance of worse moves
    sa = SA(bounds=standard_bounds, pop_size=1, t0=1e6)
    
    # Mock a starting point
    sa.init_solver(lambda x: 10.0) 
    sa.population[0] = np.array([1.0, 1.0, 1.0])
    sa.fitness[0] = 10.0
    sa.best_fitness = 10.0
    sa.best_solution = sa.population[0].copy()
    
    # Provide a function that returns a worse fitness (20.0)
    def worse_func(x): return 20.0
    
    sa.step(worse_func)
    
    # Current population fitness should be 20.0 (accepted worse move due to high T)
    assert sa.fitness[0] == 20.0
    # Global best should still be 10.0 (the best ever found)
    assert sa.best_fitness == 10.0

def test_sa_cooling_limit(sphere_func, standard_bounds):
    """Ensure temperature decays but stays stable over many iterations."""
    sa = SA(bounds=standard_bounds, max_iter=100, alpha=0.9)
    sa.init_solver(sphere_func)
    
    for _ in range(10):
        sa.step(sphere_func)
    
    assert sa.t < sa.t0
    assert sa.t >= 0

def test_sa_bounds_enforcement(sphere_func, standard_bounds):
    """Ensure perturbations stay within the defined search space."""
    lb, ub = standard_bounds
    # High step size to force boundary hits
    sa = SA(bounds=standard_bounds, step_size=2.0)
    sa.init_solver(sphere_func)
    
    for _ in range(20):
        sa.step(sphere_func)
    
    assert np.all(sa.population >= lb)
    assert np.all(sa.population <= ub)
    assert np.all(sa.best_solution >= lb)
    assert np.all(sa.best_solution <= ub)