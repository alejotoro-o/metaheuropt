import numpy as np
from metaheuropt.solvers import GA

def test_ga_initialization(standard_bounds):
    """Test if GA initializes parameters correctly."""
    ga = GA(bounds=standard_bounds, pop_size=20)
    assert ga.dim == 3
    assert ga.pm == 1.0 / 3
    assert ga.name == "GA"

def test_ga_lifecycle_step(sphere_func, standard_bounds):
    """Test the new init_solver and step architecture."""
    ga = GA(bounds=standard_bounds, pop_size=20, max_iter=10)
    
    # 1. Test Initialization
    ga.init_solver(sphere_func)
    assert ga.population.shape == (20, 3)
    assert ga.fitness.shape == (20,)
    assert ga.best_fitness != np.inf
    
    # 2. Test a single step
    initial_best = ga.best_fitness
    ga.step(sphere_func)
    
    # Best fitness should be equal or better (never worse)
    assert ga.best_fitness <= initial_best
    assert ga.population.shape == (20, 3)

def test_ga_bounds_enforcement(sphere_func, standard_bounds):
    """Ensure population never escape the search space after steps."""
    lb, ub = standard_bounds
    ga = GA(bounds=standard_bounds, pop_size=10, max_iter=5)
    
    ga.init_solver(sphere_func)
    for _ in range(5):
        ga.step(sphere_func)
    
    # Check current population and best solution
    assert np.all(ga.population >= lb)
    assert np.all(ga.population <= ub)
    assert np.all(ga.best_solution >= lb)
    assert np.all(ga.best_solution <= ub)

def test_ga_improvement_flag(sphere_func, standard_bounds):
    """Verify the step method returns a boolean indicating improvement."""
    ga = GA(bounds=standard_bounds, pop_size=50)
    ga.init_solver(sphere_func)
    
    # We can't guarantee an improvement in one step, 
    # but we can check if the returned type is boolean.
    improved = ga.step(sphere_func)
    assert isinstance(improved, bool)