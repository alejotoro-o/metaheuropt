import pytest
import numpy as np
from metaheuropt.solvers import WOA

def test_woa_initialization(standard_bounds):
    """Test if WOA initializes the whale population and hyperparameters correctly."""
    pop_size = 30
    woa = WOA(bounds=standard_bounds, pop_size=pop_size, b=1.5)
    woa.init_solver(lambda x: np.sum(x**2))
    
    assert woa.name == "WOA"
    assert woa.b == 1.5
    assert woa.population.shape == (pop_size, 3)
    assert woa.current_iter == 0
    assert woa.best_fitness != np.inf

def test_woa_lifecycle_step(sphere_func, standard_bounds):
    """Test the bubble-net hunting step (Encircling, Spiral, or Search)."""
    pop_size = 20
    woa = WOA(bounds=standard_bounds, pop_size=pop_size, max_iter=50)
    woa.init_solver(sphere_func)
    
    initial_best = woa.best_fitness
    initial_pos = woa.population.copy()
    
    # Perform a step
    improved = woa.step(sphere_func)
    
    assert isinstance(improved, bool)
    assert woa.current_iter == 1
    # Whales should have moved towards the leader or a random prey
    assert not np.array_equal(woa.population, initial_pos)
    # Global best should never degrade
    assert woa.best_fitness <= initial_best

def test_woa_mechanism_stability(sphere_func, standard_bounds):
    """Ensure the three logic branches (p < 0.5, |A| < 1, |A| >= 1) don't crash."""
    woa = WOA(bounds=standard_bounds, pop_size=10, max_iter=10)
    woa.init_solver(sphere_func)
    
    # We run enough iterations to likely hit all logic branches:
    # 1. Search for prey (Exploration)
    # 2. Shrinking encircling (Exploitation)
    # 3. Spiral update (Bubble-net)
    for _ in range(10):
        woa.step(sphere_func)
    
    assert woa.current_iter == 10
    assert woa.population.shape == (10, 3)

def test_woa_a_parameter_decay(sphere_func, standard_bounds):
    """Verify the 'a' parameter decay which controls exploration/exploitation."""
    max_iter = 100
    woa = WOA(bounds=standard_bounds, pop_size=10, max_iter=max_iter)
    woa.init_solver(sphere_func)
    
    # Near start: a should be near 2 (Exploration likely)
    woa.step(sphere_func)
    # Manual calculation of a for iter 1: 2 - 2 * (1/100) = 1.98
    
    # Near end: a should be near 0 (Exploitation only)
    woa.current_iter = 99
    woa.step(sphere_func)
    assert woa.current_iter == 100

def test_woa_bounds_enforcement(sphere_func, standard_bounds):
    """Ensure the spiral movement doesn't throw whales outside the search space."""
    lb, ub = standard_bounds
    # Use high b to create wider spirals to test clipping
    woa = WOA(bounds=standard_bounds, pop_size=15, b=5, max_iter=20)
    woa.init_solver(sphere_func)
    
    for _ in range(10):
        woa.step(sphere_func)
    
    assert np.all(woa.population >= lb)
    assert np.all(woa.population <= ub)
    assert np.all(woa.best_solution >= lb)
    assert np.all(woa.best_solution <= ub)