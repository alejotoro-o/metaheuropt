import numpy as np
from metaheuropt.solvers import ABC

def test_abc_initialization(standard_bounds):
    """Test if ABC initializes parameters correctly."""
    pop_size = 20
    limit = 15
    abc = ABC(bounds=standard_bounds, pop_size=pop_size, limit=limit)
    
    assert abc.name == "ABC"
    assert abc.pop_size == pop_size
    assert abc.limit == limit
    assert abc.dim == 3

def test_abc_lifecycle_step(sphere_func, standard_bounds):
    """Test the ABC initialization and step execution."""
    pop_size = 20
    abc = ABC(bounds=standard_bounds, pop_size=pop_size, max_iter=10)
    
    # 1. Test Initialization
    abc.init_solver(sphere_func)
    assert abc.population.shape == (pop_size, 3)
    assert abc.fitness.shape == (pop_size,)
    assert abc.trials.shape == (pop_size,)
    assert np.all(abc.trials == 0)
    assert abc.best_fitness != np.inf
    
    # 2. Test a single step
    initial_best = abc.best_fitness
    improved = abc.step(sphere_func)
    
    # Check states after step
    assert isinstance(improved, bool)
    assert abc.best_fitness <= initial_best
    assert abc.population.shape == (pop_size, 3)

def test_abc_scout_mechanism(sphere_func, standard_bounds):
    """Verify that the Scout Bee mechanism replaces exhausted food sources."""
    # Create an ABC instance with a very low limit to trigger Scouts quickly
    limit = 1
    abc = ABC(bounds=standard_bounds, pop_size=5, limit=limit)
    abc.init_solver(sphere_func)
    
    # Manually force the trial counter to exceed the limit for the first bee
    abc.trials[0] = 5 
    original_position = abc.population[0].copy()
    
    # Perform a step; the scout should abandon population[0]
    abc.step(sphere_func)
    
    # The trial counter should have reset to 0
    assert abc.trials[0] == 0
    # The position should have changed (highly likely due to random re-init)
    assert not np.array_equal(abc.population[0], original_position)

def test_abc_probability_logic(sphere_func, standard_bounds):
    """Ensure the probability calculation for Onlooker Bees is valid."""
    abc = ABC(bounds=standard_bounds, pop_size=10)
    abc.init_solver(sphere_func)
    
    probs = abc._calculate_probabilities()
    
    # Probabilities should sum to 1.0 (or very close)
    assert np.isclose(np.sum(probs), 1.0)
    assert np.all(probs >= 0)
    
    # In ABC, lower fitness (better) should result in higher probability
    best_idx = np.argmin(abc.fitness)
    worst_idx = np.argmax(abc.fitness)
    
    # Only assert if there's actually a difference in fitness to avoid noise
    if abc.fitness[best_idx] < abc.fitness[worst_idx]:
        assert probs[best_idx] > probs[worst_idx]

def test_abc_bounds_enforcement(sphere_func, standard_bounds):
    """Ensure population stays within bounds after multiple steps."""
    lb, ub = standard_bounds
    abc = ABC(bounds=standard_bounds, pop_size=15, max_iter=20)
    abc.init_solver(sphere_func)
    
    for _ in range(10):
        abc.step(sphere_func)
    
    assert np.all(abc.population >= lb)
    assert np.all(abc.population <= ub)
    assert np.all(abc.best_solution >= lb)
    assert np.all(abc.best_solution <= ub)