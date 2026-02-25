import numpy as np
from metaheuropt.solvers import JADE

def test_jade_initialization(standard_bounds):
    """Test if JADE initializes parameters and adaptive means correctly."""
    pop_size = 30
    jade = JADE(bounds=standard_bounds, pop_size=pop_size)
    jade.init_solver(lambda x: np.sum(x**2))
    
    assert jade.name == "JADE"
    assert jade.mu_f == 0.5
    assert jade.mu_cr == 0.5
    assert len(jade.archive) == 0
    assert jade.current_iter == 0

def test_jade_lifecycle_step(sphere_func, standard_bounds):
    """Test the JADE step including parameter adaptation and archive growth."""
    pop_size = 20
    jade = JADE(bounds=standard_bounds, pop_size=pop_size, max_iter=50)
    jade.init_solver(sphere_func)
    
    initial_mu_f = jade.mu_f
    initial_mu_cr = jade.mu_cr
    
    # Perform a step
    improved = jade.step(sphere_func)
    
    assert isinstance(improved, bool)
    assert jade.population.shape == (pop_size, 3)
    # The current_iter should increment
    assert jade.current_iter == 1
    # Best fitness should be monotonic
    assert jade.best_fitness <= np.min(jade.fitness)
    
    # The means should likely change if there were any successful individuals
    # (Though in 1 step it's not guaranteed, we check they remain valid)
    assert 0 <= jade.mu_f <= 1.0
    assert 0 <= jade.mu_cr <= 1.0

def test_jade_archive_management(sphere_func, standard_bounds):
    """Verify that the archive stays within the size limit (pop_size)."""
    pop_size = 10
    jade = JADE(bounds=standard_bounds, pop_size=pop_size)
    jade.init_solver(sphere_func)
    
    # Run multiple steps to force archive overflow
    # Replaced parents are added to the archive in every successful selection
    for _ in range(5):
        jade.step(sphere_func)
    
    # Archive should never exceed pop_size based on Step 3 of the algorithm
    assert len(jade.archive) <= pop_size

def test_jade_lehmer_mean():
    """Test the Lehmer mean helper function specifically."""
    jade = JADE(bounds=(np.array([0]), np.array([1])), pop_size=10)
    
    # Test empty list
    assert jade._lehmer_mean([]) == 0
    
    # Test values (e.g., 2 and 4)
    # Lehmer mean L(2,4) = (2^2 + 4^2) / (2 + 4) = 20 / 6 = 3.333...
    values = [2.0, 4.0]
    expected = (2.0**2 + 4.0**2) / (2.0 + 4.0)
    assert np.isclose(jade._lehmer_mean(values), expected)

def test_jade_cauchy_logic():
    """Ensure Cauchy generator produces positive values within [0, 1] range."""
    jade = JADE(bounds=(np.array([0]), np.array([1])), pop_size=10)
    
    for _ in range(50):
        val = jade._cauchy_pos(0.5, 0.1)
        assert val > 0
        assert val <= 1.0

def test_jade_bounds_enforcement(sphere_func, standard_bounds):
    """Ensure JADE mutation/crossover respects boundary constraints."""
    lb, ub = standard_bounds
    jade = JADE(bounds=standard_bounds, pop_size=15, max_iter=20)
    jade.init_solver(sphere_func)
    
    for _ in range(10):
        jade.step(sphere_func)
    
    assert np.all(jade.population >= lb)
    assert np.all(jade.population <= ub)
    assert np.all(jade.best_solution >= lb)
    assert np.all(jade.best_solution <= ub)