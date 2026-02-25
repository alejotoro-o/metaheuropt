import numpy as np
from metaheuropt.solvers import GWO

def test_gwo_initialization(standard_bounds):
    """Test if GWO initializes the social hierarchy correctly."""
    pop_size = 20
    gwo = GWO(bounds=standard_bounds, pop_size=pop_size)
    
    # Initialize with a simple sum function
    gwo.init_solver(lambda x: np.sum(x))
    
    assert gwo.name == "GWO"
    assert gwo.alpha_pos is not None
    assert gwo.beta_pos is not None
    assert gwo.delta_pos is not None
    # Hierarchy check: Alpha should be better (lower) than Beta, and Beta better than Delta
    assert gwo.alpha_score <= gwo.beta_score
    assert gwo.beta_score <= gwo.delta_score
    assert gwo.current_iter == 0

def test_gwo_lifecycle_step(sphere_func, standard_bounds):
    """Test the GWO hunting step and hierarchy update."""
    pop_size = 15
    gwo = GWO(bounds=standard_bounds, pop_size=pop_size, max_iter=50)
    gwo.init_solver(sphere_func)
    
    initial_alpha_score = gwo.alpha_score
    initial_pos = gwo.population.copy()
    
    improved = gwo.step(sphere_func)
    
    assert isinstance(improved, bool)
    # The pack should have moved towards the leaders
    assert not np.array_equal(gwo.population, initial_pos)
    # Global best (alpha) should be monotonic
    assert gwo.alpha_score <= initial_alpha_score
    # Ensure current_iter incremented for the 'a' parameter decay
    assert gwo.current_iter == 1

def test_gwo_hierarchy_integrity(sphere_func, standard_bounds):
    """Verify that the social hierarchy remains sorted by fitness after a step."""
    gwo = GWO(bounds=standard_bounds, pop_size=10)
    gwo.init_solver(sphere_func)
    
    # Run a few steps
    for _ in range(3):
        gwo.step(sphere_func)
        
    # Check that scores are strictly ordered: Alpha <= Beta <= Delta
    assert gwo.alpha_score <= gwo.beta_score
    assert gwo.beta_score <= gwo.delta_score
    # Best fitness should mirror alpha_score
    assert gwo.best_fitness == gwo.alpha_score
    assert np.array_equal(gwo.best_solution, gwo.alpha_pos)

def test_gwo_a_parameter_decay(sphere_func, standard_bounds):
    """Check if the logic for 'a' correctly handles the final iteration."""
    max_iter = 100
    gwo = GWO(bounds=standard_bounds, pop_size=10, max_iter=max_iter)
    gwo.init_solver(sphere_func)
    
    # Manual check: when current_iter = max_iter, a should be 0 (full exploitation)
    gwo.current_iter = 99
    gwo.step(sphere_func)
    assert gwo.current_iter == 100

def test_gwo_bounds_enforcement(sphere_func, standard_bounds):
    """Ensure the hunting movement doesn't push wolves out of the search space."""
    lb, ub = standard_bounds
    gwo = GWO(bounds=standard_bounds, pop_size=12, max_iter=20)
    gwo.init_solver(sphere_func)
    
    for _ in range(5):
        gwo.step(sphere_func)
    
    # Check hierarchy and population
    assert np.all(gwo.alpha_pos >= lb) and np.all(gwo.alpha_pos <= ub)
    assert np.all(gwo.beta_pos >= lb) and np.all(gwo.beta_pos <= ub)
    assert np.all(gwo.delta_pos >= lb) and np.all(gwo.delta_pos <= ub)
    assert np.all(gwo.population >= lb) and np.all(gwo.population <= ub)