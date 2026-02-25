import numpy as np
from metaheuropt.solvers import HHO

def test_hho_initialization(standard_bounds):
    """Test if HHO initializes the hawk population and best solution correctly."""
    pop_size = 20
    hho = HHO(bounds=standard_bounds, pop_size=pop_size)
    hho.init_solver(lambda x: np.sum(np.square(x)))
    
    assert hho.name == "HHO"
    assert hho.population.shape == (pop_size, 3)
    assert hho.fitness.shape == (pop_size,)
    assert hho.best_fitness != np.inf
    assert hho.current_iter == 0

def test_hho_lifecycle_step(sphere_func, standard_bounds):
    """Test the HHO step execution and the multi-phase movement logic."""
    pop_size = 15
    hho = HHO(bounds=standard_bounds, pop_size=pop_size, max_iter=50)
    hho.init_solver(sphere_func)
    
    initial_best = hho.best_fitness
    initial_pos = hho.population.copy()
    
    improved = hho.step(sphere_func)
    
    assert isinstance(improved, bool)
    # The current_iter should increment to update Prey Energy E
    assert hho.current_iter == 1
    # Best fitness should be monotonic
    assert hho.best_fitness <= initial_best
    # Hawks should have moved from their initial positions
    assert not np.array_equal(hho.population, initial_pos)

def test_hho_rapid_dive_logic(sphere_func, standard_bounds):
    """Verify the internal _rapid_dive (Levy Flight) doesn't break the solver."""
    hho = HHO(bounds=standard_bounds, pop_size=10)
    hho.init_solver(sphere_func)
    
    # E is the energy factor, soft is the phase type
    new_pos = hho._rapid_dive(sphere_func, i=0, E=0.5, soft=True)
    
    assert new_pos.shape == (3,)
    # Resulting position should be within bounds
    assert np.all(new_pos >= hho.lb)
    assert np.all(new_pos <= hho.ub)

def test_hho_energy_decay(sphere_func, standard_bounds):
    """Test that the internal current_iter properly impacts the energy logic."""
    max_iter = 100
    hho = HHO(bounds=standard_bounds, pop_size=10, max_iter=max_iter)
    hho.init_solver(sphere_func)
    
    # Run a step mid-way
    hho.current_iter = 50
    hho.step(sphere_func)
    assert hho.current_iter == 51
    
    # Run a step near the end (where |E| will almost certainly be < 1)
    hho.current_iter = 99
    hho.step(sphere_func)
    assert hho.current_iter == 100

def test_hho_bounds_enforcement(sphere_func, standard_bounds):
    """Ensure the complex pounce strategies respect search space boundaries."""
    lb, ub = standard_bounds
    hho = HHO(bounds=standard_bounds, pop_size=12, max_iter=20)
    hho.init_solver(sphere_func)
    
    for _ in range(10):
        hho.step(sphere_func)
    
    # Check current population and best solution
    assert np.all(hho.population >= lb)
    assert np.all(hho.population <= ub)
    assert np.all(hho.best_solution >= lb)
    assert np.all(hho.best_solution <= ub)