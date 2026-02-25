import numpy as np
from metaheuropt.solvers import DE

def test_de_initialization(standard_bounds):
    """Test if DE initializes parameters and state correctly."""
    pop_size = 30
    F, CR = 0.6, 0.8
    de = DE(bounds=standard_bounds, pop_size=pop_size, F=F, CR=CR)
    
    assert de.name == "DE"
    assert de.F == F
    assert de.CR == CR
    assert de.pop_size == pop_size
    assert de.current_iter == 0

def test_de_lifecycle_step(sphere_func, standard_bounds):
    """Test the DE initialization and greedy selection step."""
    pop_size = 20
    de = DE(bounds=standard_bounds, pop_size=pop_size, max_iter=50)
    
    # 1. Test Initialization
    de.init_solver(sphere_func)
    assert de.population.shape == (pop_size, 3)
    assert de.fitness.shape == (pop_size,)
    assert de.best_fitness != np.inf
    
    # 2. Test a single step
    initial_best = de.best_fitness
    initial_pop = de.population.copy()
    
    improved = de.step(sphere_func)
    
    assert isinstance(improved, bool)
    assert de.population.shape == (pop_size, 3)
    # Best fitness should be monotonic
    assert de.best_fitness <= initial_best
    
    # Verify greedy selection: no individual should have a worse fitness than before
    # We re-evaluate the population to ensure the state is consistent
    current_fitness = np.array([sphere_func(ind) for ind in de.population])
    assert np.all(current_fitness <= de.fitness) 

def test_de_bounds_enforcement(sphere_func, standard_bounds):
    """Ensure DE mutation/crossover results stay within bounds."""
    lb, ub = standard_bounds
    de = DE(bounds=standard_bounds, pop_size=15, max_iter=20)
    de.init_solver(sphere_func)
    
    for _ in range(10):
        de.step(sphere_func)
    
    # Check current population and best solution
    assert np.all(de.population >= lb)
    assert np.all(de.population <= ub)
    assert np.all(de.best_solution >= lb)
    assert np.all(de.best_solution <= ub)

def test_de_mutation_logic(sphere_func, standard_bounds):
    """
    Verify that the current_iter increments and mutation 
    doesn't crash with small populations.
    """
    # DE/rand/1 requires at least 4 individuals (i + r1 + r2 + r3)
    de = DE(bounds=standard_bounds, pop_size=4)
    de.init_solver(sphere_func)
    
    assert de.current_iter == 0
    de.step(sphere_func)
    assert de.current_iter == 1