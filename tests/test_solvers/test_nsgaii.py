import numpy as np
from metaheuropt.solvers import NSGAII

def test_nsgaii_initialization(standard_bounds):
    """Test if NSGA-II initializes parameters and offspring logic correctly."""
    pop_size = 24
    nsga = NSGAII(bounds=standard_bounds, pop_size=pop_size)
    nsga.init_solver(lambda x: np.sum(x**2))
    
    assert nsga.name == "NSGAII"
    assert nsga.pop_size == pop_size
    # Default mutation probability should be 1/dim
    assert nsga.pm == 1.0 / 3 
    assert nsga.current_iter == 0

def test_nsgaii_lifecycle_step(sphere_func, standard_bounds):
    """Test the SBX, Mutation, and Elitist selection cycle."""
    pop_size = 20
    nsga = NSGAII(bounds=standard_bounds, pop_size=pop_size, max_iter=50)
    nsga.init_solver(sphere_func)
    
    initial_best = nsga.best_fitness
    
    # Perform a step
    improved = nsga.step(sphere_func)
    
    assert isinstance(improved, bool)
    assert nsga.population.shape == (pop_size, 3)
    # Due to elitism (Mu + Lambda), the best fitness must never get worse
    assert nsga.best_fitness <= initial_best
    # The fitness array should always be sorted because of the internal argsort
    assert np.all(np.diff(nsga.fitness) >= 0)

def test_nsgaii_sbx_logic(standard_bounds):
    """Verify that SBX crossover produces two children within bounds."""
    nsga = NSGAII(bounds=standard_bounds)
    p1 = np.array([-1.0, -1.0, -1.0])
    p2 = np.array([1.0, 1.0, 1.0])
    
    c1, c2 = nsga._sbx_crossover(p1, p2)
    
    assert c1.shape == (3,)
    assert c2.shape == (3,)
    # Children should stay within the configured search space
    assert np.all(c1 >= nsga.lb) and np.all(c1 <= nsga.ub)
    assert np.all(c2 >= nsga.lb) and np.all(c2 <= nsga.ub)

def test_nsgaii_mutation_logic(standard_bounds):
    """Verify that Polynomial Mutation respects boundaries and shifts values."""
    nsga = NSGAII(bounds=standard_bounds, pm=1.0) # Force mutation
    x = np.array([0.0, 0.0, 0.0])
    
    mutated_x = nsga._polynomial_mutation(x)
    
    assert mutated_x.shape == (3,)
    # With pm=1.0, the value should almost certainly change
    assert not np.array_equal(x, mutated_x)
    assert np.all(mutated_x >= nsga.lb) and np.all(mutated_x <= nsga.ub)

def test_nsgaii_elitism_survival(sphere_func, standard_bounds):
    """Verify that the best individual always survives into the next generation."""
    nsga = NSGAII(bounds=standard_bounds, pop_size=10)
    nsga.init_solver(sphere_func)
    
    # Force a known very good solution into the population
    nsga.population[0] = np.zeros(3)
    nsga.fitness[0] = sphere_func(nsga.population[0])
    nsga.best_fitness = nsga.fitness[0]
    
    nsga.step(sphere_func)
    
    # The zero vector (global optimum for sphere) should still be at index 0
    # due to the Mu+Lambda sorting logic.
    assert nsga.fitness[0] == 0.0
    assert np.allclose(nsga.population[0], [0, 0, 0])

def test_nsgaii_bounds_enforcement(sphere_func, standard_bounds):
    """Check overall boundary integrity after multiple generations."""
    lb, ub = standard_bounds
    nsga = NSGAII(bounds=standard_bounds, pop_size=10, max_iter=20)
    nsga.init_solver(sphere_func)
    
    for _ in range(5):
        nsga.step(sphere_func)
    
    assert np.all(nsga.population >= lb)
    assert np.all(nsga.population <= ub)
    assert np.all(nsga.best_solution >= lb)
    assert np.all(nsga.best_solution <= ub)