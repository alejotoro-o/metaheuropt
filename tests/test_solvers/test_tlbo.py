import numpy as np
from metaheuropt.solvers import TLBO

def test_tlbo_initialization(standard_bounds):
    """Test if TLBO initializes the classroom (population) correctly."""
    pop_size = 25
    tlbo = TLBO(bounds=standard_bounds, pop_size=pop_size)
    tlbo.init_solver(lambda x: np.sum(x**2))
    
    assert tlbo.name == "TLBO"
    assert tlbo.population.shape == (pop_size, 3)
    assert tlbo.fitness.shape == (pop_size,)
    assert tlbo.best_fitness != np.inf

def test_tlbo_lifecycle_step(sphere_func, standard_bounds):
    """Test the Teacher and Learner phases in a single step."""
    pop_size = 15
    tlbo = TLBO(bounds=standard_bounds, pop_size=pop_size)
    tlbo.init_solver(sphere_func)
    
    initial_best = tlbo.best_fitness
    initial_pop = tlbo.population.copy()
    
    # Perform a step
    improved = tlbo.step(sphere_func)
    
    assert isinstance(improved, bool)
    # Classroom should have updated positions
    assert not np.array_equal(tlbo.population, initial_pop)
    # Global best (the current Teacher) should be monotonic
    assert tlbo.best_fitness <= initial_best
    # The fitness array should be updated and consistent with population
    recalculated_fit = np.array([sphere_func(ind) for ind in tlbo.population])
    assert np.allclose(tlbo.fitness, recalculated_fit)

def test_tlbo_teacher_phase_logic(sphere_func, standard_bounds):
    """Verify the Teacher Phase influences the population correctly."""
    tlbo = TLBO(bounds=standard_bounds, pop_size=10)
    tlbo.init_solver(sphere_func)
    
    # Force a specific teacher (optimum)
    tlbo.best_solution = np.zeros(3)
    tlbo.best_fitness = 0.0
    
    # After a step, individuals should generally move toward the teacher
    old_dist_to_teacher = np.linalg.norm(tlbo.population - tlbo.best_solution, axis=1)
    tlbo.step(sphere_func)
    new_dist_to_teacher = np.linalg.norm(tlbo.population - tlbo.best_solution, axis=1)
    
    # On a simple convex sphere, fitness should generally improve or stay same
    # due to the greedy selection in both phases.
    assert np.all(new_dist_to_teacher <= old_dist_to_teacher + 1e-9)

def test_tlbo_learner_phase_interaction(sphere_func, standard_bounds):
    """Ensure the Learner phase logic doesn't crash with small populations."""
    # Minimum population for interaction is 2
    tlbo = TLBO(bounds=standard_bounds, pop_size=2)
    tlbo.init_solver(sphere_func)
    
    # This should run without infinite loops in the 'while k == i' block
    improved = tlbo.step(sphere_func)
    assert isinstance(improved, bool)

def test_tlbo_bounds_enforcement(sphere_func, standard_bounds):
    """Ensure that the pedagogical movement stays within boundaries."""
    lb, ub = standard_bounds
    tlbo = TLBO(bounds=standard_bounds, pop_size=10, max_iter=20)
    tlbo.init_solver(sphere_func)
    
    for _ in range(5):
        tlbo.step(sphere_func)
    
    # Check current population and best solution
    assert np.all(tlbo.population >= lb)
    assert np.all(tlbo.population <= ub)
    assert np.all(tlbo.best_solution >= lb)
    assert np.all(tlbo.best_solution <= ub)