import numpy as np
from metaheuropt.solvers import PSO

def test_pso_initialization(standard_bounds):
    """Test if PSO initializes swarm state (velocity, pbest) correctly."""
    pop_size = 30
    pso = PSO(bounds=standard_bounds, pop_size=pop_size)
    pso.init_solver(lambda x: np.sum(x**2))
    
    assert pso.name == "PSO"
    assert pso.velocity.shape == (pop_size, 3)
    assert np.all(pso.velocity == 0)
    assert pso.pbest_pos.shape == (pop_size, 3)
    assert pso.pbest_val.shape == (pop_size,)
    assert pso.current_iter == 0
    # Check velocity limits (20% of range [-5, 5] is 2.0)
    assert np.all(pso.v_max == 2.0)

def test_pso_lifecycle_step(sphere_func, standard_bounds):
    """Test the velocity update, position shift, and pbest/gbest tracking."""
    pop_size = 20
    pso = PSO(bounds=standard_bounds, pop_size=pop_size, max_iter=50)
    pso.init_solver(sphere_func)
    
    initial_best = pso.best_fitness
    initial_pos = pso.population.copy()
    
    # Perform a step
    improved = pso.step(sphere_func)
    
    assert isinstance(improved, bool)
    assert pso.current_iter == 1
    # Particles should have moved
    assert not np.array_equal(pso.population, initial_pos)
    # Velocity should no longer be zero
    assert np.any(pso.velocity != 0)
    # Global best should never degrade
    assert pso.best_fitness <= initial_best

def test_pso_personal_best_update(sphere_func, standard_bounds):
    """Verify that individual personal bests are only updated when fitness improves."""
    pso = PSO(bounds=standard_bounds, pop_size=5)
    pso.init_solver(sphere_func)
    
    old_pbest_val = pso.pbest_val.copy()
    
    # Step 1
    pso.step(sphere_func)
    
    # Each personal best value must be <= its previous value
    assert np.all(pso.pbest_val <= old_pbest_val)

def test_pso_velocity_clamping(sphere_func, standard_bounds):
    """Ensure that particle velocities never exceed the defined v_max."""
    # Set high coefficients to encourage large velocity jumps
    pso = PSO(bounds=standard_bounds, pop_size=10, c1=10.0, c2=10.0)
    pso.init_solver(sphere_func)
    
    for _ in range(5):
        pso.step(sphere_func)
        # Check all velocity components across all particles
        assert np.all(pso.velocity <= pso.v_max + 1e-15)
        assert np.all(pso.velocity >= pso.v_min - 1e-15)

def test_pso_bounds_enforcement(sphere_func, standard_bounds):
    """Check that particles stay within the search space after position updates."""
    lb, ub = standard_bounds
    pso = PSO(bounds=standard_bounds, pop_size=15, max_iter=20)
    pso.init_solver(sphere_func)
    
    for _ in range(10):
        pso.step(sphere_func)
    
    assert np.all(pso.population >= lb)
    assert np.all(pso.population <= ub)
    assert np.all(pso.best_solution >= lb)
    assert np.all(pso.best_solution <= ub)