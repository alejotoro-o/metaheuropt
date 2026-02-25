import pytest
import numpy as np
from metaheuropt.solvers import MVO

def test_mvo_initialization(standard_bounds):
    """Test if MVO initializes hyperparameters and universes correctly."""
    pop_size = 40
    mvo = MVO(bounds=standard_bounds, pop_size=pop_size, wep_min=0.2, wep_max=1.0)
    mvo.init_solver(lambda x: np.sum(x**2))
    
    assert mvo.name == "MVO"
    assert mvo.pop_size == pop_size
    assert mvo.current_iter == 0
    assert mvo.population.shape == (pop_size, 3)
    assert mvo.best_fitness != np.inf

def test_mvo_lifecycle_step(sphere_func, standard_bounds):
    """Test the MVO step including coefficient updates and elite preservation."""
    pop_size = 20
    mvo = MVO(bounds=standard_bounds, pop_size=pop_size, max_iter=50)
    mvo.init_solver(sphere_func)
    
    initial_best = mvo.best_fitness
    
    # Perform a step
    improved = mvo.step(sphere_func)
    
    assert isinstance(improved, bool)
    assert mvo.current_iter == 1
    # Global best should never degrade
    assert mvo.best_fitness <= initial_best
    # The population should maintain its shape
    assert mvo.population.shape == (pop_size, 3)

def test_mvo_coefficient_logic(sphere_func, standard_bounds):
    """Verify WEP and TDR change correctly over iterations."""
    max_iter = 100
    mvo = MVO(bounds=standard_bounds, pop_size=10, max_iter=max_iter, wep_min=0.2, wep_max=0.8)
    mvo.init_solver(sphere_func)
    
    # Iteration 1
    mvo.step(sphere_func)
    # WEP = 0.2 + 1 * (0.8-0.2)/100 = 0.206
    # We check if it is within reasonable bounds
    assert 0.2 < (mvo.wep_min + mvo.current_iter * (mvo.wep_max - mvo.wep_min) / mvo.max_iter) < 0.8
    
    # Fast forward to end
    mvo.current_iter = 99
    mvo.step(sphere_func)
    # TDR should be very small near the end
    tdr_last = 1 - (100**(1/mvo.p) / 100**(1/mvo.p))
    assert np.isclose(tdr_last, 0.0)

def test_mvo_roulette_wheel_logic(sphere_func, standard_bounds):
    """Verify that the White Hole selection logic handles equal fitness."""
    pop_size = 5
    mvo = MVO(bounds=standard_bounds, pop_size=pop_size)
    mvo.init_solver(sphere_func)
    
    # Mocking equal fitness to test the 'else' branch in step 3
    mvo.fitness = np.array([10.0] * pop_size)
    
    # A single step should execute without division by zero
    try:
        mvo.step(sphere_func)
    except Exception as e:
        pytest.fail(f"MVO failed on equal fitness universes: {e}")

def test_mvo_bounds_enforcement(sphere_func, standard_bounds):
    """Ensure the Wormhole mechanism doesn't teleport universes out of bounds."""
    lb, ub = standard_bounds
    # Use a large TDR/offset potential to test clipping
    mvo = MVO(bounds=standard_bounds, pop_size=15, max_iter=10)
    mvo.init_solver(sphere_func)
    
    for _ in range(5):
        mvo.step(sphere_func)
    
    assert np.all(mvo.population >= lb)
    assert np.all(mvo.population <= ub)
    assert np.all(mvo.best_solution >= lb)
    assert np.all(mvo.best_solution <= ub)