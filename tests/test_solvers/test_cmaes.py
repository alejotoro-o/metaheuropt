import numpy as np
from metaheuropt.solvers import CMAES

def test_cmaes_initialization(standard_bounds):
    """Test if CMA-ES initializes matrices and paths correctly."""
    pop_size = 10
    cma = CMAES(bounds=standard_bounds, pop_size=pop_size)
    cma.init_solver(lambda x: np.sum(x**2))
    
    assert cma.name == "CMAES"
    assert cma.dim == 3
    assert cma.pc.shape == (3,)
    assert cma.ps.shape == (3,)
    assert cma.C.shape == (3, 3)
    # Covariance matrix should initially be identity
    assert np.allclose(cma.C, np.eye(3))

def test_cmaes_lifecycle_step(sphere_func, standard_bounds):
    """Test the CMA-ES step execution and state transition."""
    cma = CMAES(bounds=standard_bounds, pop_size=12, max_iter=10)
    cma.init_solver(sphere_func)
    
    initial_sigma = cma.sigma
    initial_best = cma.best_fitness
    
    # Perform a step
    improved = cma.step(sphere_func)
    
    assert isinstance(improved, bool)
    # Step-size sigma should have been updated
    assert cma.sigma != initial_sigma
    # Best fitness should be equal or better
    assert cma.best_fitness <= initial_best

def test_cmaes_covariance_properties(sphere_func, standard_bounds):
    """Ensure the covariance matrix C remains symmetric and positive definite."""
    cma = CMAES(bounds=standard_bounds, pop_size=20, max_iter=50)
    cma.init_solver(sphere_func)
    
    # Run for a few iterations to let C evolve
    for _ in range(5):
        cma.step(sphere_func)
    
    # 1. Symmetry check
    assert np.allclose(cma.C, cma.C.T, atol=1e-8)
    
    # 2. Positive Definiteness: All eigenvalues must be positive
    eigenvalues = np.linalg.eigvalsh(cma.C)
    assert np.all(eigenvalues > 0)

def test_cmaes_weights_logic(standard_bounds):
    """Verify strategy weights (self.w) properties."""
    cma = CMAES(bounds=standard_bounds, pop_size=20)
    # Strategy weights are calculated in __init__
    
    # Weights should be decreasing (best individuals have more weight)
    assert np.all(np.diff(cma.w) < 0)
    # Weights should be positive and sum to 1
    assert np.all(cma.w > 0)
    assert np.isclose(np.sum(cma.w), 1.0)

def test_cmaes_bounds_enforcement(sphere_func, standard_bounds):
    """Ensure CMA-ES respects boundaries after iterations."""
    lb, ub = standard_bounds
    cma = CMAES(bounds=standard_bounds, pop_size=10, max_iter=20)
    cma.init_solver(sphere_func)
    
    for _ in range(5):
        cma.step(sphere_func)
    
    # Check best solution
    assert np.all(cma.best_solution >= lb)
    assert np.all(cma.best_solution <= ub)