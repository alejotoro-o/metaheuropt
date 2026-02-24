import pytest
import numpy as np
import shutil
from pathlib import Path

## To run without plugins and avoid errors use:
## pytest --disable-plugin-autoload

## Utils
@pytest.fixture
def sphere_func():
    """Sphere Function: f(x) = sum(x^2). Global min at x=0, f(x)=0."""
    return lambda x: np.sum(np.square(x))

@pytest.fixture
def standard_bounds():
    """Standard 3D bounds for testing."""
    lb = np.array([-5.0, -5.0, -5.0])
    ub = np.array([5.0, 5.0, 5.0])
    return lb, ub

## Optimizer
@pytest.fixture
def temp_results_dir(tmp_path):
    """Provides a temporary directory and deletes it after the test."""
    path = tmp_path / "results"
    path.mkdir(parents=True, exist_ok=True)
    
    yield path  # This is where the test happens
    
    # --- Teardown: This runs after the test finishes ---
    if path.exists():
        shutil.rmtree(path)

@pytest.fixture
def simple_ga(standard_bounds):
    """A GA instance for testing the optimizer."""
    from metaheuropt.solvers import GeneticAlgorithm
    return GeneticAlgorithm(bounds=standard_bounds, pop_size=10, max_iter=5)

## Solvers
@pytest.fixture
def ga_config(standard_bounds):
    """A default GA configuration for tests."""
    return {
        "bounds": standard_bounds,
        "pop_size": 30,
        "max_iter": 50,
        "pc": 0.8
    }