import numpy as np
from metaheuropt.core import Optimizer

def test_optimizer_initialization(simple_ga, sphere_func):
    """Test if the optimizer accepts a single solver or a list."""
    opt = Optimizer(solvers=simple_ga, obj_func=sphere_func, num_runs=2)
    assert len(opt.solvers) == 1
    
    opt_list = Optimizer(solvers=[simple_ga, simple_ga], obj_func=sphere_func)
    assert len(opt_list.solvers) == 2

def test_optimizer_run_and_files(simple_ga, sphere_func, temp_results_dir):
    """Test if optimizer runs and creates the pkl and csv files."""
    num_runs = 2
    opt = Optimizer(solvers=simple_ga, obj_func=sphere_func, num_runs=num_runs)
    
    # Run the optimizer pointing to our temp directory
    opt.run(save_results=True, results_folder=str(temp_results_dir))
    
    # Check if results dictionary is populated
    assert "GA" in opt.results
    assert len(opt.results["GA"]) == num_runs
    
    # Check if files exist
    assert (temp_results_dir / "GA_data.pkl").exists()
    assert (temp_results_dir / "summary_results.csv").exists()

def test_optimizer_summary_logic(simple_ga, sphere_func):
    """Verify that statistics are calculated correctly in the summary."""
    opt = Optimizer(solvers=simple_ga, obj_func=sphere_func, num_runs=3)
    opt.run(save_results=False) # Don't need files for this
    
    # Check if we have results for each run
    ga_results = opt.results["GA"]
    fitness_values = [r["fitness"] for r in ga_results]
    
    # Summary printing should not crash
    opt.summary() 
    
    assert len(fitness_values) == 3
    assert all(isinstance(f, (float, np.float64)) for f in fitness_values)

def test_optimizer_custom_folder_creation(simple_ga, sphere_func, tmp_path):
    """Ensure optimizer creates nested directories if they don't exist."""
    custom_path = tmp_path / "nested" / "output" / "folder"
    opt = Optimizer(solvers=simple_ga, obj_func=sphere_func, num_runs=1)
    
    opt.run(save_results=True, results_folder=str(custom_path))
    
    assert custom_path.exists()
    assert (custom_path / "summary_results.csv").exists()