import numpy as np
import time
import pickle
import csv
from tqdm import tqdm
from pathlib import Path
from typing import List, Union, Optional
from ..solvers.base import BaseSolver

class Optimizer:
    """
    Orchestrator for benchmarking and running multiple metaheuristic solvers.

    This class handles the execution of several optimization algorithms over 
    multiple independent runs, manages progress tracking, calculates aggregated 
    statistics, and exports results for post-processing.

    Attributes:
        solvers (List[BaseSolver]): List of initialized solver instances to compare.
        obj_func (callable): The objective function to minimize.
        num_runs (int): Number of independent executions per solver for statistical validity.
        results (dict): Internal storage for raw data collected during execution.
        output_dir (Path): Directory where results and summaries will be exported.
    """

    def __init__(self, solvers: Union[BaseSolver, List[BaseSolver]], obj_func, num_runs: int = 30):
        """
        Initializes the Optimizer with selected solvers and a target function.

        Args:
            solvers (Union[BaseSolver, List[BaseSolver]]): A single solver or a list of solvers.
            obj_func (callable): Function that accepts a 1D NumPy array and returns a scalar.
            num_runs (int): Number of times to repeat the optimization for each algorithm.
        """

        self.solvers = solvers if isinstance(solvers, list) else [solvers]
        self.obj_func = obj_func
        self.num_runs = num_runs
        self.results = {}
        # Default output directory
        self.output_dir = Path("results")

    def run(self, save_results: bool = True, results_folder: Optional[str] = None):
        """
        Executes the benchmarking process for all registered solvers.

        Iterates through the solvers and performs the specified number of independent runs,
        monitoring convergence and stagnation for each.

        Args:
            save_results (bool): If True, serializes detailed metrics to Disk (.pkl and .csv).
            results_folder (Optional[str]): Custom path for output files. Defaults to "results/".
        """

        # Update output directory if a custom folder is provided
        if results_folder:
            self.output_dir = Path(results_folder)
        
        if save_results:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        for solver in self.solvers:
            run_data = []
            print(f"\n{'='*50}\nOptimizing with: {solver.name}\n{'='*50}")
            
            for r in range(self.num_runs):
                solver.init_solver(self.obj_func)
                convergence = [solver.best_fitness]
                stagnation_counter = 0
                start_t = time.time()

                # Inner progress bar
                pbar = tqdm(range(solver.max_iter), desc=f"Run {r+1}/{self.num_runs}", leave=False)
                for _ in pbar:
                    improved = solver.step(self.obj_func)
                    convergence.append(solver.best_fitness)
                    
                    stagnation_counter = 0 if improved else stagnation_counter + 1
                    pbar.set_postfix({"Best": f"{solver.best_fitness:.4e}"})
                    
                    if solver.stop_patience and stagnation_counter >= solver.stop_patience:
                        break
                
                run_data.append({
                    "fitness": solver.best_fitness,
                    "solution": solver.best_solution,
                    "time": time.time() - start_t,
                    "convergence": np.array(convergence)
                })

            self.results[solver.name] = run_data
            
            if save_results:
                self._save_solver_data(solver.name, run_data)
        
        self.summary()
        if save_results:
            self.save_summary()

    def _save_solver_data(self, solver_name: str, run_data: list):
        """
        Processes and saves the statistical data for a specific solver.

        Normalizes convergence curves using edge padding (to account for early stopping)
        and serializes a dictionary of metrics via Pickle.

        Args:
            solver_name (str): Name identifier of the algorithm.
            run_data (list): List of dictionaries containing metrics for each independent run.
        """

        max_len = max(len(d["convergence"]) for d in run_data)
        padded_conv = np.array([
            np.pad(d["convergence"], (0, max_len - len(d["convergence"])), 'edge') 
            for d in run_data
        ])
        
        stats = {
            "conv_mean": np.mean(padded_conv, axis=0),
            "conv_std": np.std(padded_conv, axis=0),
            "best_fitness_all": [d["fitness"] for d in run_data],
            "best_solutions_all": [d["solution"] for d in run_data],
            "execution_times": [d["time"] for d in run_data]
        }
        
        filename = self.output_dir / f"{solver_name}_data.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(stats, f)
        print(f" > Detailed data for {solver_name} saved to {filename}")

    def summary(self):
        """
        Calculates and displays a formatted summary table in the console.

        The table includes the Mean Best Fitness, Standard Deviation, and Average
        Execution Time for every algorithm tested.
        """

        print(f"\n{'='*25} GLOBAL SUMMARY {'='*25}")
        print(f"{'Algorithm':<12} | {'Mean Best':<12} | {'Std Dev':<12} | {'Avg Time (s)':<12}")
        print("-" * 65)
        
        for name, data in self.results.items():
            fitnesses = [d["fitness"] for d in data]
            times = [d["time"] for d in data]
            print(f"{name:<12} | {np.mean(fitnesses):<12.4e} | {np.std(fitnesses):<12.4e} | {np.mean(times):<12.2f}")
        print("-" * 65)

    def save_summary(self, filename: str = "summary_results.csv"):
        """
        Exports the global statistical summary to a CSV file.

        Args:
            filename (str): Name of the CSV file. Defaults to "summary_results.csv".
        """

        path = self.output_dir / filename
        headers = ["Algorithm", "Mean Fitness", "Std Fitness", "Mean Time (s)", "Runs"]
        
        with open(path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for name, data in self.results.items():
                fitnesses = [d["fitness"] for d in data]
                times = [d["time"] for d in data]
                writer.writerow({
                    "Algorithm": name,
                    "Mean Fitness": f"{np.mean(fitnesses):.8e}",
                    "Std Fitness": f"{np.std(fitnesses):.8e}",
                    "Mean Time (s)": f"{np.mean(times):.4f}",
                    "Runs": self.num_runs
                })
        print(f"Global summary saved to {path}")

    def get_overall_best(self, solver_name):
        """
        Retrieves the single best run result for a given algorithm.

        Args:
            solver_name (str): The name of the solver to query.

        Returns:
            dict: Dictionary containing the best fitness, solution vector, 
                  execution time, and convergence history for the best run.
                  Returns None if the solver_name is not found.
        """
        
        if solver_name not in self.results: return None
        best_run = min(self.results[solver_name], key=lambda x: x["fitness"])
        return best_run