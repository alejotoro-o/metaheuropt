import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from itertools import combinations

class ResultsAnalyzer:
    """
    A post-processing utility for analyzing and visualizing metaheuristic optimization results.

    This class loads experiment data stored in pickle format, calculates descriptive 
    and inferential statistics, and generates comparative visualizations including 
    convergence curves, boxplots, and performance radar charts.

    Attributes:
        folder (Path): Directory path containing the .pkl result files.
        data (dict): Nested dictionary storing loaded metrics for each algorithm.
    """

    def __init__(self, results_folder: str):
        """
        Initializes the analyzer and automatically loads available data.

        Args:
            results_folder (str): Path to the directory where Optimizer results are saved.
        """

        self.folder = Path(results_folder)
        self.data = {}
        self._load_data()

    def _load_data(self):
        """
        Internal method to discover and deserialize all *_data.pkl files.
        
        Populates the self.data dictionary using the file stem as the algorithm key.
        """

        files = list(self.folder.glob("*_data.pkl"))
        if not files:
            print(f"No results found in {self.folder}")
            return

        for f in files:
            name = f.stem.replace("_data", "")
            with open(f, 'rb') as pkl:
                self.data[name] = pickle.load(pkl)

    def plot_convergence(self, log_scale: bool = False):
        """
        Plots the Mean convergence trajectory with Standard Deviation bands.

        Args:
            log_scale (bool): If True, sets the Y-axis to a logarithmic scale. 
                              Useful for visualizing high-precision convergence.
        """

        plt.figure(figsize=(10, 6))
        
        for name, d in self.data.items():
            mean_conv = d["conv_mean"]
            std_conv = d["conv_std"]
            iters = np.arange(len(mean_conv))
            
            plt.plot(iters, mean_conv, label=f"{name}", linewidth=2)
            plt.fill_between(iters, 
                             mean_conv - std_conv, 
                             mean_conv + std_conv, 
                             alpha=0.15)

        if log_scale:
            plt.yscale('log')
        
        plt.title("Convergence Analysis (Mean ± Std)")
        plt.xlabel("Iteration")
        plt.ylabel("Objective Function Value")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.show()

    def plot_boxplots(self):
        """
        Generates boxplots to compare the distribution of final best fitness values.
        
        Visualizes the median, quartiles, and outliers for all loaded algorithms 
        to assess robustness and consistency.
        """

        plt.figure(figsize=(10, 6))
        
        fitness_data = [d["best_fitness_all"] for d in self.data.values()]
        labels = list(self.data.keys())
        
        plt.boxplot(fitness_data, labels=labels, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='blue'))
        
        plt.title("Statistical Distribution of Final Best Fitness")
        plt.ylabel("Objective Value")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def plot_radar(self):
        """
        Plots a normalized radar (spider) chart comparing multiple performance metrics.

        Metrics evaluated: Best Fitness, Mean Fitness, Std Dev, and Execution Time.
        Note: Values are Min-Max normalized and inverted so that the outer perimeter 
        always represents superior performance (Higher is Better).
        """

        # Metrics to compare
        metrics = ['Best', 'Mean', 'Std', 'Time']
        algo_names = list(self.data.keys())
        
        # Collect values
        raw_values = []
        for name in algo_names:
            d = self.data[name]
            raw_values.append([
                np.min(d["best_fitness_all"]),
                np.mean(d["best_fitness_all"]),
                np.std(d["best_fitness_all"]),
                np.mean(d["execution_times"])
            ])
        
        M = np.array(raw_values)
        # Min-Max normalization (Inverted because lower is better)
        # Formula: 1 - (val - min) / (max - min)
        Mn = 1 - (M - M.min(axis=0)) / (M.max(axis=0) - M.min(axis=0) + 1e-12)
        
        # Radar Chart Logic
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1] # Close the circle
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        for i, name in enumerate(algo_names):
            values = Mn[i].tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, label=name)
            ax.fill(angles, values, alpha=0.1)
            
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
        plt.title("Normalized Performance Radar (Higher is Better)")
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.show()

    def _cliffs_delta(self, x, y):
        """
        Calculates Cliff's Delta non-parametric effect size.

        Measures how often values in one distribution are larger than values in another,
        providing a magnitude for the difference that is robust to non-normal distributions.

        Args:
            x (array_like): First distribution of fitness values.
            y (array_like): Second distribution of fitness values.

        Returns:
            float: Delta value in the range [-1, 1].
        """

        nx, ny = len(x), len(y)
        x = np.asarray(x)
        y = np.asarray(y)
        # Matrix of all pairwise comparisons
        diffs = x[:, None] - y
        delta = (np.sum(diffs > 0) - np.sum(diffs < 0)) / (nx * ny)
        return delta

    def perform_stats(self, full: bool = False):
        """
        Performs a rigorous statistical comparison between algorithms.

        Executes the Kruskal-Wallis H-test for global significance. If significant 
        and 'full' is True, it proceeds to pairwise Mann-Whitney U tests.

        Args:
            full (bool): If True, performs pairwise comparisons with Benjamini-Hochberg 
                         (FDR) p-value correction and Cliff's Delta effect size.
        """
        
        if len(self.data) < 2:
            print("Error: Need at least two algorithms for comparison.")
            return

        names = list(self.data.keys())
        fitness_list = [self.data[n]["best_fitness_all"] for n in names]
        
        # 1. Global Non-parametric Test (Kruskal-Wallis)
        h_stat, p_kw = stats.kruskal(*fitness_list)
        
        print("\n" + "="*40)
        print(f"{'GLOBAL STATISTICAL ANALYSIS':^40}")
        print("="*40)
        print(f"Kruskal-Wallis H-test: {h_stat:.4f}")
        print(f"p-value: {p_kw:.4e}")
        
        significant = p_kw < 0.05
        print(f"Significant difference (p < 0.05): {'YES' if significant else 'NO'}")

        # 2. Extended Pairwise Analysis
        if full and significant:
            print("\n" + "-"*40)
            print(f"{'PAIRWISE COMPARISONS (FDR Adjusted)':^40}")
            print("-"*40)
            
            pairs = list(combinations(names, 2))
            raw_results = []
            
            for n1, n2 in pairs:
                x, y = self.data[n1]["best_fitness_all"], self.data[n2]["best_fitness_all"]
                
                # Use mannwhitneyu (the Python equivalent of MATLAB's ranksum)
                # alternative='two-sided' ensures it tests if they are different, not just better/worse
                _, p_rank = stats.mannwhitneyu(x, y, alternative='two-sided')
                
                delta = self._cliffs_delta(x, y)
                raw_results.append({"pair": f"{n1} vs {n2}", "p": p_rank, "delta": delta})

            # Benjamini-Hochberg (FDR) Correction
            raw_results = sorted(raw_results, key=lambda x: x["p"])
            m = len(raw_results)
            for i, res in enumerate(raw_results):
                res["p_fdr"] = min(1.0, res["p"] * m / (i + 1))

            # Table Output
            header = f"{'Comparison':<25} | {'p-val':<10} | {'FDR p':<10} | {'Delta':<8}"
            print(header)
            print("-" * len(header))
            for res in raw_results:
                print(f"{res['pair']:<25} | {res['p']:<10.2e} | {res['p_fdr']:<10.2e} | {res['delta']:<8.3f}")
        
        elif full and not significant:
            print("\n[Note] Pairwise tests skipped: Global test not significant.")