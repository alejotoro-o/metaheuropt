# MetaHeurOpt

<!-- ![docs](https://readthedocs.org/projects/metaheuropt/badge/?version=latest) -->
![PyPI - License](https://img.shields.io/pypi/l/metaheuropt)
![PyPI - Version](https://img.shields.io/pypi/v/metaheuropt)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/metaheuropt)
![PyPI Downloads](https://pepy.tech/badge/metaheuropt)

Here is the rest of the **README.md** for your package. I have researched the original citations for each algorithm to ensure the table is academically complete.

## Description

**MetaHeurOpt** is a modular Python library designed for the execution, benchmarking, and statistical analysis of metaheuristic optimization algorithms. It provides a standardized framework to compare diverse algorithms—ranging from classic evolutionary strategies to modern physics-inspired and swarm intelligence models—on complex continuous optimization problems.

## Installation

You can install the stable version of **MetaHeurOpt** directly from PyPI:

```bash
pip install metaheuropt

```

## Usage Example

Below is a quick start guide to comparing solvers. For a more detailed walkthrough, including statistical tables and 3D landscape visualization, check out the **`/examples/example.ipynb`** notebook.

```python
import numpy as np
from metaheuropt.solvers import GA, PSO
from metaheuropt.core import Optimizer
from metaheuropt.analysis import ResultsAnalyzer

# 1. Define your objective function
def sphere(x):
    return np.sum(np.square(x))

# 2. Configure experiment
dims = 10
bounds = (np.array([-5.12] * dims), np.array([5.12] * dims))
num_runs = 5
max_iter = 100

# 3. Select solvers
solvers = [
    GA(bounds=bounds, pop_size=50, max_iter=max_iter),
    PSO(bounds=bounds, pop_size=50, max_iter=max_iter),
]

# 4. Run optimization
opt = Optimizer(solvers=solvers, obj_func=sphere, num_runs=num_runs)
opt.run(save_results=True, results_folder="demo_results")

# 5. Analyze and visualize
analyzer = ResultsAnalyzer("demo_results")
analyzer.plot_convergence()
analyzer.perform_stats(full=True)

```

## Available Algorithms

The library currently supports the following high-performance metaheuristics:

| Code | Name | Original Reference |
| --- | --- | --- |
| **GA** | Genetic Algorithm | [K. F. Man et al. (1976)](https://doi.org/10.1109/41.538609) |
| **PSO** | Particle Swarm Optimization | [Kennedy & Eberhart (1995)](https://ieeexplore.ieee.org/document/488968) |
| **MVO** | Multi-Verse Optimizer | [Mirjalili et al. (2016)](https://link.springer.com/article/10.1007/s00521-015-1870-7) |
| **CMA-ES** | Covariance Matrix Adaptation Evolution Strategy | [Hansen & Ostermeier (2001)](https://doi.org/10.1162/106365603321828970) |
| **GWO** | Grey Wolf Optimizer | [Mirjalili et al. (2014)](https://doi.org/10.1109/CEC.2007.4424751) |
| **JADE** | Adaptive Differential Evolution with Archive | [Zhang & Sanderson (2009)](https://doi.org/10.1109/CEC.2007.4424751) |
| **NSGA-II** | Non-dominated Sorting Genetic Algorithm II | [Deb et al. (2002)](https://doi.org/10.1109/4235.996017) |
| **WOA** | Whale Optimization Algorithm | [Mirjalili & Lewis (2016)](https://doi.org/10.1016/J.ADVENGSOFT.2016.01.008) |
