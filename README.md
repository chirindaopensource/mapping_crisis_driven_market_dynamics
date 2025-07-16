# README.md

# *Mapping Crisis-Driven Market Dynamics: A Transfer Entropy and Kramers–Moyal Approach to Financial Networks* Implementation
<br>

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/imports-isort-1674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Type Checking: mypy](https://img.shields.io/badge/type_checking-mypy-blue)](http://mypy-lang.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=flat&logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/seaborn-%233776AB.svg?style=flat&logo=python&logoColor=white)](https://seaborn.pydata.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-blue.svg?style=flat&logo=python&logoColor=white)](https://networkx.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2507.09554-b31b1b.svg)](https://arxiv.org/abs/2507.09554)
[![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2507.09554-blue)](https://doi.org/10.48550/arXiv.2507.09554)
[![Research](https://img.shields.io/badge/Research-Financial%20Networks-green)](https://github.com/chirindaopensource/mapping_crisis_driven_market_dynamics)
[![Discipline](https://img.shields.io/badge/Discipline-Econophysics-blue)](https://github.com/chirindaopensource/mapping_crisis_driven_market_dynamics)
[![Methodology](https://img.shields.io/badge/Methodology-Information%20Theory%20%26%20Stochastic%20Methods-orange)](https://github.com/chirindaopensource/mapping_crisis_driven_market_dynamics)
[![Year](https://img.shields.io/badge/Year-2025-purple)](https://github.com/chirindaopensource/mapping_crisis_driven_market_dynamics)
<br>

**Repository:** https://github.com/chirindaopensource/mapping_crisis_driven_market_dynamics

**Owner:** 2025 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade implementation of the research methodology from the 2025 paper entitled **"Mapping Crisis-Driven Market Dynamics: A Transfer Entropy and Kramers–Moyal Approach to Financial Networks"** by:

*   Pouriya Khalilian
*   Amirhossein N. Golestani
*   Mohammad Eslamifar
*   Mostafa T. Firouzjaee
*   Javad T. Firouzjaee

The project provides a robust, end-to-end Python pipeline for constructing a multi-layered "Digital Twin" of financial market interactions. It moves beyond traditional correlation analysis to map the dynamic, non-linear, and directed relationships between assets, offering a powerful tool for systemic risk analysis, adaptive hedging, and macro-prudential policy assessment.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: run_master_pipeline](#key-callable-run_master_pipeline)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the methodologies presented in the 2025 paper "Mapping Crisis-Driven Market Dynamics." The core of this repository is the iPython Notebook `mapping_crisis_driven_market_dynamics_draft.ipynb`, which contains a comprehensive suite of functions to model financial networks using a dual information-theoretic and stochastic approach.

Traditional measures like Pearson correlation are symmetric and linear, failing to capture the complex, directed, and non-linear feedback loops that characterize modern financial markets, especially during crises. This framework addresses these shortcomings by integrating two advanced methodologies:
1.  **Transfer Entropy (TE):** A non-parametric measure from information theory that quantifies the directed flow of information between time series.
2.  **Kramers-Moyal (KM) Expansion:** A method from stochastic calculus that approximates the underlying deterministic "drift" forces governing the system's dynamics.

This codebase enables researchers, quantitative analysts, and portfolio managers to:
-   Rigorously compute static and dynamic TE and KM network matrices.
-   Quantify the intensification of information flow during market crises.
-   Identify persistent, stable relationships (e.g., safe-haven effects) and moments of significant structural change (regime shifts).
-   Perform advanced robustness and error analysis to validate findings.
-   Replicate and extend the results of the original research paper.



## Theoretical Background

The methodology implemented in this project is a direct translation of the unified framework presented in the source paper. It is designed to overcome the limitations of traditional linear correlation by employing a dual approach rooted in information theory and stochastic calculus. The theoretical pipeline can be understood in four distinct stages:

### 1. Foundational Data Transformation

The analysis begins with a critical econometric principle: financial asset prices ($P_t$) are generally non-stationary (i.e., they contain unit roots), making them unsuitable for most statistical models. To address this, the pipeline first transforms the raw price series into a stationary log-return series ($r_t$) using the standard formula:

$r_t = \log(P_t) - \log(P_{t-1})$

This transformation yields a series whose statistical properties (like mean and variance) are more stable over time, forming a valid basis for the subsequent analyses. The pipeline empirically verifies this transformation using the Augmented Dickey-Fuller test.

### 2. Layer 1: Information-Theoretic Linkages (Transfer Entropy)

To map the directed, non-linear flow of information between assets, the framework employs Transfer Entropy (TE). TE measures the reduction in uncertainty about a target asset's future state given knowledge of a source asset's past state, beyond what the target's own past already explains. It is formally defined in the paper's **Equation 2**:

$T_{j \to i} = \sum_{i_{t+1}, i_t, j_t} P(i_{t+1}, i_t, j_t) \log_2 \frac{P(i_{t+1} | i_t, j_t)}{P(i_{t+1} | i_t)}$

-   $T_{j \to i}$ represents the information flowing from asset `j` to asset `i`.
-   The measure is inherently **asymmetric** ($T_{j \to i} \neq T_{i \to j}$), allowing us to identify sources and sinks of information flow.
-   The calculation requires discretizing the continuous return data into bins to estimate the necessary joint and conditional probability distributions, as outlined conceptually in **Algorithm 1** of the paper's framework.

### 3. Layer 2: Stochastic System Dynamics (Kramers-Moyal Expansion)

To complement the TE analysis, the framework models the system's evolution using the Kramers-Moyal (KM) expansion. This method describes a stochastic process in terms of its deterministic "drift" and stochastic "diffusion" components. This implementation focuses on the first KM coefficient—the drift vector $D^{(1)}$—which represents the deterministic forces governing the system's expected movement.

The paper makes a crucial simplification by approximating this drift with a linear model:

$\frac{d}{dt}\mathbf{x}(t) \approx A\mathbf{x}(t)$

Here, $\mathbf{x}(t)$ is the vector of asset returns, and $A$ is the $N \times N$ drift coefficient matrix. The elements $A_{ij}$ represent the signed, linear influence of asset `j`'s current return on the expected change in asset `i`'s return. This matrix is estimated by solving a system of linear equations derived from the moment conditions specified in **Equation 9** of the paper:

$\langle (x_i(t+dt) - x_i(t)) x_k(t) \rangle = \sum_{j=1}^{N} A_{ij} \langle x_j(t) x_k(t) \rangle$

This provides a signed, directed map of the linear relationships, where a negative $A_{ij}$ can be interpreted as a hedging or mean-reverting influence, and a positive $A_{ij}$ suggests co-movement.

### 4. Dynamic Analysis via Sliding Window

A static analysis of the full time series provides only an average picture of the network. To capture the evolving nature of market dynamics, both the TE and KM analyses are applied within a **sliding window** framework, as described conceptually in **Algorithm 3**. The pipeline moves a window of a fixed size (e.g., 252 days) across the entire dataset with a given step size (e.g., 21 days), re-calculating the TE and KM matrices for each window.

This procedure generates a time series of network matrices, transforming the static snapshot into a dynamic "movie" of the financial system. This allows for the direct observation of how network structures change over time and, most importantly, how they are reshaped by major market events, which is the central empirical contribution of the source paper.

## Features

The provided iPython Notebook (`mapping_crisis_driven_market_dynamics_draft.ipynb`) implements the full research pipeline, including:

-   **Rigorous Validation:** Comprehensive checks for all input data and configurations.
-   **Professional Preprocessing:** A robust pipeline for cleaning financial time series and transforming them into stationary log-returns.
-   **Methodologically Pure Calculations:** Precise, numerically stable implementations of the Transfer Entropy and Kramers-Moyal drift matrix calculations.
-   **Dynamic Analysis Engine:** A flexible sliding-window framework for time-resolved analysis.
-   **Automated Interpretation:** Algorithms to automatically identify persistent network links and detect significant regime shifts.
-   **Crisis and Robustness Analysis:** A full suite of tools to quantify crisis impacts and perform sensitivity analysis on key hyperparameters.
-   **Error Analysis:** Block bootstrapping to generate confidence intervals for model estimates.
-   **Integrated Reporting:** A final function that generates a self-contained, interactive HTML report with embedded tables and figures.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Data Preparation (Tasks 1-3):** The pipeline ingests raw price data, validates it, preprocesses it into clean log-returns, and confirms the stationarity properties of the resulting series.
2.  **Static Network Calculation (Tasks 4-6):** It computes the static (full-period) TE and KM matrices, providing a baseline snapshot of the network.
3.  **Dynamic Network Analysis (Tasks 7-8):** It implements the sliding window procedure to generate a time series of network matrices and analyzes these to quantify the average network structure during specific crisis periods versus a normal baseline.
4.  **Interpretation and Meta-Analysis (Tasks 10-12):** The pipeline automatically interprets the dynamic results to find persistent links and regime shifts, performs robustness checks across different parameters, and conducts error analysis via bootstrapping.
5.  **Reporting (Task 13-14):** All findings are compiled into a comprehensive results bundle and a final, user-selectable HTML or Markdown report.

## Core Components (Notebook Structure)

The `mapping_crisis_driven_market_dynamics_draft.ipynb` notebook is structured as a logical pipeline with modular functions for each task:

-   **`validate_inputs`**: The initial quality gate for all inputs.
-   **`preprocess_price_data`**: The data cleaning and transformation engine.
-   **`perform_stationarity_analysis`**: Econometric validation of the data.
-   **`generate_descriptive_statistics`**: Initial data characterization.
-   **`calculate_transfer_entropy`**: Core information-theoretic calculation.
-   **`calculate_kramers_moyal_drift_matrix`**: Core stochastic dynamics calculation.
-   **`perform_time_resolved_analysis`**: The dynamic analysis engine.
-   **`analyze_crisis_periods`**: Crisis impact quantification.
-   **`plot_matrix_heatmap`, `plot_network_graph`**: Visualization utilities.
-   **`interpret_dynamic_results`**: Automated interpretation engine.
-   **`run_full_analysis_pipeline`**: Orchestrator for a single, complete analysis run.
-   **`perform_robustness_analysis`**: Higher-level orchestrator for sensitivity analysis.
-   **`perform_error_analysis`**: Higher-level orchestrator for confidence interval estimation.
-   **`run_master_pipeline`**: The single, top-level entry point to the entire project.

## Key Callable: run_master_pipeline

The central function in this project is `run_master_pipeline`. It orchestrates the entire analytical workflow from raw data to final report.

```python
def run_master_pipeline(
    raw_price_df: pd.DataFrame,
    study_config: Dict[str, Any],
    param_grid: Dict[str, List[Any]],
    report_format: str = 'html',
    run_robustness_analysis: bool = True,
    run_error_analysis: bool = True,
    bootstrap_samples: int = 100
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    The master orchestrator for the end-to-end Digital Twin analysis pipeline.
    ... (full docstring is in the notebook)
    """
    # ... (implementation is in the notebook)
```

## Prerequisites

-   Python 3.9+
-   Core dependencies as listed in `requirements.txt`: `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `networkx`, `statsmodels`, `tqdm`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/mapping_crisis_driven_market_dynamics.git
    cd mapping_crisis_driven_market_dynamics
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies from `requirements.txt`:**
    ```sh
    pip install -r requirements.txt
    ```

## Input Data Structure

The primary input is a `pandas.DataFrame` with a `DatetimeIndex` and columns containing the daily closing prices of the assets to be analyzed. The index should consist of business days.

**Example:**
```
                  Nasdaq  Crude-oil      Gold  US-dollar
Date
2014-08-11  4401.330078  98.080002  1310.300049  81.489998
2014-08-12  4434.129883  97.370003  1310.500000  81.620003
2014-08-13  4456.020020  97.570000  1314.599976  81.580002
...                 ...        ...          ...        ...
```

## Usage

The entire pipeline is executed through the `run_master_pipeline` function. The user must provide the raw price data, a study configuration dictionary, and a parameter grid for robustness checks.

```python
import pandas as pd

# 1. Load your data
# raw_price_df = pd.read_csv("your_data.csv", index_col=0, parse_dates=True)
# For this example, we create synthetic data.
date_rng = pd.date_range(start='2014-08-01', end='2024-09-30', freq='B')
price_data = {asset: 100 * np.exp(np.cumsum(np.random.randn(len(date_rng)) * 0.01)) for asset in ['Nasdaq', 'Crude-oil', 'Gold', 'US-dollar']}
raw_price_df = pd.DataFrame(price_data, index=date_rng)

# 2. Define your configurations (see notebook for full example)
study_config = { ... } # As defined in the notebook
param_grid = { ... }   # As defined in the notebook

# 3. Run the master pipeline
# from mapping_crisis_driven_market_dynamics_draft import run_master_pipeline
master_results, html_report = run_master_pipeline(
    raw_price_df=raw_price_df,
    study_config=study_config,
    param_grid=param_grid,
    report_format='html'
)

# 4. Save the report and explore results
with open("analysis_report.html", "w", encoding="utf-8") as f:
    f.write(html_report)

# Programmatically access results
robust_links = master_results['robustness_analysis']['persistent_links']['transfer_entropy']
print(robust_links.head())
```

## Output Structure

The `run_master_pipeline` function returns a tuple: `(master_results, report_string)`.

-   `master_results`: A deeply nested dictionary containing all data artifacts. Top-level keys include:
    -   `main_analysis`: Results from the baseline run (processed data, static matrices, dynamic results, interpretations, figures).
    -   `robustness_analysis`: Results from the sensitivity analysis, including robustness scores for key findings.
    -   `error_analysis`: Results from the bootstrapping, including confidence intervals for static estimates.
-   `report_string`: A string containing the full source code of the generated HTML or Markdown report.

## Project Structure

```
mapping_crisis_driven_market_dynamics/
│
├── mapping_crisis_driven_market_dynamics_draft.ipynb  # Main implementation notebook
├── requirements.txt                                   # Python package dependencies
├── LICENSE                                            # MIT license file
└── README.md                                          # This documentation file
```

## Customization

The pipeline is highly customizable via the `study_config` and `param_grid` dictionaries passed to `run_master_pipeline`. Users can easily modify:
-   The `date_range` and `crisis_periods` to analyze different time frames.
-   The `discretization_bins` and `window_size_days` to test different model specifications.
-   All `visualization_params` to change the aesthetic of plots and graphs.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{khalilian2025mapping,
  title={Mapping Crisis-Driven Market Dynamics: A Transfer Entropy and Kramers--Moyal Approach to Financial Networks},
  author={Khalilian, Pouriya and Golestani, Amirhossein N and Eslamifar, Mohammad and Firouzjaee, Mostafa T and Firouzjaee, Javad T},
  journal={arXiv preprint arXiv:2507.09554},
  year={2025}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2025). A Python Implementation of the Transfer Entropy and Kramers-Moyal Framework for Financial Networks. 
GitHub repository: https://github.com/chirindaopensource/mapping_crisis_driven_market_dynamics
```

## Acknowledgments

-   Credit to Pouriya Khalilian, Amirhossein N. Golestani, Mohammad Eslamifar, Mostafa T. Firouzjaee, and Javad T. Firouzjaee for the novel analytical framework.
-   Thanks to the developers of the `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `networkx`, `statsmodels`, and `tqdm` libraries, which are the foundational pillars of this analytical pipeline.

--

*This README was generated to document the code and methodology contained within `mapping_crisis_driven_market_dynamics_draft.ipynb` and follows best practices for open-source research software.*
