# MDFS: Multi-view Deep Feature Selection Framework

This repository contains the official implementation of **MDFS** (Multi-view Deep Feature Selection), a versatile deep learning-based framework designed to select informative features from multi-view data.

The project supports three core machine learning tasks:
1.  **Regression**
2.  **Classification**
3.  **Clustering** (Unsupervised)

## 📂 Project Structure

The project is organized as follows:

```text
MDFS_Gemini/
├── src/                        # Source code directory
│   ├── methods/                # Implementation of MDFS and baseline methods
│   │   ├── regression/         # Regression models (Proposed, AdaCoop, MSGLasso, SLRFS)
│   │   ├── classification/     # Classification models (Proposed, Baselines)
│   │   └── clustering/         # Clustering models (Proposed, MCFL, MRAG, NSGL)
│   ├── simulations/            # Data generation and experiment logic
│   └── utils/                  # Utility functions (metrics, logging, etc.)
├── scripts/                    # Executable scripts for running simulations
│   ├── run_sim_1_regression.py
│   ├── run_sim_2_classification.py
│   └── run_sim_3_clustering.py
├── results/                    # Output directory for simulation results (auto-generated)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## ⚙️ Environment Setup

It is recommended to use a virtual environment (e.g., Conda) to manage dependencies.

### 1. Create and Activate Environment

```bash
conda create -n mdfs_env python=3.9
conda activate mdfs_env
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
* `numpy`, `scipy`, `pandas`: Data manipulation.
* `torch`: Deep learning framework for MDFS.
* `scikit-learn`: Baselines and evaluation metrics.
* `dcor`: Distance correlation for feature screening.
* `tqdm`: Progress bars.

### 3. R Environment (Optional)

If you intend to run the `MSGLasso` baseline for regression, you must have **R** installed and configured, as it relies on the `MSGLasso` R package via `rpy2`.

* Ensure R is installed and added to your system PATH.
* The code includes a wrapper (`src/methods/regression/baselines.py`) that attempts to set up the R environment automatically.

## 🚀 How to Run

There are three main simulation scripts corresponding to the three tasks. Each script will:
1.  Generate synthetic multi-view data.
2.  Train the proposed **MDFS** model and comparison **Baselines**.
3.  Calculate metrics (Precision, Recall) for feature selection.
4.  Save results to the `results/` folder and print a summary.

### 1. Regression Task

```bash
python scripts/run_sim_1_regression.py
```

### 2. Classification Task

```bash
python scripts/run_sim_2_classification.py
```

### 3. Clustering Task

```bash
python scripts/run_sim_3_clustering.py
```

## 🧠 Implemented Methods

### Proposed Method
* **MDFS (Multi-view Deep Feature Selection)**: A neural network-based feature selection method that utilizes a sparse gating mechanism and information bottleneck principle to identify task-relevant features across multiple views.
    * **MDFSRegressor**: For continuous targets.
    * **MDFSClassifier**: For binary or multiclass targets.
    * **MDFSClustering**: An unsupervised variant using reconstruction and regularization losses.

### Baseline Methods

We compare against the following state-of-the-art methods:

#### Regression Baselines
* **AdaCoop** (Ding et al., 2022): Cooperative Learning for Multi-view Analysis.
* **MSGLasso**: Multivariate Sparse Group Lasso (Wrapper for R package).
* **SLRFS** (Hu et al., 2017): Low-rank feature selection for multi-view regression.

#### Clustering Baselines
* **MCFL** (Wang et al., 2013): Multi-view Clustering and Feature Learning via Structured Sparsity.
* **MRAG** (Jing et al., 2021): Learning Robust Affinity Graph Representation for Multi-view Clustering.
* **NSGL** (Bai et al., 2020): Multi-view Feature Selection via Nonnegative Structured Graph Learning.

## 📊 Results

After running the scripts, the detailed results for each random seed will be saved in CSV format in the `results/` directory:
* `sim_1_regression_results.csv`
* `sim_2_classification_results.csv`
* `sim_3_clustering_results.csv`

The console output will display the **Mean ± Standard Deviation** for key metrics (e.g., `recall_total`, `precision_total`) across all runs.

## 📝 License

This project is open-sourced under the MIT License.