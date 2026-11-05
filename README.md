# Personalized Federated Hierarchical Gaussian Processes for Privacy-Preserving Modeling of Heterogeneous Distributed Systems

Code for the paper *Personalized Federated Hierarchical Gaussian Processes for Privacy-Preserving Modeling of Heterogeneous Distributed Systems* by Xianjian Xie and Hao Yan.

## Installation

Tested with Python 3.8.18 on CPU.

```bash
conda create -n pfedhgp python=3.8.18
conda activate pfedhgp
pip install -r requirements.txt
```

## Data

The data files are included in the repository:

- `tonnage.mat`: tonnage signal data used by `tonnage_classification.py`
- `beijing/PRSA_Data_*.csv`: Beijing multi-site air-quality data (12 stations, 2013-03-01 to 2017-02-28) used by `climate_clustering.py`

`numerical_study.py` uses simulated data and needs no input files.

## Running the code

Run all commands from the repository root.

### 1. Numerical study

```bash
python numerical_study.py
```

By default this runs 30 seeds over all scenarios and models and writes to `numerical_results/`:

- `tables/results_summary_seedlevel.csv`
- `figures/*.png`
- `models/<scenario>/*.pt`
- `environment.txt`

Useful options:

```bash
python numerical_study.py                                 # full run
python numerical_study.py --n-seeds 5 --device cuda       # fewer seeds, on GPU
python numerical_study.py --scenarios matched correlated  # subset of scenarios
python numerical_study.py --help                          # all options
```

### 2. Tonnage classification

```bash
python tonnage_classification.py
```

Reads `tonnage.mat`, and writes to `tonnage_result.csv`.

### 3. Climate clustering (Beijing air quality)

```bash
python climate_clustering.py
```

Reads `beijing/PRSA_Data_*.csv` and writes to `climate_results/`:

- `clustering_result1.png`
- `map1.png`
- `clustering_result1.json`

