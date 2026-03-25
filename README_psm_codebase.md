# Neural Network and Logistic Regression Propensity Score Estimation

This repository provides stand-alone code for simulation-based propensity score research using:

- **R** for data generation, logistic-regression propensity score estimation, and weighting-based evaluation
- **Python** for DNN- and CNN-based propensity score estimation

The public scripts were refactored from project-specific research code so that users can adapt the workflow to their own settings, directory structures, and hyperparameter choices.

## Included scripts

### R
- `psm_data_generation.R`  
  Generates simulation datasets under user-defined correlation structures and treatment prevalence settings.

- `psm_logistic_regression.R`  
  Fits a logistic-regression propensity score model to bundled simulation datasets and saves predicted propensity scores.

- `psm_weighting_evaluation.R`  
  Reads one or more sets of propensity score predictions and computes unweighted, odds-weighted, and IPTW-weighted treatment-versus-control coefficient differences for each covariate and the outcome.

### Python
- `dnn_public.py`  
  DNN propensity score estimation.

- `cnn_public.py`  
  CNN propensity score estimation.

## Recommended folder structure

```text
project_root/
├── basic_info.txt
├── datasets/
│   ├── XX10_TY00_TS10/
│   ├── XX10_TY00_TS25/
│   └── ...
├── datasets_noised/              # optional
├── results/
│   ├── LR/
│   ├── DNN/
│   ├── CNN/
│   └── weighting_evaluation/
├── psm_data_generation.R
├── psm_logistic_regression.R
├── psm_weighting_evaluation.R
├── dnn_public.py
└── cnn_public.py
```

## R package requirements

### Data generation
- `MASS`

### Logistic regression
- `jsonlite`

### Weighting evaluation
- no additional non-base package requirements

## Python package requirements

Install the required Python packages for the DNN and CNN scripts in your environment.

## Typical workflow

### 1. Generate simulation data

```bash
Rscript psm_data_generation.R \
  --root_path ./project_root \
  --n_rep 200 \
  --n_sample 10000 \
  --num_x 18 \
  --r_xxs 0.1,0.3,0.5 \
  --r_xys 0.1,0.3,0.5 \
  --r_xts 0.1,0.2,0.3,0.4,0.5,0.6 \
  --group_sizes 0.1,0.25,0.4 \
  --r_tys 0,0.1,0.3,0.5
```

### 2. Fit logistic-regression propensity score models

```bash
Rscript psm_logistic_regression.R \
  --data_root ./project_root \
  --results_root ./project_root/results \
  --bundle_size 50 \
  --include_squared true \
  --include_interactions true
```

### 3. Fit DNN and CNN propensity score models

Examples:

```bash
python dnn_public.py \
  --data-root ./project_root \
  --results-root ./project_root/results \
  --bundle-size 50 \
  --hidden-units 189,94 \
  --optimizer adam \
  --learning-rate 0.001 \
  --batch-size 256 \
  --epochs 50
```

```bash
python cnn_public.py \
  --data-root ./project_root \
  --results-root ./project_root/results \
  --bundle-size 50 \
  --conv-filters 32,64,96 \
  --dense-units 64,32 \
  --optimizer adam \
  --learning-rate 0.001 \
  --batch-size 128 \
  --epochs 30
```

### 4. Evaluate weighting performance

The `--methods` argument maps a method label to the directory containing that method's prediction files.
Provide the most specific run directory possible for each method.

```bash
Rscript psm_weighting_evaluation.R \
  --data_root ./project_root \
  --output_dir ./project_root/results/weighting_evaluation \
  --bundle_size 50 \
  --methods LR=./project_root/results/LR,CNN=./project_root/results/CNN,DNN=./project_root/results/DNN
```

## Outputs

### Data generation
- `basic_info.txt`
- `datasets/`
- `corr_mats/`
- `perfect_sep/`

### Logistic regression / DNN / CNN
- `bundle_000_predictions.csv`, `bundle_001_predictions.csv`, etc.
- `run_metadata.json`

### Weighting evaluation
- `covariate_effects_long.csv`
- `covariate_effects_summary.csv`

## Notes on public release

This public codebase provides a **generalized and user-configurable implementation**.
It is not intended to reproduce any published manuscript unless the user explicitly matches the original settings, feature engineering choices, bundling structure, and hyperparameters.

## Ownership and licensing

This code is distributed under the **MIT License**.
Copyright remains with **Seungman Kim**.

## Suggested citation

If you use or adapt this code, please cite the related paper and acknowledge the code repository.
