# FR-GLTSK

This repository provides a Python implementation of the FR-GLTSK model 
---

## Requirements

- Python >= 3.11  
- PyTorch >= 2.3.1  
- NumPy, SciPy, scikit-learn  

All experiments were conducted under a standard Python scientific computing environment.

---
## Hyperparameter Settings and Tuning

The hyperparameters used in the experiments, including learning rate, regularization coefficients, and threshold parameters, are specified in the experiment scripts under the `experiments` directory.

- First, determine the regularization coefficients (`lambda1` and `lambda2`) within the range [0.0001, 0.001] to ensure the model forms reasonable sparsity.  
- Next, set the threshold parameters (`gamma1` and `gamma2`) based on dataset dimensionality:  
  - For low-dimensional datasets, use the 75th percentile (P75) and adjust `gamma` within [-1, 1].  
  - For high-dimensional datasets, use the maximum value (P100) and adjust `gamma` within [0, 1].

Multiple parameter combinations within these ranges are predefined and evaluated. The final configuration is selected by jointly considering training error, model performance, and sparsity of selected features and extracted rules.

---
## Cross-Validation

Ten-fold cross-validation is employed to ensure robust evaluation. The dataset is split into ten subsets, iteratively used for training and validation. Final performance metrics are averaged across all folds and repeated runs.
