
# paO<sub>2</sub> Prediction in Neurosurgical Patients

The repository contains the code (quarto notebooks) for paO<sub>2</sub> prediction in neurosurgical patients. Four notebooks are needed. The first one is used for preprocessing like data cleaning, missing values, scaling the data and feature selection. The second one is used for hyperparameter tuning using a grid search approach. The third one evaluates the tuned estimators. The fourth one includes an additional variable, the first measured p/F ratio to improve prediction.

## Authors

- [@abeckerp](https://gitlab.lrz.de/abeckerp)

## Install and run the project

To run the project, the following software and python packages have to be installed:

- Quarto: >= 1.3.450
- Python: 3.12.0
    - pandas: 2.1.1
    - matplotlib: 3.8.0
    - seaborn: 0.13.0
    - sklearn: 1.3.1
    - numpy: 1.26.0
    - scipy: 1.11.3
    - statsmodels: 0.14.0
    - yaml: 6.0.1
    - notebooks: 7.0.4
    - shap: 0.46.0

Clone the repository into your local directory, activate the virtualenv (`poetry shell`) and run the desired .qmd file using `quarto run {filename}`.

## Files

The repository contains the following files:

1. `config.yaml`: configuration file containing file names, plotting settings, threshold value, name of the best regressor (needed for `4_Evaluation.qmd` and `5_First_Horowitz.qmd`)
2. `1_Preprocessing.qmd`: Preprocessing, exclusion criteria, missing data, creation of training and test datasets, scaling of data. Please note, that this file cannot be run due to the unavailability of the raw dataset (EU privacy issues).
3. `2_RFE.qmd`: Recursive feature elimination, using results from `2_1_hpc_script.py` which is run on a high-performance cluster.
4. `3_Hyperparameters_GridSearch.qmd`: Definition of hyperparameters and selection of best hyperparameters after grid search, using results from `3_1_hpc_script.py` which is run on a high-performance cluster.
5. `4_Evaluation.qmd`: Evaluation of all algorithms and calculation of performance matrix.
6. `5_First_Horowitz.qmd`: Refitting the best algorithm using the first measured p/f ratio of each patient.
7. `custom_functions.py`: evaluation and plotting functions which are imported into the .qmd files.

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)