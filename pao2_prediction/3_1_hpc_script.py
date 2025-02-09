# - CPUs per task: 128
# - RAM: 128Gb
# - Python 3.10.12 (main, Jun 11 2023, 05:26:28) [GCC 11.4.0]
# - Linux-5.15.0-127-generic-x86_64-with-glibc2.35

# Packages:

# - joblib              1.3.2
# - numpy               1.26.0
# - pandas              2.1.1
# - scipy               1.11.4
# - session_info        1.0.0
# - sklearn             1.3.1

# Script arguments:

# - number of threads = 16
# - number of parallel jobs = 16

import time, pickle, os, session_info, traceback, argparse, pprint, platform

from datetime import datetime
from time import perf_counter

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn import preprocessing, metrics
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import sklearn
import warnings
from sklearn.exceptions import ConvergenceWarning

from pathlib import Path

from joblib import Parallel, delayed, parallel_config


def cal_adjR2(y_true, y_pred, p):
    r2 = metrics.r2_score(y_true, y_pred)
    n = len(y_true) 
    adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
    return adjusted_r2

def perform_grid_search(name: str, slicing = 0, num_threads=1, num_jobs=1, verbose = 1) -> dict:
    start = perf_counter()
    if slicing >= len(param_grid[name]):
        return f"Grid Search for {name} ended with {slicing} slices (max. slice was {slicing})."
    print(
        f"\nStarting with {name} at {datetime.strftime(datetime.now(), '%d.%m.%y %H:%M:%S')} for slice {slicing} and hyperparameters:\n{pp.pformat(param_grid[name][slicing])}.\n"
    )
    grid_search = GridSearchCV(
        estimator=algorithm_dict.get(name),
        param_grid=param_grid.get(name)[slicing],
        cv=train_test_data.get("groupfold"),
        scoring= {
            "RMSE": "neg_root_mean_squared_error", 
            "adjR2": metrics.make_scorer(cal_adjR2, p=len(algo_train_data_dict.get(name).get('selected_features')))
        }, #"neg_root_mean_squared_error",
        refit = False,
        n_jobs = num_jobs,
        verbose = verbose,
    )
    return_value = False
    try:
        grid_search.fit(
            algo_train_data_dict.get(name).get("x_train"),
            train_test_data.get("train").get("y_scaled")[:, 1],
            groups=train_test_data.get("train").get("group"),
        )
        pd.DataFrame(grid_search.cv_results_).to_csv(f"./tuning/out/cv_results_{name}{slicing}.csv")
        return_value = True
    except Exception as e:
        with open('./tuning/tuning_errors.txt', 'a') as error_file:
            error_file.write(f"Fitting error: {name} at {slicing} -> {repr(e)}:\n########################################\n\t{traceback.format_exc()}########################################\n")
        return_value =  f"{repr(e)}"

    finally:
        print(
            f"Done with {name} slice {slicing} at {datetime.strftime(datetime.now(), '%d.%m.%y %H:%M:%S')}.\nTotal time: {round((perf_counter()-start)/60,2)} min.\n"
        )
        return return_value

if __name__ == "__main__":
    np.random.seed(42)
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-e", "--estimator", help="estimator to use; if multiple estimators are provided, please separate with '_'")
    argParser.add_argument("-j", "--num_jobs", help="nummber of jobs", type=int, default=1)
    argParser.add_argument("-t", "--num_threads", help="number of threads", type=int, default=1)
    argParser.add_argument("-s", "--slice", help="which slice of hyperparameters", type=int, default=99)

    args = argParser.parse_args()
    names = args.estimator.split('_')

    num_threads=args.num_threads
    num_jobs=args.num_jobs
    slicing = args.slice

    pp = pprint.PrettyPrinter(indent=4)

    print(session_info.show())
    print(platform.system(), platform.release(), platform.freedesktop_os_release())

    algorithm_dict = {
        "gbr": GradientBoostingRegressor(random_state=42,),
        "knn": KNeighborsRegressor(),
        "rfr": RandomForestRegressor(random_state=42),
        "svr": SVR(max_iter = -1),
        "sgd": SGDRegressor(random_state=42, max_iter = 100_000_000),
        "mlr": LinearRegression(),
        "mlp": MLPRegressor(random_state=42, max_iter=500)
    }

    with open(f"./tuning/1_05_train_test_data.pickle", "rb") as train_test_pickle:
        train_test_data = pickle.load(train_test_pickle)
    with open(f"./tuning/3_02_completed_train_data.pickle", "rb") as algo_pickle:
        algo_train_data_dict = pickle.load(algo_pickle)
    with open(f"./tuning/3_04_hyperparameters.pickle", "rb") as param_pickle:
        param_grid = pickle.load(param_pickle)

    error_slices = []

    if slicing == 99:
        for name in names:
            print(f"Found {len(param_grid[name])} slice(s) for {name}.\n")
            for slicing in range(len(param_grid[name])):
                try:
                    print(f"\n\nTrying slice {slicing} for algorithm {name}.")
                    print(f"{num_threads} threads with {num_jobs} parallel jobs = {num_threads*num_jobs} CPUs needed.")
                    with parallel_config(backend="loky", inner_max_num_threads=num_threads):
                        result = perform_grid_search(name, slicing=slicing, num_threads=num_threads, num_jobs=num_jobs, verbose=1)
                except Exception as e:
                    result = False
                    with open('./tuning/tuning_errors.txt', 'a') as error_file: 
                        error_file.write(f"NO fitting error: {name} at {slicing} -> {repr(e)}:\n########################################\n\t{traceback.format_exc()}########################################\n")
                    print(f"Error with {name} at slice {slicing}: {repr(e)}\nthis is not a fitting error!")
                finally:
                    if result == True:
                        print(f"Successfully finished grid search.")
                    else:
                        print(f"An error occured, result: {result}.")
                continue
    else:
        for name in names:
            try:
                print(f"\n\nTrying slice {slicing} for algorithm {name}.")
                print(f"{num_threads} threads with {num_jobs} parallel jobs = {num_threads*num_jobs} CPUs needed.")
                with parallel_config(backend="loky", inner_max_num_threads=num_threads):
                    result = perform_grid_search(name, slicing=slicing, num_threads=num_threads, num_jobs=num_jobs, verbose=1)
            except Exception as e:
                result = False
                with open('./tuning/tuning_errors.txt', 'a') as error_file: 
                    error_file.write(f"NO fitting error: {name} at {slicing} -> {repr(e)}:\n########################################\n\t{traceback.format_exc()}########################################\n")
                print(f"Error with {name} at slice {slicing}: {repr(e)}\nthis is not a fitting error!")
            except ConvergenceWarning as w:
                result = False
                print(f"Error with {name} at slice {slicing}: {repr(w)}")
            finally:
                if result == True:
                    print(f"Successfully finished grid search.")
                else:
                    print(f"An error occured, result: {result}.")

