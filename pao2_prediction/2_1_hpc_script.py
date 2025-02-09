# - CPUs per task: 128
# - RAM: 128Gb
# - Python 3.10.12 (main, Jun 11 2023, 05:26:28) [GCC 11.4.0]
# - Linux-5.15.0-127-generic-x86_64-with-glibc2.35

# Packages:

# - joblib              1.3.2
# - numpy               1.26.0
# - scipy               1.11.4
# - session_info        1.0.0
# - sklearn             1.3.1

# Script arguments:

# - number of threads = 16
# - number of parallel jobs = 16


import time, pickle, session_info, traceback, argparse
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from pathlib import Path

from joblib import Parallel, delayed, parallel_config

# https://stackoverflow.com/questions/76532825/joblib-on-a-slurm-cluster-lokyprocess-failed


def cv_rfe(estimator, X, y, groups, cv, num_threads=1, num_jobs=1, scoring='neg_root_mean_squared_error', min_features_to_select=1)->dict:
    """
    Perform Cross-Validated Recursive Feature Elimination (CV-RFE) on the given estimator.

    Parameters
    ----------
    estimator : object
        A scikit-learn estimator that supports the `fit` method.
    X : array-like, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples,)
        The target values.
    groups : array-like, shape (n_samples,)
        The groups for each sample.
    cv : int or cross-validation generator
        The number of folds (k) or a cross-validation generator.
    scoring : str, default='neg_root_mean_squared_error'
        The scoring metric to use for feature selection. It should be a string that corresponds to a valid scoring metric for the given estimator. Examples include 'accuracy', 'roc_auc', 'f1', etc. The default value is 'neg_root_mean_squared_error', which is the negative square root of the mean squared error (RMSE) of a regression problem.
    min_features_to_select : int, default=1
        The minimum number of features to select. The function will select `min_features_to_select` features and continue to remove features until only one feature remains.

    Returns
    -------
    cv_scores_dict: dict. 
        A dictionary containing the selected features and their corresponding CV scores. The keys represent the number of features used, and the values are dictionaries with the keys 'features' and 'cv_score'. The value of 'features' is a list of tuples, where each tuple represents a feature and its corresponding CV score. The value of 'cv_score' is the average CV score for the selected features.
    """
    n_features = X.shape[1]
    cv_scores_dict = {i: {} for i in range(n_features, min_features_to_select - 1, -1)}

    print(f"{num_threads} threads with {num_jobs} parallel jobs = {num_threads*num_jobs} CPUs needed.")
    with parallel_config(backend="loky", inner_max_num_threads=num_threads):
        for i in range(n_features, min_features_to_select - 1, -1):
            print(f"\tFind {i} features...")
            start_time = time.time()
            support_ = np.ones(n_features, dtype=bool) # support for all of the features
            cv_scores = np.zeros(n_features) # cv score is zero for every feature
            while sum(support_) != i: # while sum of supported features is not the same as the number of features investigated
                def cv_score_j(j): # calculate cv score for specific feature
                    if support_[j]: # if feature is supported
                        support_[j] = False
                        X_selected = X[:, support_] # get all supported features without the j-th one
                        cv_scores[j] = np.mean(cross_val_score(estimator, X_selected, y.ravel(), groups=groups, cv=cv, scoring=scoring)) # calculate cv score
                        support_[j] = True # set support for j-th feature to true again
                    return cv_scores[j] # return its cv score
                cv_scores = np.array(Parallel(n_jobs=num_jobs)(delayed(cv_score_j)(j) for j in range(n_features))) # calculate cv score for every supported feature in this iteration
                ranking = stats.rankdata(-cv_scores, method='ordinal') # rank the cv scores
                idx = np.argmax(ranking) # get the highest one
                support_[idx] = False # remove the feature with the highest (lowest) score
                cv_scores[idx] = 0 # set cv score for this feature to 0

            tmp_selected_features = np.array(feature_list)[support_]
            tmp_cv_scores = cv_scores[support_]
            cv_scores_dict[i]['features'] = list(zip(tmp_selected_features, tmp_cv_scores))
            cross_val_scores = cross_val_score(estimator, X[:, support_], y.ravel(), groups=groups, cv=cv, scoring=scoring, n_jobs=num_jobs) # calculate the cv score for the model with the number of selected features
            cv_scores_dict[i]['cv_score'] = np.mean(cross_val_scores)
            std_error = np.std(cross_val_scores, ddof=1) / np.sqrt(5)
            t_value = stats.t.ppf(1 - 0.025, df=4)
            cv_scores_dict[i]['confidence_interval'] = (np.mean(cross_val_scores) - t_value * std_error,
                                                        np.mean(cross_val_scores) + t_value * std_error)
            end_time = time.time()
            duration = (end_time - start_time) / 60
            print(f"\tDuration for this iteration: {duration:.2f} minutes.")

    return cv_scores_dict

if __name__ == "__main__":
    print(session_info.show())
    np.random.seed(42)
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-e", "--estimator", help="estimator to use; if multiple are provided, please separate with '_'")
    argParser.add_argument("-j", "--num_jobs", help="nummber of jobs", type=int, default=1)
    argParser.add_argument("-t", "--num_threads", help="number of threads", type=int, default=1)
    args = argParser.parse_args()
    total_start = time.time()

    algorithm_dict = {
        "gbr": GradientBoostingRegressor(random_state=42,),
        "knn": KNeighborsRegressor(),
        "rfr": RandomForestRegressor(random_state=42),
        "svr": SVR(max_iter = -1),
        "sgd": SGDRegressor(random_state=42, max_iter = 100_000_000),
        "mlr": LinearRegression(),
        "mlp": MLPRegressor(random_state = 42)
    }

    feature_list_path = Path(f"./rfecv_data/1_03_feature_list.pickle")
    if feature_list_path.exists():
        feature_list = pickle.load(open(feature_list_path, "rb"))
    else:
        print(f"File not found in directory data: {[item.name for item in Path('./rfecv_data/').iterdir()]}")
        exit()

    print(feature_list)
    train_test_data = pickle.load(open(f"./rfecv_data/1_05_train_test_data.pickle", "rb"))
    x_train_scaled = train_test_data.get("train").get("x_scaled")
    y_train_scaled = train_test_data.get("train").get("y_scaled")
    training_groups = train_test_data.get("train").get("group")
    group_kfold = train_test_data.get("groupkfold")
    np.random.seed(42)

    for name in args.estimator.split('_'):
        if name not in algorithm_dict.keys():
            continue
        print(name)
        start = time.time()
        stopping = False
        counter = 10
        while stopping == False or counter > 0:
            try:
                cv_scores_dict = cv_rfe(algorithm_dict.get(name), x_train_scaled[:, 1:], y_train_scaled[:, 1:], training_groups, group_kfold, min_features_to_select=5, num_threads=args.num_threads, num_jobs=args.num_jobs)
                stopping = True
                counter = 0
            except Exception as e:
                print(f"Error, try again!\n{repr(e)}")
                with open(f'./rfecv_data/error_logs.txt', 'a') as f:
                    f.write(f"{algorithm_dict.get(name)}: {repr(e)}\n{traceback.format_exc()}\n\n")
            counter -= 1
        print(f"Ended with RFECV for {name} after a total of {round((time.time()-start)/60,2)} min.")
        with open(f"./rfecv_data/out/2_01_cv_scores_dict_{name}.pickle","wb") as features_cv_pickle:
            pickle.dump(cv_scores_dict, features_cv_pickle)

    print(f"Ended with RFECV after a total of {round((time.time()-total_start)/60,2)} min.")
    