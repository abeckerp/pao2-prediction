import pandas as pd
import numpy as np
from sklearn import metrics
import scipy.stats as stats
import math
from sklearn.base import clone
import matplotlib.pyplot as plt

def evaluate(
    predictions: list, test_data: list, p: int
) -> dict:
    """
    Evaluates predictions and test labels by returning evaluation metrics
    IN: predictions, test labels, y scaler, number of features
    OUT: Dictionary with mean absolute percentage error, mean absolute error,
         root mean squared error, standard deviation of errors, adjusted R2, Spearman's rho
         with its 95% confidence interval
    """
    predictions = np.array(predictions).reshape(-1,1)
    pooled_paO2 = [np.mean(test_data[test_data[:,0] == opid, 1]) for opid in np.unique(test_data[:,0])]
    prediction_data = np.zeros((predictions.shape[0], predictions.shape[1] + 1))
    prediction_data[:,0] = test_data[:,0]
    prediction_data[:,1] = predictions[:,0]
    pooled_predictions = [np.mean(prediction_data[prediction_data[:,0] == opid, 1]) for opid in np.unique(prediction_data[:,0])]

    num = len(test_data)

    # spearman's rho
    correlation = stats.spearmanr(pooled_paO2, pooled_predictions)
    r = correlation[0]
    stderr = 1 / math.sqrt(num - 3)
    delta = 1.96 * stderr
    lower = math.tanh(math.atanh(r) - delta)
    upper = math.tanh(math.atanh(r) + delta)

    # errors
    errors = predictions - test_data[:,1]

    mae = np.mean(abs(errors))

    mape = 100 * metrics.mean_absolute_percentage_error(test_data[:,1], predictions)

    mse = metrics.mean_squared_error(test_data[:,1], predictions)
    rmse = np.sqrt(mse)

    # adjusted R2
    r2 = metrics.r2_score(test_data[:,1], predictions)
    n = len(test_data[:,1]) 
    adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))

    return {
        "mape": mape,
        "mae": mae,
        "rmse": rmse,
        "adjusted_r2": adjusted_r2,
        "rho": r,
        "ci_rho_lower": lower,
        "ci_rho_upper": upper,
    }


# lambda function for quicker re-scaling of scaled values (array!)
inverse_transform_shaped = lambda scaler, scaled: scaler.inverse_transform(
    scaled.reshape(-1, 1)
).reshape(1, -1)[0]


def get_evaluation(rmse_list: list) -> dict:
    """
    Returns statistics for list of cross validated root mean squared errors
    IN: list of cross validated root mean squared errors
    OUT: mean value my, 95% confidence interval, rmse list
    """

    my = np.mean(rmse_list)
    lower, upper = stats.t.interval(
        0.95, len(rmse_list) - 1, loc=np.mean(rmse_list), scale=stats.sem(rmse_list)
    )
    return {"my": my, "upper": upper, "lower": lower, "values": rmse_list}


def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """
    Create a sample plot for indices of a cross-validation object.
    IN:
    OUT:
    """

    cmap_data = plt.cm.Paired
    cmap_cv = plt.cm.coolwarm
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group), start=1):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Formatting
    yticklabels = list(range(1, n_splits + 1, 1))
    ax.set(
        yticks=np.arange(n_splits) + 1.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
        # xlim=[0, 100],
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)
    return ax


return_table = lambda df, var_list: df.groupby(var_list).size()


def plot_predictions(y, y_pred, x_label, y_label, title, plotname):
    plt.rcParams["figure.figsize"] = [8, 8]
    plt.scatter(y, y_pred, edgecolors=(0, 0, 0))
    plt.plot(
        [y.min(), y.max()],
        [y_pred.min(), y_pred.max()],
        "k--",
        lw=2,
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(0,610)
    plt.ylim(0,610)
    plt.title(title)
    plt.savefig(plotname, dpi=300, bbox_inches="tight")
    plt.show() 