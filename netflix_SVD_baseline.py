import numpy as np
from numba import njit # , jit
import time
import pickle
from functools import wraps
from math import trunc
import pandas as pd
import sys
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space.space import Real, Integer
from skopt import callbacks, load
from skopt.callbacks import CheckpointSaver

import random


def _initialization(n_users, n_items, n_features):
    deviation = 0.1
    bu = np.zeros(n_users)
    bi = np.zeros(n_items)
    pu = np.random.uniform(low=-deviation, high=deviation, size=(n_users, n_features))
    qi = np.random.uniform(low=-deviation, high=deviation, size=(n_items, n_features))
    return bu, bi, pu, qi

@njit
def _run_epoch(X, bu, bi, pu, qi, global_mean_, lr, reg):
    """Runs an epoch, updating model weights (pu, qi, bu, bi).

    Parameters
    ----------
    X : numpy.array
        Training set.
    bu : numpy.array
        User biases vector.
    bi : numpy.array
        Item biases vector.
    pu : numpy.array
        User latent features matrix.
    qi : numpy.array
        Item latent features matrix.
    global_mean : float
        Ratings arithmetic mean.
    n_features : int
        Number of latent features.
    lr : float
        Learning rate.
    reg : float
        L2 regularization feature.

    Returns:
    --------
    bu : numpy.array
        User biases vector.
    bi : numpy.array
        Item biases vector.
    pu : numpy.array
        User latent features matrix.
    qi : numpy.array
        Item latent features matrix.
    Lost Func:
    J(0) = \sigma (r_ui - (bu + bi + pu^T*qi))^2 + reg*(||pu||^2 + ||qi||^2 + ||bu||^2 + ||bi||^2)
    update params:

    """
    n_shape = X.shape[0]
    for i in range(n_shape):
        user, item, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]
        pred = global_mean_ + bu[user] + bi[item] + pu[user].T @ qi[item]
        err = rating - pred
        bu[user] += lr * (err - reg * bu[user])
        pu[user] += lr * (err * qi[item] - reg * pu[user])
        bi[item] += lr * (err - reg * bi[item])
        qi[item] += lr * (err * pu[user] - reg * qi[item])
    return bu, bi, pu, qi


@njit
def _compute_val_metrics(X_test, bu, bi, pu, qi, global_mean_):
    residuals = []
    n_shape = X_test.shape[0]
    bu[-1] = 0
    bi[-1] = 0
    pu[-1] = 0
    qi[-1] = 0
    preds = []

    for i in range(n_shape):
        user, item, rating = int(X_test[i, 0]), int(X_test[i, 1]), X_test[i, 2]
        pred = global_mean_ + bu[user] + bi[item] + pu[user].T @ qi[item]
        preds.append(pred)
        residuals.append(rating - pred)
    residuals = np.array(residuals)
    rmse = np.sqrt(np.square(residuals).mean())
    mae = np.absolute(residuals).mean()
    return preds, rmse, mae


class SVD_baseline:
    def __init__(
        self,
        lr=0.005,
        reg=0.0,
        n_features=100,
        n_epochs=300,
        storefile=False,
        filename="",
        min_rmse=10000.0,
        verbose=True,
    ):

        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.n_features = n_features
        self.storefile = storefile
        self.filename = filename
        self.min_rmse = min_rmse
        self.verbose = verbose

    # @_timer(text="\nDone ")
    def fit(self, X, X_test=None):
        X = self._preprocess_data(X)

        if X_test is not None:
            X_test = self._preprocess_data(X_test, train=False, verbose=False)
            self._init_metrics()
        self.global_mean_ = np.mean(X[:, 2])
        self.std_ = 0
        rmse = self._run_sgd(X, X_test)

        return rmse

    def _preprocess_data(self, X, train=True, verbose=True):
        X = X.copy()
        if train:  # Mappings have to be created)
            user_ids = X["u_id"].unique().tolist()
            item_ids = X["i_id"].unique().tolist()
            n_users = len(user_ids)
            n_items = len(item_ids)
            user_idx = range(n_users)
            item_idx = range(n_items)
            self.user_mapping_ = dict(zip(user_ids, user_idx))
            self.item_mapping_ = dict(zip(item_ids, item_idx))
        X["u_id"] = X["u_id"].map(self.user_mapping_)
        X["i_id"] = X["i_id"].map(self.item_mapping_)

        # Tag validation set unknown users/items with -1 (enables
        # `fast_methods._compute_val_metrics` detecting them)
        X.fillna(-1, inplace=True)
        X["u_id"] = X["u_id"].astype(np.int32)
        X["i_id"] = X["i_id"].astype(np.int32)
        X["rating"] = X["rating"].astype(np.int32)

        return X[["u_id", "i_id", "rating"]].values

    def _init_metrics(self):
        metrics = np.zeros((self.n_epochs, 2), dtype=float)
        self.metrics_ = pd.DataFrame(metrics, columns=["RMSE", "MAE"])

    def _run_sgd(self, X, X_test):

        n_users = len(np.unique(X[:, 0]))
        n_items = len(np.unique(X[:, 1]))
        bu, bi, pu, qi = _initialization(n_users, n_items, self.n_features)
        rmse = 10000.0
        count = 0
        count_lr = 0
        change_rmse = False
        count_epoch = 0
        for epoch_ix in range(self.n_epochs):
            if self.verbose:
                start = self._on_epoch_begin(epoch_ix)
            bu, bi, pu, qi = _run_epoch(
                X=X,
                bu=bu,
                bi=bi,
                pu=pu,
                qi=qi,
                global_mean_=self.global_mean_,
                lr=self.lr,
                reg=self.reg,
            )
            count_epoch += 1
            pu_1 = np.array(pd.DataFrame(pu).mean(axis=0))
            qi_1 = np.array(pd.DataFrame(qi).mean(axis=0))
            bu_1 = np.mean(bu)
            bi_1 = np.mean(bi)

            _, rmse_epoch_ix, mae_epoch_ix = _compute_val_metrics(
                X_test=X_test,
                bu=bu,
                bi=bi,
                pu=pu,
                qi=qi,
                global_mean_=self.global_mean_,
            )
            self.metrics_.loc[epoch_ix, :] = (rmse_epoch_ix, mae_epoch_ix)
            if self.verbose:
                self._on_epoch_end(
                    start,
                    self.metrics_.loc[epoch_ix, "RMSE"],
                    self.metrics_.loc[epoch_ix, "MAE"],
                )
            val_rmse = self.metrics_.loc[epoch_ix, "RMSE"]
            if val_rmse < rmse - 1e-4:
                count = 0
                count_lr = 0
                rmse = val_rmse
                if self.storefile and (val_rmse < self.min_rmse):
                    change_rmse = True
                    self.min_rmse = val_rmse
                    data = [bu, bi, pu, qi, self.global_mean_, self.std_]
            else:
                count += 1
                count_lr += 1
            if (count_lr >= 1) and (self.lr > 1e-4):
                self.lr = self.lr / 10
                count_lr = 0
            if count >= 5:
                break

        print(round(rmse, 4), self.n_features, round(self.reg, 6), count_epoch)
        if self.storefile and change_rmse:
            dbfile = open(self.filename, "wb")
            pickle.dump(data, dbfile)
        return rmse

    def _on_epoch_begin(self, epoch_ix):
        start = time.process_time()
        end = "  | " if epoch_ix < 9 else " | "
        print("Epoch {}/{}".format(epoch_ix + 1, self.n_epochs), end=end)
        return start

    def _on_epoch_end(self, start, val_rmse=None, val_mae=None):
        end = time.process_time()
        if (val_rmse is not None):
            print(f"val_rmse: {val_rmse:.4f}", end=" - ")
            print(f"val_mae: {val_mae:.4f}", end=" - ")
        print(f"took {end - start:.1f} sec")


def _timer(text=""):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.process_time()
            result = func(*args, **kwargs)
            end = time.process_time()

            hours = trunc((end - start) / 3600)
            minutes = trunc((end - start - hours * 3600) / 60)
            seconds = round((end - start) % 60)

            if hours > 1:
                print(
                    text + "{} hours {} min and {} sec".format(hours, minutes, seconds)
                )
            elif hours == 1:
                print(
                    text + "{} hour {} min and {} sec".format(hours, minutes, seconds)
                )
            elif minutes >= 1:
                print(text + "{} min and {} sec".format(minutes, seconds))
            else:
                print(text + "{} sec".format(seconds))
            return result
        return wrapper
    return decorator

##############FIRST PART: TRAINING################
##############COMMENT OUT WHEN TESTING############
# if __name__ == "__main__":
#
#     ##################################
#     path_file_train = r"/home/anhnt/MEGA/AI-Recommender/data/netflix_data/train.txt"
#     path_file_test = r"/home/anhnt/MEGA/AI-Recommender/data/netflix_data/test.txt"
#
#     train = pd.read_csv(
#         path_file_train, sep=",", names=["u_id", "i_id", "rating", "time"]
#     )
#     test = pd.read_csv(
#         path_file_test, sep=",", names=["u_id", "i_id", "rating", "time"]
#     )
#     # del test["time"]
#     #########################################################
#     alg = "baseline"
#     dataset = "netflix"
#     num_epochs = 1000
#     #########################################################
#     space_SVD_baseline = [
#         Real(1e-4, 1e0, "uniform", name="reg"),
#         Integer(50, 500, "uniform", name="n_features"),
#     ]
#
#     @use_named_args(space_SVD_baseline)
#     def f_SVD_baseline(reg, n_features):
#         # print(round(reg, 6), n_features)
#         try:
#             res_ = load(path + filename_pkl)
#             min_rmse = res_.fun
#             x = res_.x
#             x_old = res_.x_iters
#         except:
#             min_rmse = 100000.0
#             x = []
#             x_old = []
#         # print(round(min_rmse, 4), x, len(x_old))
#         estimator = SVD_baseline(
#             lr=5e-3,  # ml: 5e-3 ok, yelp: 1e-3 ok
#             reg=reg,
#             n_features=n_features,
#             n_epochs=num_epochs,
#             storefile=True,
#             filename=path + filename_params,
#             min_rmse=min_rmse,
#             verbose=True,
#         )
#         a = estimator.fit(train, test)
#
#         return a
#
#     ##############################################################
#     prefix = dataset + "_" + alg
#     func = f_SVD_baseline
#     dimensions = space_SVD_baseline
#
#     filename_txt = prefix + ".txt"
#     filename_pkl = prefix + ".pkl"
#     filename_params = prefix + "_params"
#     # path = "/home/anhnt/results_baseline/"
#     path = "/tmp/"
#     sys.stdout = open(path + filename_txt, "a", buffering=1)
#     sys.stderr = sys.stdout
#     pkl = CheckpointSaver(path + filename_pkl, compress=9)
#
#     try:
#         res = load(path + filename_pkl)
#         x_0 = res.x_iters
#         y_0 = res.func_vals
#         n_init_points = 10 - len(x_0)
#         if n_init_points <= 0:
#             n_init_points = 1
#         ncalls = 100 - len(x_0)
#         r_state = res.random_state
#         b_estimator = res.specs["args"]["base_estimator"]
#     except:
#         x_0 = None
#         y_0 = None
#         n_init_points = 10
#         ncalls = 100
#         r_state = None
#         b_estimator = None
#
#     res = gp_minimize(
#         func=func,
#         dimensions=dimensions,
#         n_calls=ncalls,
#         verbose=True,
#         x0=x_0,
#         y0=y_0,
#         n_initial_points=n_init_points,
#         random_state=r_state,
#         base_estimator=b_estimator,
#         callback=[pkl],
#     )
#     print(res.fun, res.x)
#     sys.stdout.close()


#######SECOND PART: TESTING#######################
#### COMMENT OUT WHEN TRAINING####################

path_file_train = r"/home/anhnt/MEGA/AI-Recommender/data/netflix_data/train.txt"
path_file_val = r"/home/anhnt/MEGA/AI-Recommender/data/netflix_data/val.txt"

train = pd.read_csv(
    path_file_train, sep=",", names=["u_id", "i_id", "rating", "timestamps"]
)
val = pd.read_csv(
    path_file_val, sep=",", names=["u_id", "i_id", "rating", "timestamps"]
)


def _preprocess_train(X):
    user_ids = X["u_id"].unique().tolist()
    item_ids = X["i_id"].unique().tolist()
    n_users = len(user_ids)
    n_items = len(item_ids)
    user_idx = range(n_users)
    item_idx = range(n_items)
    user_mapping_ = dict(zip(user_ids, user_idx))
    item_mapping_ = dict(zip(item_ids, item_idx))
    return user_mapping_, item_mapping_


def _preprocess_val(X, user_mapping_, item_mapping_):
    X["u_id"] = X["u_id"].map(user_mapping_)
    X["i_id"] = X["i_id"].map(item_mapping_)
    X.fillna(-1, inplace=True)
    X["u_id"] = X["u_id"].astype(np.int32)
    X["i_id"] = X["i_id"].astype(np.int32)
    X["rating"] = X["rating"].astype(np.int32)
    return X[["u_id", "i_id", "rating"]].values


def evaluate_baseline(f):
    list_params = pickle.load(f)
    bu = list_params[0]
    bi = list_params[1]
    pu = list_params[2]
    qi = list_params[3]
    global_mean_ = list_params[4]

    pu_1 = np.array(pd.DataFrame(pu).mean(axis=0))
    qi_1 = np.array(pd.DataFrame(qi).mean(axis=0))
    bu_1 = np.mean(bu)
    bi_1 = np.mean(bi)

    return _compute_val_metrics(
        X_test=user_item_rating_from_val_set,
        bu=bu,
        bi=bi,
        pu=pu,
        qi=qi,
        global_mean_=global_mean_,
    )


user_mapping_, item_mapping_ = _preprocess_train(train)
user_item_rating_from_val_set = _preprocess_val(val, user_mapping_, item_mapping_)

path = "/tmp/"
dataset = "netflix"
#################################################################################

f = open(path + dataset + "_baseline" + "_params", "rb")
print(evaluate_baseline(f)[1])
