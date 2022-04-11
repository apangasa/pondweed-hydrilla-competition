from __future__ import annotations
from random import random

import processData
from globals import *

from typing import *

import numpy as np
from sklearn import linear_model, model_selection, metrics, tree, ensemble


def get_metrics(feature_matrix: np.ndarray, y_col: np.ndarray, model: linear_model._base.LinearModel) -> tuple[float]:
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        feature_matrix, y_col, train_size=TRAIN_PROPORTION, random_state=SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2: float = metrics.r2_score(y_test, y_pred)
    mae: float = metrics.mean_absolute_error(y_test, y_pred)
    mse: float = metrics.mean_squared_error(y_test, y_pred)

    return r2, mae, mse


def coeff_imp(psi_bar: np.ndarray, y_col: np.ndarray, model: linear_model._base.LinearModel) -> tuple[np.ndarray]:
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        psi_bar, y_col, train_size=TRAIN_PROPORTION, random_state=SEED)
    model.fit(X_train, y_train)

    # y_pred = model.predict(X_test)
    # r2 = metrics.r2_score(y_test, y_pred)
    # mae = metrics.mean_absolute_error(y_test, y_pred)
    # mse = metrics.mean_squared_error(y_test, y_pred)

    coeffs: np.ndarray = model.coef_
    sorted_feature_indices: np.ndarray = np.flip(np.argsort(np.abs(coeffs)))
    return sorted_feature_indices, coeffs


def cart_imp(psi_bar: np.ndarray, y_col: np.ndarray, model: tree.DecisionTreeRegressor | ensemble.RandomForestRegressor) -> tuple[np.ndarray]:
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        psi_bar, y_col, train_size=TRAIN_PROPORTION, random_state=SEED)
    model.fit(X_train, y_train)

    # y_pred = model.predict(X_test)
    # r2 = metrics.r2_score(y_test, y_pred)
    # mae = metrics.mean_absolute_error(y_test, y_pred)
    # mse = metrics.mean_squared_error(y_test, y_pred)

    coeffs: np.ndarray = model.feature_importances_
    sorted_feature_indices: np.ndarray = np.flip(np.argsort(np.abs(coeffs)))
    return sorted_feature_indices, coeffs


def main():
    psi_bar, y = processData.process_data()

    y_pis = y[:, 0]
    y_hva = y[:, 1]

    y_col = y_pis

    s_list = range(10, 101, 10)

    coeff_models = [
        linear_model.LinearRegression(),
        # linear_model.Lasso(),
        linear_model.Ridge(),
        # linear_model.ElasticNet()
    ]

    indices_list = []

    for model in coeff_models:
        indices, coeffs = coeff_imp(psi_bar, y_col, model)

        max_r2: float = None
        max_r2_s: int = 0
        min_mse: float = 0

        for s in s_list:
            r2, _, mse = get_metrics(psi_bar[:, indices[:s]], y_col, model)

            if max_r2 is None or r2 > max_r2:
                max_r2 = r2
                max_r2_s = s
                min_mse = mse
        print(
            f'For {model}, the optimal value of s={max_r2_s}, yielding test R^2 = {max_r2} and MSE = {min_mse}')
        indices_list.append(indices)

    s = 30
    linear_indices = set(indices_list[0][:s])
    ridge_indices = set(indices_list[1][:s])

    u = linear_indices.union(ridge_indices)
    print(len(u))

    cart_models = [
        tree.DecisionTreeRegressor(),
        ensemble.RandomForestRegressor()
    ]

    for model in cart_models:
        indices, coeffs = cart_imp(psi_bar, y_col, model)

        max_r2: float = None
        max_r2_s: int = 0
        min_mse: float = 0

        for s in s_list:
            r2, _, mse = get_metrics(psi_bar[:, indices[:s]], y_col, model)

            if max_r2 is None or r2 > max_r2:
                max_r2 = r2
                max_r2_s = s
                min_mse = mse
        print(
            f'For {model}, the optimal value of s={max_r2_s}, yielding test R^2 = {max_r2} and MSE = {min_mse}')
        indices_list.append(indices)

    s = 50
    dt_indices = set(indices_list[2][:s])
    rf_indices = set(indices_list[3][:s])

    u = dt_indices.union(rf_indices)
    print(len(u))

    s = 50

    indices = indices_list[1][:s]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        psi_bar[:, indices], y_col, train_size=TRAIN_PROPORTION, random_state=SEED)

    param_grid = {'alpha': [1, 1.95]}
    reg = linear_model.Ridge(
        # max_iter=10000
    )
    model = search_train_eval(
        reg, param_grid, X_train, y_train)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(metrics.mean_squared_error(y_test, y_pred))


def do_grid_search(model, param_grid, x, y):
    gs = model_selection.GridSearchCV(
        model, param_grid, scoring='neg_mean_squared_error', error_score='raise')
    gs_res = gs.fit(x, y)
    return gs_res.best_params_


def search_train_eval(model, param_grid, tr_x, tr_y):
    search_x = tr_x
    search_y = tr_y

    hyperparams = do_grid_search(model, param_grid, search_x, search_y)

    class_obj = type(model)
    m = class_obj(**hyperparams).fit(tr_x, tr_y)

    cn = str(class_obj).split("'")[1]
    cn = cn.split('.')[-1]
    print('{}({})'.format(cn, hyperparams))

    return m


if __name__ == '__main__':
    main()
