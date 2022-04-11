from __future__ import annotations
from random import random

import processData
from globals import *

from typing import *

import numpy as np
from sklearn import linear_model, model_selection, metrics


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


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


def main():
    psi_bar, y = processData.process_data()

    y_pis = y[:, 0]
    y_hva = y[:, 1]

    s_list = range(10, 101, 10)

    coeff_models = [
        linear_model.LinearRegression(),
        # linear_model.Lasso(),
        linear_model.Ridge(),
        # linear_model.ElasticNet()
    ]

    indices_list = []

    for model in coeff_models:
        indices, coeffs = coeff_imp(psi_bar, y_pis, model)

        max_r2: float = None
        max_r2_s: int = 0

        for s in s_list:
            r2, _, _ = get_metrics(psi_bar[:, indices[:s]], y_pis, model)

            if max_r2 is None or r2 > max_r2:
                max_r2 = r2
                max_r2_s = s
        #print(f'For {model}, the optimal value of s={max_r2_s}, yielding test R^2 = {max_r2}')
        indices_list.append(indices)

    s = 100
    linear_indices = set(indices_list[0][:s])
    ridge_indices = set(indices_list[1][:s])

    u = linear_indices.union(ridge_indices)
    print(len(u))


if __name__ == '__main__':
    main()
