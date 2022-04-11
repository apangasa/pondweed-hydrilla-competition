from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import *

import numpy as np
import random

from globals import *

from sklearn import preprocessing


def average(x: list[float]) -> float:
    return sum(x) / len(x) if len(x) else 0.0


class Category(Enum):
    INCREASED = 0
    DECREASED = 1
    EXTINCT = 2


@dataclass
class InitialMassHandler:
    avg_unit_pis: float
    avg_unit_hva: float

    def get_initial_masses(self, read_ratio: str) -> tuple[float, float]:
        hva_units, pis_units = tuple([int(term)
                                      for term in read_ratio.split(':')])
        return self.avg_unit_pis * pis_units, self.avg_unit_hva * hva_units


@dataclass
class RawData:
    pis_initial: float
    hva_initial: float
    nutrient_ratio: float
    shade: int

    pis_final: float
    hva_final: float

    def transform_features(self, transform_functions: TransformFunctions) -> np.array:
        new_features: list = []

        new_features.extend(transform_functions.transform(self.pis_initial))
        new_features.extend(transform_functions.transform(self.hva_initial))
        new_features.extend(transform_functions.transform(self.nutrient_ratio))
        new_features.extend(transform_functions.transform(self.shade))

        np_new_features: np.ndarray = preprocessing.PolynomialFeatures(
            degree=transform_functions.poly_degree).fit_transform(np.array([new_features]))

        np_new_features = np_new_features.flatten()

        return np_new_features

    def categorize(self):
        self.pis_category: Category = None
        self.hva_category: Category = None

        if self.pis_final >= self.pis_initial:
            self.pis_category = Category.INCREASED
        elif self.pis_final == 0:
            self.pis_category = Category.EXTINCT
        else:
            self.pis_category = Category.DECREASED

        if self.hva_final >= self.hva_initial:
            self.hva_category = Category.INCREASED
        elif self.hva_final == 0:
            self.hva_category = Category.EXTINCT
        else:
            self.hva_category = Category.DECREASED


@dataclass
class DataContainer:
    data_list: list[RawData]
    psi_bar: np.ndarray = None
    y: np.ndarray = None
    scaler: preprocessing.MinMaxScaler = None

    def transform_features(self, transform_functions: TransformFunctions) -> np.ndarray:
        self.psi_bar = np.array([point.transform_features(
            transform_functions) for point in self.data_list])
        self.y = np.array([[point.pis_final, point.hva_final]
                           for point in self.data_list])

    def categorize_y(self):
        for point in self.data_list:
            point.categorize()
        self.y = np.array([[point.pis_category.value, point.hva_category.value]
                           for point in self.data_list])

    def get_features_and_classes(self, transform_functions: TransformFunctions, normalize: bool = False, shuffle: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if shuffle:
            random.Random(SEED).shuffle(self.data_list)
        self.transform_features(transform_functions)
        if normalize:
            self.scaler = preprocessing.MinMaxScaler()
            self.psi_bar = self.scaler.fit_transform(self.psi_bar)

        return self.psi_bar, self.y


@dataclass
class TransformFunctions:
    poly_degree: int
    functions: list[Callable]

    def __post_init__(self):
        self.functions.append(lambda x: x)

    def transform(self, feature: float) -> list[float]:
        return [func(feature) for func in self.functions]
