from __future__ import annotations
from dataclasses import dataclass
from typing import *

import numpy as np
import random

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures


def average(x: list[float]) -> float:
    return sum(x) / len(x) if len(x) else 0.0


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
        new_features = []

        new_features.extend(transform_functions.transform(self.pis_initial))
        new_features.extend(transform_functions.transform(self.hva_initial))
        new_features.extend(transform_functions.transform(self.nutrient_ratio))
        new_features.extend(transform_functions.transform(self.shade))

        new_features = PolynomialFeatures(
            degree=transform_functions.poly_degree).fit_transform(new_features)

        return np.array(new_features)


@dataclass
class DataContainer:
    data_list: list[RawData]
    phi: np.ndarray = None
    y: np.ndarray = None
    scaler: MinMaxScaler = None

    def transform_features(self, transform_functions: TransformFunctions) -> np.ndarray:
        self.phi = np.array([point.transform_features(
            transform_functions) for point in self.data_list])
        self.y = np.array([[point.pis_final, point.hva_final]
                           for point in self.data_list])

    def get_features_and_classes(self, transform_functions: TransformFunctions, normalize: bool = False, shuffle: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if shuffle:
            random.shuffle(self.data_list)
        self.transform_features(transform_functions)
        self.scaler = MinMaxScaler()
        self.phi = self.scaler.fit_transform(self.phi)

        return self.phi, self.y


@dataclass
class TransformFunctions:
    poly_degree: int
    functions: list[Callable]

    def __post_init__(self):
        self.functions.append(lambda x: x)

    def transform(self, feature: float) -> list[float]:
        return [func(feature) for func in self.functions]
