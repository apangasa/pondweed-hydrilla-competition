from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np


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
    shade: bool

    pis_final: float
    hva_final: float

    def transform_features(self, transform_functions: TransformFunctions) -> np.array:
        new_features = []

        new_features.extend(transform_functions.transform(self.pis_initial))
        new_features.extend(transform_functions.transform(self.hva_initial))
        new_features.extend(transform_functions.transform(self.nutrient_ratio))

        new_features.append(int(self.shade))

        return np.array(new_features)


@dataclass
class TransformFunctions:
    functions: list[Callable]

    def __post_init__(self):
        self.functions.append(lambda x: x)

    def transform(self, feature: float) -> list[float]:
        return [func(feature) for func in self.functions]
