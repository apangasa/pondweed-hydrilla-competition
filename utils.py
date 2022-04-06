from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np


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


@dataclass
class TransformFunctions:
    functions: set[Callable]

    def __post_init__(self):
        self.functions.add(lambda x: x)

    def transform(self, feature: float) -> list[float]:
        return [func(feature) for func in self.functions]
