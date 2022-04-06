from __future__ import annotations
from typing import *

import csv

from utils import *

# Globals
EPSILON = 1e-8

# Type Aliases
DataMap = NewType('DataMap', Dict[str, List[float]])


def read_data(fname: str, initial_mass_handler: InitialMassHandler) -> dict[int, DataMap]:
    data_dict: dict[int, DataMap] = {}

    with open(fname, 'r') as data_file:
        reader = csv.reader(data_file)
        next(reader)
        for row in reader:
            bucket_num: int = int(row[0])

            data_dict[bucket_num] = {
                'pis_initial': [],
                'hva_initial': [],
                'nutrient_ratio': 0,
                'shade': None,
                'pis_final': [],
                'hva_final': []
            }

            pis_initial, hva_initial = initial_mass_handler.get_initial_masses(
                row[1])

            data_dict[bucket_num]['pis_initial'].append(pis_initial)
            data_dict[bucket_num]['hva_initial'].append(hva_initial)
            data_dict[bucket_num]['nutrient_ratio'] = 0  # TODO
            data_dict[bucket_num]['shade'] = int(row[2]) == 1

            if 'PIS' in row[3]:
                data_dict[bucket_num]['pis_final'].append(float(row[4]))
            elif 'HVA' in row[3]:
                data_dict[bucket_num]['hva_final'].append(float(row[4]))

    return data_dict
