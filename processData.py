from __future__ import annotations
from typing import *

import csv
import math
import random

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

            if bucket_num not in data_dict.keys():
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


def format_data(data_dict: dict[int, DataMap]) -> list[RawData]:
    data: list[RawData] = []

    for bucket_num in data_dict.keys():
        pis_initial_list = data_dict[bucket_num]['pis_initial']
        hva_initial_list = data_dict[bucket_num]['hva_initial']
        nutrient_ratio = data_dict[bucket_num]['nutrient_ratio']
        shade = data_dict[bucket_num]['shade']
        pis_final_list = data_dict[bucket_num]['pis_final']
        hva_final_list = data_dict[bucket_num]['hva_final']

        pis_initial = average(pis_initial_list)
        hva_initial = average(hva_initial_list)
        pis_final = average(pis_final_list)
        hva_final = average(hva_final_list)

        data.append(RawData(pis_initial=pis_initial, hva_initial=hva_initial,
                            nutrient_ratio=nutrient_ratio, shade=shade, pis_final=pis_final, hva_final=hva_final))
    return data


def main():
    initial_mass_handler = InitialMassHandler(0.11, 0.13)
    transform_functions = TransformFunctions({
        lambda x: math.log(x) if x != 0 else math.log(EPSILON),
        lambda x: 1 / x if x != 0 else 1 / EPSILON,
        lambda x: x ** 2,
        lambda x: x ** 3,
        lambda x: math.sqrt(x)
    })

    data_dict: dict[int, DataMap] = read_data(
        './data/shade_run1.csv', initial_mass_handler)
    data: list[RawData] = format_data(data_dict)

    random.shuffle(data)

    phi = np.array([point.transform_features(transform_functions)
                    for point in data])
    y = np.array([[point.pis_final, point.hva_final] for point in data])


if __name__ == '__main__':
    main()
