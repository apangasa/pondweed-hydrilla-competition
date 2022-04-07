from __future__ import annotations
from typing import *

import csv
import math

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

            nitrogen, phosphorous = row[2].split(':')

            data_dict[bucket_num]['nutrient_ratio'] = float(
                nitrogen) / float(phosphorous)
            data_dict[bucket_num]['shade'] = int(row[3])

            if 'PIS' in row[4]:
                data_dict[bucket_num]['pis_final'].append(float(row[5]))
            elif 'HVA' in row[4]:
                data_dict[bucket_num]['hva_final'].append(float(row[5]))

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
    transform_functions = TransformFunctions(poly_degree=3, functions=[
        lambda x: math.log(x) if x != 0 else math.log(
            EPSILON),
        lambda x: 1 / x if x != 0 else 1 / EPSILON,
        lambda x: math.sqrt(x)])

    data: list[RawData] = []

    initial_mass_handler = InitialMassHandler(
        avg_unit_pis=0.105, avg_unit_hva=0.125)
    data_dict: dict[int, DataMap] = read_data(
        './data/shade_run1.csv', initial_mass_handler)
    data.extend(format_data(data_dict))

    initial_mass_handler = InitialMassHandler(
        avg_unit_pis=0.178, avg_unit_hva=0.098)
    data_dict: dict[int, DataMap] = read_data(
        './data/shade_run2.csv', initial_mass_handler)
    data.extend(format_data(data_dict))

    initial_mass_handler = InitialMassHandler(
        avg_unit_pis=0.1125, avg_unit_hva=0.0975)
    data_dict: dict[int, DataMap] = read_data(
        './data/np_run.csv', initial_mass_handler)
    data.extend(format_data(data_dict))

    data_container = DataContainer(data_list=data)
    phi, y = data_container.get_features_and_classes(
        transform_functions=transform_functions, normalize=True, shuffle=True)

    print(phi.shape)
    print(y.shape)


if __name__ == '__main__':
    main()
