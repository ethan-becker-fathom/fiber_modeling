import re
import numpy
import matplotlib.pyplot as plt
import json
import csv
import os
from collections import defaultdict
import pathlib
import re


def read_json_files(directory):
    loss_data = defaultdict(dict)
    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(file):
            print(file)
            with open(file) as f:
                data = json.load(f)


            res = re.search(r'wavelength-(\d+)nm_bend-([\d\.]+)mm.json', filename)
            wavelength_nm = res[1]
            bend = res [2]
            print(wavelength_nm, bend)
            print(data)
            for key, val in data.items():
                print(key, val)
                loss_data[f'{wavelength_nm}nm-mode{key}'][bend] = val['loss']




if __name__ == '__main__':
    dir = 'Lumerical_Results'

    # read_json_files(dir)
    # filename = pathlib.Path('Lumerical_Results/TAF_8.3-20-30_dispersion-870nm_bend-10.0mm.json')
    # print(filename)

    # for filename in os.listdir(dir):
    #     filename = os.path.join(dir, filename)
    #     print(filename)
    #     with open(filename, 'r') as file:
    #         data = json.load(file)
    #     print(data)
    #     data['wavelength'] = data['1']['wavelength']
    #     data['bend_radius'] = data['1']['bend_radius']
    #     for i in range(1,7):
    #         del data[str(i)]['wavelength']
    #         del data[str(i)]['bend_radius']
    #     print(data)
    #
    #     with open(filename, 'w') as file:
    #         json.dump(data, file)

    for filename in os.listdir(dir):
        filename = os.path.join(dir, filename)
        if not filename.endswith('.json'):
            continue
        if filename.endswith('full-dataset.json') or filename.endswith('mode-overlaps.json'):
            continue
        base = re.match(r'(.*).json', filename)
        print(filename, f'{base[1]}_summary.json')
        os.rename(filename, f'{base[1]}_summary.json')
