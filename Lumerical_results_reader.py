import re
import numpy
import matplotlib.pyplot as plt
import json
import csv
import os
from collections import defaultdict


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

    print(loss_data)
    print(list(list(loss_data.values())[0].keys()))
    with open('bending_loss.csv', 'w', newline='') as csvfile:
        fieldnames = [int(i) for i in list(loss_data.values())[0].keys()].sort()

        fieldnames = ['name'] + fieldnames
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=",")

        writer.writeheader()
        for key, val in loss_data.items():
            d = val
            d['name'] = key
            writer.writerow(d)
            print(d)


if __name__ == '__main__':
    dir = 'Lumerical_Results'

    read_json_files(dir)