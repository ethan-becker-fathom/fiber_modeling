import numpy as np
import matplotlib.pyplot as plt
# import json
# import scipy.io
from scipy import interpolate, ndimage, datasets, optimize
# import os
import pandas as pd
# import collections
from pathlib import Path
# import tkinter as tk
# from tkinter import filedialog
import math
import random
import logging
import time
import pickle
# import cv2
import functools

from numba import jit


# from line_profiler_pycharm import profile


@jit(cache=True, nopython=True)
def polar_to_rect(r, phi):
    return r * (np.cos(phi) + np.sin(phi) * 1j)


def load_mode_data(file_path='TAF Fiber Modes only Farfield.csv'):
    df = pd.read_csv(file_path)

    df.drop('farfield1_exact_Ey(real)', axis=1, inplace=True)
    df.drop('farfield1_exact_Ey(imag)', axis=1, inplace=True)
    df.drop('farfield1_exact_Ez(imag)', axis=1, inplace=True)
    df.drop('farfield1_exact_Ez(real)', axis=1, inplace=True)
    df.drop('farfield1_exact_Hx(imag)', axis=1, inplace=True)
    df.drop('farfield1_exact_Hx(real)', axis=1, inplace=True)
    df.drop('farfield1_exact_Hy(imag)', axis=1, inplace=True)
    df.drop('farfield1_exact_Hy(real)', axis=1, inplace=True)
    df.drop('farfield1_exact_Hz(imag)', axis=1, inplace=True)
    df.drop('farfield1_exact_Hz(real)', axis=1, inplace=True)

    df.drop('farfield3_exact_Ey(real)', axis=1, inplace=True)
    df.drop('farfield3_exact_Ey(imag)', axis=1, inplace=True)
    df.drop('farfield3_exact_Ez(imag)', axis=1, inplace=True)
    df.drop('farfield3_exact_Ez(real)', axis=1, inplace=True)
    df.drop('farfield3_exact_Hx(imag)', axis=1, inplace=True)
    df.drop('farfield3_exact_Hx(real)', axis=1, inplace=True)
    df.drop('farfield3_exact_Hy(imag)', axis=1, inplace=True)
    df.drop('farfield3_exact_Hy(real)', axis=1, inplace=True)
    df.drop('farfield3_exact_Hz(imag)', axis=1, inplace=True)
    df.drop('farfield3_exact_Hz(real)', axis=1, inplace=True)

    df.drop('farfield4_exact_Ey(real)', axis=1, inplace=True)
    df.drop('farfield4_exact_Ey(imag)', axis=1, inplace=True)
    df.drop('farfield4_exact_Ez(imag)', axis=1, inplace=True)
    df.drop('farfield4_exact_Ez(real)', axis=1, inplace=True)
    df.drop('farfield4_exact_Hx(imag)', axis=1, inplace=True)
    df.drop('farfield4_exact_Hx(real)', axis=1, inplace=True)
    df.drop('farfield4_exact_Hy(imag)', axis=1, inplace=True)
    df.drop('farfield4_exact_Hy(real)', axis=1, inplace=True)
    df.drop('farfield4_exact_Hz(imag)', axis=1, inplace=True)
    df.drop('farfield4_exact_Hz(real)', axis=1, inplace=True)

    df['farfield1_exact_Ex_power'] = np.abs(df['farfield1_exact_Ex(real)'] + 1j * df['farfield1_exact_Ex(imag)'])
    df['farfield3_exact_Ex_power'] = np.abs(df['farfield3_exact_Ex(real)'] + 1j * df['farfield3_exact_Ex(imag)'])
    df['farfield4_exact_Ex_power'] = np.abs(df['farfield4_exact_Ex(real)'] + 1j * df['farfield4_exact_Ex(imag)'])

    farfield1_exact_Ex_power_pivot = df.pivot(index='far_y', columns='far_x', values='farfield1_exact_Ex_power')
    farfield3_exact_Ex_power_pivot = df.pivot(index='far_y', columns='far_x', values='farfield3_exact_Ex_power')
    farfield4_exact_Ex_power_pivot = df.pivot(index='far_y', columns='far_x', values='farfield4_exact_Ex_power')

    return farfield1_exact_Ex_power_pivot.values, farfield3_exact_Ex_power_pivot.values, farfield4_exact_Ex_power_pivot.values


@jit(cache=True, nopython=True)
def pad_and_combine(array_1, array_2):
    # array_1[:array_2.shape[0], :array_2.shape[1]] += array_2
    for i in range(array_2.shape[0]):
        for j in range(array_2.shape[1]):
            array_1[i, j] += array_2[i, j]
    return array_1


@jit(cache=True)
# @profile
def create_mode_image(mode_1,
                      mode_2,
                      mode_3,
                      mode_1_power: float,
                      mode_2_power: float,
                      mode_2_phase: float,
                      mode_3_power: float,
                      mode_3_phase: float,
                      rotate: float,
                      scale: float,
                      shift_x: float,
                      shift_y: float,
                      final_dimensions=(4112, 3008)
                      ):
    mode_1_complex = mode_1 * mode_1_power

    mode_2_modulus = mode_2 * mode_2_power
    mode_2_angle = np.full_like(mode_2_modulus, mode_2_phase)
    mode_2_complex = polar_to_rect(mode_2_modulus, mode_2_angle)

    mode_3_modulus = mode_3 * mode_3_power
    mode_3_angle = np.full_like(mode_3_modulus, mode_3_phase)
    mode_3_complex = polar_to_rect(mode_3_modulus, mode_3_angle)

    total_complex = mode_1_complex + mode_2_complex + mode_3_complex
    total_intensity = np.square(np.abs(total_complex))

    total_rotate = ndimage.rotate(total_intensity, rotate, reshape=False, order=0)
    total_rotate_zoom = ndimage.zoom(total_rotate, scale, order=0)
    full_image = np.zeros(final_dimensions)
    # pad_x = full_image.shape[0] - total_rotate_zoom.shape[0]
    # pad_y = full_image.shape[1] - total_rotate_zoom.shape[1]
    # full_image = full_image + np.pad(total_rotate_zoom, ((0, pad_x), (0, pad_y)), 'constant', constant_values=(0, 0))
    full_image = pad_and_combine(full_image, total_rotate_zoom)
    # full_image_shift = ndimage.shift(full_image, (shift_x, shift_y), cval=0, order=0)
    shift_x = int(round(shift_x))
    shift_y = int(round(shift_y))

    full_image_shift = np.roll(full_image, (shift_x, shift_y), axis=(1, 0))
    return full_image_shift


@jit(cache=True, nopython=True)
def image_difference(image_1, image_2):
    diff = image_1 - image_2
    diff_sq = np.square(diff)
    sum = np.sum(diff_sq)
    return sum


@jit(cache=True)
def create_and_diff(
        args,
        # mode_1_power: float,
        # mode_2_power: float,
        # mode_2_phase: float,
        # mode_3_power: float,
        # mode_3_phase: float,
        # rotate: float,
        scale: float,
        shift_x: float,
        shift_y: float,
        mode_1,
        mode_2,
        mode_3,
        compare_image,
        final_dimensions=(4112, 3008),
):
    mode_1_power, mode_2_power, mode_2_phase, mode_3_power, mode_3_phase, rotate = args
    # print(args)
    # print(mode_1_power, mode_2_power, mode_2_phase, mode_3_power, mode_3_phase, rotate, scale, shift_x, shift_y)

    image = create_mode_image(mode_1=mode_1,
                              mode_2=mode_2,
                              mode_3=mode_3,
                              mode_1_power=mode_1_power,
                              mode_2_power=mode_2_power,
                              mode_2_phase=mode_2_phase,
                              mode_3_power=mode_3_power,
                              mode_3_phase=mode_3_phase,
                              rotate=rotate,
                              scale=scale,
                              shift_x=shift_x,
                              shift_y=shift_y,
                              final_dimensions=final_dimensions
                              )

    d = image_difference(image, compare_image)
    return d


# @profile
def main():
    Path('log_files/').mkdir(exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s',
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler(),
            # logging.FileHandler(f'log_files/{Path(__file__).stem}-{time.strftime("%Y-%m-%d--%H-%M-%S")}.log'),
        ]
    )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    z = np.zeros((100, 100))
    # pad_and_combine(z, z)
    # polar_to_rect(z, z)

    try:
        with open('data.pickle', 'rb') as f:
            data = pickle.load(f)
            mode_1, mode_2, mode_3 = data
            print('pickle loaded')
    except:
        mode_1, mode_2, mode_3 = load_mode_data()
        with open('data.pickle', 'wb') as f:
            data = mode_1, mode_2, mode_3
            pickle.dump(data, f)
            print('pickle saved')

    param_bounds = {
        'mode_1_power': (0, 1),
        'mode_2_power': (0, 1),
        'mode_2_phase': (0, 2 * math.pi),
        'mode_3_power': (0, 1),
        'mode_3_phase': (0, 2 * math.pi),
        'rotate': (0, 90),
        # 'scale': (0, 1),
        # 'shift_x': (0, 1000),
        # 'shift_y': (0, 1000)
    }


    results = []

    for i in range(10):

        random_dict = {}
        for param, bound in param_bounds.items():
            random_dict[param] = random.uniform(bound[0], bound[1])

        image = create_mode_image(
            mode_1=mode_1,
            mode_2=mode_2,
            mode_3=mode_3,
            mode_1_power=random_dict['mode_1_power'],
            mode_2_power=random_dict['mode_2_power'],
            mode_2_phase=random_dict['mode_2_phase'],
            mode_3_power=random_dict['mode_3_power'],
            mode_3_phase=random_dict['mode_3_phase'],
            rotate=random_dict['rotate'],
            scale=1,
            shift_x=0,
            shift_y=0,
            final_dimensions=(500, 500)
        )
        print(random_dict)


        partial_create_and_diff = functools.partial(
            create_and_diff,
            scale=1,
            shift_x=0,
            shift_y=0,
            mode_1=mode_1,
            mode_2=mode_2,
            mode_3=mode_3,
            compare_image=image,
            final_dimensions=(500, 500)
        )

        t = time.time()
        res_1 = optimize.brute(
            partial_create_and_diff,
            # x0=np.array([random.random(), random.random(), random.random(), random.random(), random.random()]),
            # bounds=tuple(param_bounds.values()),
            ranges = tuple(param_bounds.values()),
            # locally_biased=False,
            # maxfun=100000,
            # vol_tol=1e-30,
            # f_min=0,
            # f_min_rtol=1e-10
        )
        t2 = time.time()
        print(f'Total time: {t2 - t}')
        print(res_1)
        print(random_dict)

        res_2 = optimize.minimize(
            partial_create_and_diff,
            x0=res_1.x,
            bounds=tuple(param_bounds.values()),
            # method='Nelder-Mead',
            method='Powell',
            tol=1e-10,
            options={'maxiter': 10000}
        )
        t3 = time.time()
        print(f'Total time: {t3 - t2}')
        print(res_2)
        print(random_dict)

        results.append([random_dict, res_1, res_2, t, t2, t3])


    with open('results_Brute_&_Powell.pickle', 'wb') as f:
        pickle.dump(results, f)
        print('pickle saved')

    image_2 = create_mode_image(mode_1=mode_1,
                                mode_2=mode_2,
                                mode_3=mode_3,
                                mode_1_power=res_2.x[0],
                                mode_2_power=res_2.x[1],
                                mode_2_phase=res_2.x[2],
                                mode_3_power=res_2.x[3],
                                mode_3_phase=res_2.x[4],
                                rotate=res_2.x[5],
                                scale=1,
                                shift_x=0,
                                shift_y=0,
                                final_dimensions=(500, 500))

    d = image_difference(image, image_2)
    print(d, np.max(image - image_2))

    f, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(image)
    ax[0, 1].imshow(image_2)
    ax[1, 0].imshow(image - image_2)
    ax[1, 1].imshow(z)

    # plt.show()


if __name__ == '__main__':
    main()
