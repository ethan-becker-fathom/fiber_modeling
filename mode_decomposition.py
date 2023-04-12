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
import cv2

from numba import jit
from line_profiler_pycharm import profile


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


# @jit(cache=True)
@profile
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
    full_image_shift = np.roll(full_image, (shift_x, shift_y), axis=(1, 0))
    return full_image_shift


@jit(cache=True, nopython=True)
def image_difference(image_1, image_2):
    diff = image_1 - image_2
    diff_sq = np.square(diff)
    sum = np.sum(diff_sq)
    return sum


def create_and_diff(args,
                    # mode_1_power: float,
                    # mode_2_power: float,
                    # mode_2_phase: float,
                    # mode_3_power: float,
                    # mode_3_phase: float,
                    rotate: float,
                    scale: float,
                    shift_x: float,
                    shift_y: float,
                    mode_1,
                    mode_2,
                    mode_3,
                    compare_image,
                    final_dimensions=(4112, 3008),
                    ):
    mode_1_power, mode_2_power, mode_2_phase, mode_3_power, mode_3_phase = args
    print(args)

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
                              shift_y=shift_y
                              )

    d = image_difference(image, compare_image)
    return d


@profile
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
    pad_and_combine(z, z)
    polar_to_rect(z, z)

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

    # for i in range(10):
    #     break
    #     print(i)
    #     image = create_mode_image(mode_1=mode_1,
    #                               mode_2=mode_2,
    #                               mode_3=mode_3,
    #                               mode_1_power=random.uniform(0, 1),
    #                               mode_2_power=random.uniform(0, 1),
    #                               mode_2_phase=random.uniform(0, 2 * math.pi),
    #                               mode_3_power=random.uniform(0, 1),
    #                               mode_3_phase=random.uniform(0, 2 * math.pi),
    #                               rotate=random.uniform(0, 2 * math.pi),
    #                               scale=random.uniform(0, 5),
    #                               shift_x=random.randint(0, 500),
    #                               shift_y=random.randint(0, 500))

    random_vector = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 2 * math.pi), random.uniform(0, 1),
                     random.uniform(0, 2 * math.pi)]

    image = create_mode_image(mode_1=mode_1,
                              mode_2=mode_2,
                              mode_3=mode_3,
                              mode_1_power=random_vector[0],
                              mode_2_power=random_vector[1],
                              mode_2_phase=random_vector[2],
                              mode_3_power=random_vector[3],
                              mode_3_phase=random_vector[4],
                              rotate=45,
                              scale=3,
                              shift_x=500,
                              shift_y=500,
                              final_dimensions=(4112, 3008))

    # image_2 = create_mode_image(mode_1=mode_1,
    #                             mode_2=mode_2,
    #                             mode_3=mode_3,
    #                             mode_1_power=random.uniform(0, 1),
    #                             mode_2_power=random.uniform(0, 1),
    #                             mode_2_phase=random.uniform(0, 2 * math.pi),
    #                             mode_3_power=random.uniform(0, 1),
    #                             mode_3_phase=random.uniform(0, 2 * math.pi),
    #                             rotate=random.uniform(0, 2 * math.pi),
    #                             scale=random.uniform(0, 2),
    #                             shift_x=random.randint(0, 500),
    #                             shift_y=random.randint(0, 500),
    #                             final_dimensions=(1000, 1000))

    t = time.time()
    res = optimize.direct(
        create_and_diff,
        # x0=np.array([random.random(), random.random(), random.random(), random.random(), random.random()]),
        bounds=((0, 1), (0, 1), (0, 2 * math.pi), (0, 1), (0, 2 * math.pi)),
        args=(
            # 0,
            # 3,
            # 0,
            # math.pi,
            45,
            3,
            500,
            500,
            mode_1,
            mode_2,
            mode_3,
            image
        ),
        # method='Nelder-Mead'
    )
    print(f'Total time: {time.time() - t}')
    print(res)
    print(random_vector)

    image_2 = create_mode_image(mode_1=mode_1,
                                mode_2=mode_2,
                                mode_3=mode_3,
                                mode_1_power=res.x[0],
                                mode_2_power=res.x[1],
                                mode_2_phase=res.x[2],
                                mode_3_power=res.x[3],
                                mode_3_phase=res.x[4],
                                rotate=45,
                                scale=3,
                                shift_x=500,
                                shift_y=500,
                                final_dimensions=(4112, 3008))

    # print(res.x)

    d = image_difference(image, image_2)
    print(d)

    f, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(image)
    ax[0, 1].imshow(image_2)
    ax[1, 0].imshow(image - image_2)
    ax[1, 1].imshow(z)

    plt.show()


if __name__ == '__main__':
    main()
