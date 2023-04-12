import numpy as np
import matplotlib.pyplot as plt
import math


def nr(n0, g, r):
    return n0 * (1 - (np.square(g) * np.square(r)) / 2)


def NA_to_core_n(NA, n_cladding):
    return np.sqrt(np.square(NA) + np.square(n_cladding))


def gradient_constant(n_cladding, n_core, radius):
    return (np.sqrt(2) * np.sqrt(n_core - n_cladding)) / (np.sqrt(n_core) * radius)


def length_to_pitch(gradient_constant, length):
    return (np.sqrt(gradient_constant) * length) / (2 * np.pi)


def pitch_to_length(gradient_constant, pitch):
    return (2 * np.pi * pitch) / (gradient_constant)


def ABCD(ls, n0, n1, g, lg, zw):
    A = np.cos(g * lg) - n1 * zw * g * np.sin(g * lg)
    B = (ls + n0 * zw) * np.cos(g * lg) + (n0 / (n1 * g) - n1 * g * ls * zw) * np.sin(
        g * lg
    )
    C = -n1 * g * np.sin(g * lg)
    D = n0 * np.cos(g * lg) - n1 * g * ls * np.sin(g * lg)

    return (A, B, C, D)


def w01(A, B, C, D, w0, n0, wavelength):
    alhpa = wavelength / (np.pi * np.square(w0) * n0)

    w01 = w0 * np.sqrt(
        n0 * ((np.square(A) + np.square(alhpa) * np.square(B)) / (A * D - B * C))
    )

    return w01

if __name__ == "__main__":

    n0 = 1.447
    na = 0.2
    n_core = NA_to_core_n(na, n0)
    print(n_core)
    core_diameter = 100
    gc = gradient_constant(n0, n_core, core_diameter / 2)
    print(gc)

    # fiber_diameter = 0.125

    quarter_pitch_length = pitch_to_length(gc, 0.25)
    print(quarter_pitch_length)

    # x = np.linspace(-fiber_diameter / 2, fiber_diameter / 2, 1000)
    # n_r_core = nr(n_core, gc, x)
    # n_r_cladding = np.full_like(x, n0)

    # n_r = np.maximum(n_r_core, n_r_cladding)

    # plt.plot(x, n_r)
    # plt.show()

    w0_850 = 9.1/2
    wavelength_850 = 0.85
    w0_1000 = 10.3/2
    wavelength_1000 = 1
    lg = np.linspace(0, 1000, 1000)
    

    A, B, C, D = ABCD(0, n0, n_core, gc, lg, 0)
    spot_size_850 = w01(A, B, C, D, w0_850, n0,wavelength_850)
    spot_size_1000 = w01(A, B, C, D, w0_1000, n0,wavelength_1000)
    
    plt.plot(lg, spot_size_850, label='850nm')
    plt.plot(lg, spot_size_1000, label='1000nm')
    
    plt.xlabel('Distance (um)')
    plt.ylabel('Beam Radius (um)')
    plt.legend()
    
    plt.show()
    
    # Fucntion to use tkinter to ask to open a file