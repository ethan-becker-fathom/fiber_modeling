import math
import scipy
import numpy as np
import matplotlib.pyplot as plt


def dopant_concentration(r, t):
    D = 4.8276e-15
    # t = 10

    b = 9e-6
    B = 2 * np.sqrt(D * t)

    C = (np.square(b) / np.square(B)) * np.exp(-np.square(r) / np.square(B))

    return C


if __name__ == '__main__':
    print('starting')
    r = np.linspace(0, 10e-6, 100)
    print(r)
    C_10 = dopant_concentration(r, 10)
    C_30 = dopant_concentration(r, 30)
    C_60 = dopant_concentration(r, 60)

    plt.plot(r, C_10)
    plt.plot(r, C_30)
    plt.plot(r, C_60)
    plt.show()
