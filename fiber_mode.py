from importlib.resources import path
from statistics import mode
from matplotlib.pyplot import sci
import scipy
import scipy.io
import scipy.signal
import scipy.integrate
import scipy.optimize
import tkinter.filedialog
import matplotlib.pyplot as plt
import numpy as np


def gaussuian_2D(size, sigma=1, mu_x=0, mu_y=0, max_val = 1, r=1):
    x, y = np.meshgrid(np.linspace(-r, r, size),
                       np.linspace(-r, r, size))

    gauss = max_val * np.exp( - np.square(x - mu_x) / (2 * sigma**2) - np.square(y - mu_y) / (2 * sigma**2))
    
    return gauss


def complex_field_overlap_1D(xy_field_1, xy_field_2):
    mult_field = np.conjugate(xy_field_1) * xy_field_2

    multi_field_res = scipy.integrate.simps(mult_field)

    multi_field_res_abs_sq = np.abs(multi_field_res) ** 2

    incident_field_1_5_abs_sq = np.square(np.abs(xy_field_1))
    incident_field_1_5_res = scipy.integrate.simps(incident_field_1_5_abs_sq)

    incident_field_2_0_abs_sq = np.square(np.abs(xy_field_2))
    incident_field_2_0_res = scipy.integrate.simps(incident_field_2_0_abs_sq)

    nu = multi_field_res_abs_sq / (incident_field_1_5_res * incident_field_2_0_res)

    return nu

def complex_field_overlap_2D(xy_field_1, xy_field_2):
    mult_field = np.conjugate(xy_field_1) * xy_field_2

    multi_field_res = scipy.integrate.simps(scipy.integrate.simps(mult_field))

    multi_field_res_abs_sq = np.abs(multi_field_res) ** 2

    incident_field_1_5_abs_sq = np.square(np.abs(xy_field_1))
    incident_field_1_5_res = scipy.integrate.simps(scipy.integrate.simps(incident_field_1_5_abs_sq))

    incident_field_2_0_abs_sq = np.square(np.abs(xy_field_2))
    incident_field_2_0_res = scipy.integrate.simps(scipy.integrate.simps(incident_field_2_0_abs_sq))

    nu = multi_field_res_abs_sq / (incident_field_1_5_res * incident_field_2_0_res)

    return nu

def gaussian_fit_func(vals, mode, array_size):
    amp = vals[0]
    sigma = vals[1]
    mu_x = vals[2]
    mu_y = vals[3]
    
    g = gaussuian_2D(np.size(x), sigma = .1, mu_x = 0, mu_y=0)
    
    return 1 - complex_field_overlap_2D 

if __name__ == '__main__':
    
    # mode_data_path =tkinter.filedialog.askopenfilename()
    mode_data_path='C:/Users/ethan/Fathom Radiant Dropbox/Engineering/Fiber/Simulation/FaSt/mode_SIF_9-22-40_lambda1um.mat'
    print(mode_data_path)
    mode_data = scipy.io.loadmat(mode_data_path)
    
    x = mode_data['x'][0]
    i = mode_data['I']
    
    i_0 = i[:,:,0]
    i_0_mfd = mode_data['MFD'][0][0]
    i_0_profile = i_0[int(np.round(np.size(x)/2)),:]
    
    spacing = np.abs(x[0]-x[1])
    
    
    # i_0_gaussian = scipy.signal.gaussian(np.size(x), i_0_mfd / spacing / 4)
    # i_0_gaussian = i_0_gaussian * np.max(i_0_profile)
    # print(complex_field_overlap_1D(i_0_profile, i_0_gaussian))
    # plt.contour(i_0)
    # plt.plot(i_0_profile)
    # plt.plot(i_0_gaussian)
    # plt.show()

    res = scipy.optimize.minimize(gaussian_fit_func, [1, .1, 0 , 0], args=(i_0, int(np.round(np.size(x)/2))))
    
    g = gaussuian_2D(np.size(x), sigma = .1, mu_x = 0, mu_y=0)
    plt.imshow(g)
    plt.show()