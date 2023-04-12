from distutils import core
import math
import wave
import numpy as np
import matplotlib.pyplot as plt
import scipy


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


def V_number(NA, core_diameter, wavelength=850e-9):
    return (2 * np.pi * (core_diameter / 2) * NA) / wavelength


def fiber_MFD(NA, core_diameter, wavelength=850e-9):
    normalized_frequency = V_number(NA, core_diameter, wavelength)
    mode_field_diameter = core_diameter * \
                          (0.65 + 1.619 / np.power(normalized_frequency, 3 / 2)
                           + 2.879 / np.power(normalized_frequency, 6)
                           - 0.016
                           - 1.561 / np.power(normalized_frequency, 7))

    return mode_field_diameter


def fiber_NA(core_n, cladding_n):
    return np.sqrt(np.square(core_n) - np.square(cladding_n))


def power_to_db(power: float, reference_power: float = 1) -> float:
    return 10 * np.log10(power / reference_power)


def db_to_power(db: float, reference_power: float = 1) -> float:
    return 10 ** ((db) / 10) * reference_power


def coupling_loss_lateral_offset_fiber(mfd_1_um, mfd_2_um, offset_um):
    mfr_1_um = mfd_1_um / 2
    mfr_2_um = mfd_2_um / 2

    return (((2 * mfr_1_um * mfr_2_um) / (mfr_1_um ** 2 + mfr_2_um ** 2)) ** 2) * np.exp(
        (-2 * offset_um ** 2) / (mfr_1_um ** 2 + mfr_2_um ** 2))


if __name__ == '__main__':
    # loss = coupling_loss_lateral_offset_fiber(9, 9, 3)
    # print(loss)

    # mfd = 13.19*2
    #
    # d = np.linspace(0, 20, 101)
    # c = np.zeros_like(d)
    # for i, offset in enumerate(d):
    #     loss = coupling_loss_lateral_offset_fiber(mfd, mfd, offset)
    #     c[i] = loss
    #     print(f"{offset:.2f}, {loss:.3f}, {power_to_db(loss):.3f}")
    #
    # plt.plot(d, c)
    # plt.ylabel('Coupling Efficiency (%)')
    # plt.xlabel('Lateral Displacement (um)')
    #
    # p = [5]
    # l = np.zeros_like(p)
    # for i, offset in enumerate(p):
    #     loss = coupling_loss_lateral_offset_fiber(mfd, mfd, offset)
    #     l[i] = loss
    #
    #     plt.annotate(f'{offset:.1f}um offset - {loss * 100:.1f}% Coupling',
    #                  (offset, loss),
    #                  textcoords="offset points",
    #                  xytext=(75, 0),
    #                  ha='center')
    #
    # plt.scatter(p,l)
    # plt.show()

    # wavelengths = [
    #     849.3,
    #     859.6,
    #     870.1,
    #     880.7,
    #     891.4,
    #     902.3,
    #     913.4,
    #     924.6,
    #     936.0,
    #     947.6,
    #     959.3,
    #     971.2,
    #     983.3,
    #     995.5,
    #     1008.0,
    #     1020.6,
    #     1033.4,
    #     1046.4
    # ]
    #
    # mfds = {}
    #
    #
    #
    # import csv
    #
    # with open('MFDs.csv', 'w', newline='') as csvfile:
    #     spamwriter = csv.writer(csvfile, delimiter=',',
    #                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #
    #     spamwriter.writerow(['Wavelegnth', 'MFD'])
    #
    #     for wv in wavelengths:
    #         m = fiber_MFD(0.076, 8.3e-6, wv * 1e-9)
    #         print(wv, m)
    #
    #         spamwriter.writerow([wv, m])
    #

    cladding_n = 1.45
    core_n = 1.452
    core_diameter = 8.3e-6

    diff_n = core_n - cladding_n

    # for i in np.arange(1, 5, .1):
    i = 2.62
    new_diameter = core_diameter * i
    n_scale = np.square(core_diameter) / np.square(new_diameter)
    new_core_n = diff_n * n_scale + cladding_n
    new_NA = fiber_NA(new_core_n, cladding_n)
    new_mfd = fiber_MFD(new_NA, new_diameter)
    print(i, new_diameter, n_scale, new_core_n, new_NA, new_mfd)

    # m = fiber_MFD(0.076, 8.3e-6, wv * 1e-9)
    # print(wv, m)
