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

    mfd = 10.4

    for offset in np.arange(0, 2, 0.01):
        loss = coupling_loss_lateral_offset_fiber(mfd, mfd, offset)
        print(f"{offset:.2f}, {loss:.3f}, {power_to_db(loss):.3f}")
