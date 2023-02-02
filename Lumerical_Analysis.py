import sys, os
import json
from json import JSONEncoder
import numpy as np
import logging
from pathlib import Path
import time

lumapi_path = Path("C:\\Program Files\\Lumerical\\v222\\api\\python\\")
sys.path.append(str(lumapi_path))

import lumapi


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if np.iscomplexobj(obj):
                return obj.real.tolist()
            return obj.tolist()
        return JSONEncoder.default(self, obj)


# class Lumerical_Mode:
#     def __int__(self):
#         self.lum = None
#
#         self.default_analysis_parameters = {
#             'maximum number of modes to store': 6,
#             'search': 'in range',
#             'convergence tolerance': 1e-12
#         }
#
#     def open(self, hidden=False):
#         self.lum = lumapi.MODE(hide=hidden)
#
#     def load_model(self, model_path):
#         self.lum.load(model_path)
#
#     def get_analysis_parameters(self, log=True):
#         params = lum.getanalysis().split('\n')
#         if log:
#             for param in params:
#                 try:
#                     logging.debug(f'{param}: {lum.getanalysis(item)}')
#                 except:
#                     logging.info(f'{param}. **Unable to get values**')
#
#         return params
#
#     def set_analysis_parameter(self, parameter: str, value):
#         logging.debug(f'Setting analysis parameter {parameter}: {value}')
#         lum.setanalysis(parameter, value)
#
#     def set_default_analysis_parameters(self):
#         for param, value in self.default_analysis_parameters.items():
#             self.set_analysis_parameter(param, value)
#
#     def set_lumerical_named_parameter(self, name: str, parameter: str, value):
#         self.lum.switchtolayout()
#         logging.debug(f'Setting named parameter {name},{parameter}: {value}')
#         lum.setnamed(name, parameter, value)
#
#     def find_modes(self):
#         logging.debug(f'Finding Modes')
#
#     def get_mode_overlaps(self, save_to_file: bool = True, folder: Path = Path()):
#         modes = {}
#         modes['wavelength'] = wavelength_nm
#         modes['bend_radius'] = bend_radius
#
#         modes['deltan_core'] = deltan_core
#         modes['deltan_ring'] = deltan_ring
#         modes['core_radius'] = core_radius
#         modes['trench_ID'] = trench_ID
#         modes['trench_OD'] = trench_OD
#
#         modes['mode_coupling'] = []
#
#         d = np.linspace(0, 10e-6, 101)
#
#         for i in range(1, 7):
#             for j in range(1, 7):
#                 for k, x in enumerate(d):
#                     overlap = lum.overlap(
#                         f"::model::FDE::data::mode{i}",
#                         f"::model::FDE::data::mode{j}",
#                         x,
#                         0,
#                         0
#                     )[0, 0]
#
#                     modes['mode_coupling'].append([i, j, x, overlap])
#
#         if save_to_file:
#             file_path = folder.joinpath(f'{wavelength_nm}nm_bend-{bend_radius_mm}mm_mode-overlaps.json')
#             with open(file_path, "w") as outfile:
#                 json.dump(modes, outfile, cls=NumpyArrayEncoder)
#
#         return modes
#
#     def get_mode_summary(self, save_to_file: bool = True, folder: Path = Path()):
#         modes = {}
#         modes['wavelength'] = wavelength_nm
#         modes['bend_radius'] = bend_radius
#
#         modes['deltan_core'] = deltan_core
#         modes['deltan_ring'] = deltan_ring
#         modes['core_radius'] = core_radius
#         modes['trench_ID'] = trench_ID
#         modes['trench_OD'] = trench_OD
#
#         for i in range(1, 7):
#             try:
#                 modes[str(i)] = {}
#                 modes[str(i)]['n'] = n
#                 modes[str(i)]['loss'] = lum.getresult(f"::model::FDE::data::mode{i}", "loss")
#                 modes[str(i)]['neff'] = float(lum.getresult(f"::model::FDE::data::mode{i}", "neff").flatten()[0])
#                 modes[str(i)]['mode_effective_area'] = lum.getresult(f"::model::FDE::data::mode{i}",
#                                                                      "mode effective area")
#                 modes[str(i)]['TE_polarization_fraction'] = lum.getresult(f"::model::FDE::data::mode{i}",
#                                                                           "TE polarization fraction")
#             except:
#                 pass
#
#         if save_to_file:
#             file_path = results_folder.joinpath(f'{wavelength_nm}nm_bend-{bend_radius_mm}mm_mode-summary.json')
#             with open(file_path, "w") as outfile:
#                 json.dump(modes, outfile, cls=NumpyArrayEncoder)
#
#         return modes


def set_lumerical_analysis_parameter(parameter: str, value):
    logging.debug(f'Setting analysis parameter {parameter}: {value}')
    lum.setanalysis(parameter, value)


def set_lumerical_named_parameter(name: str, parameter: str, value):
    logging.debug(f'Setting named parameter {name},{parameter}: {value}')
    lum.setnamed(name, parameter, value)


def open_lumerical_mode(hidden=False):
    return lumapi.MODE(hide=hidden)


if __name__ == '__main__':

    Path('log_files/').mkdir(exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s',
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'log_files/{Path(__file__).stem}-{time.strftime("%Y-%m-%d--%H-%M-%S")}.log'),
        ]
    )

    lum = lumapi.MODE()
    # lum = lumapi.MODE(hide=True)

    model_path = Path(
        "C:\\Users\\fathom-lumerical\\Fathom Radiant Dropbox\\Engineering\\Fiber\\Simulation\\Lumerical\\SIF_Trench 8.3-20-30.lms")
    lum.load(str(model_path))

    for item in lum.getanalysis().split('\n'):
        try:
            logging.debug(f'{item}: {lum.getanalysis(item)}')
        except:
            logging.info(f'{item}. **Unable to get values**')

    d_core = 8.3e-6
    CCDR1 = 2.4
    CCDR2 = 3.6
    deltan_core = 2e-3
    deltan_ring = 5e-3

    results_folder = Path(f'Lumerical_Results/TAF_{d_core * 1e6:.1f}-{CCDR1}-{CCDR2}_{deltan_core}-{deltan_ring}')
    results_folder.mkdir(parents=True, exist_ok=True)

    core_radius = d_core / 2
    trench_ID = d_core * CCDR1 / 2
    trench_OD = d_core * CCDR2 / 2

    lum.switchtolayout()
    set_lumerical_named_parameter("core", "radius", core_radius)
    set_lumerical_named_parameter("ring", "inner radius", trench_ID)
    set_lumerical_named_parameter("ring", "outer radius", trench_OD)

    set_lumerical_analysis_parameter('maximum number of modes to store', 6)
    set_lumerical_analysis_parameter('search', 'in range')
    set_lumerical_analysis_parameter('convergence tolerance', 1e-12)

    bend_radii = [0, .3, .2, .1, .075, .05, .045, .04, .035, .03, .025, .02, .015, .01, .005]
    # bend_radii = [0]

    wavelengths_nm = [800, 810, 820, 830, 840, 850, 860, 870, 880, 890, 900, 910, 920, 930, 940, 950, 960, 970, 980, 990, 1000, 1010, 1020, 1030, 1040, 1050]
    # wavelengths_nm = [800]

    for wavelength_nm in wavelengths_nm:

        # material dispersion for fused silica
        Sellmeier_B = np.array([0.6961663, 0.4079426, 0.8974794])
        Sellmeier_C = np.array([0.0684043 ** 2, 0.1162414 ** 2, 9.896161 ** 2])

        wavelength_mu = wavelength_nm / 1000

        nhelp = 1
        for idx, ival in enumerate(Sellmeier_B):
            nhelp += Sellmeier_B[idx] * wavelength_mu ** 2 / (wavelength_mu ** 2 - Sellmeier_C[idx])
        n = np.sqrt(nhelp)

        # activate for fixed refractive index
        # n = 1.45

        lum.switchtolayout()
        set_lumerical_named_parameter('clad', 'index', n)
        set_lumerical_named_parameter('core', 'index', n + deltan_core)
        set_lumerical_named_parameter('ring', 'index', n - deltan_ring)

        logging.debug(f'Set n to {n}, for wavelength {wavelength_nm}nm')


        for bend_radius in bend_radii:

            wavelength = wavelength_nm / 1e9
            bend_radius_mm = bend_radius * 1e3

            set_lumerical_analysis_parameter('wavelength', wavelength)
            set_lumerical_analysis_parameter('n1', n + deltan_core)
            set_lumerical_analysis_parameter('n2', n - deltan_core)

            if bend_radius == 0:
                set_lumerical_analysis_parameter('bent waveguide', 0)
            else:
                set_lumerical_analysis_parameter('bent waveguide', 1)
                set_lumerical_analysis_parameter('bend radius', bend_radius)

            logging.info(f'Finding Modes for wavelength:{wavelength_nm}nm, and bending radius: {bend_radius_mm}mm')

            lum.findmodes()

            if bend_radius == 0:
                # if False:
                modes = {}
                modes['wavelength'] = wavelength_nm
                modes['bend_radius'] = bend_radius

                modes['deltan_core'] = deltan_core
                modes['deltan_ring'] = deltan_ring
                modes['cladding_n'] = n
                modes['core_n'] = n + deltan_core
                modes['trench_n'] = n - deltan_ring
                modes['core_radius'] = core_radius
                modes['trench_ID'] = trench_ID
                modes['trench_OD'] = trench_OD

                modes['mode_coupling'] = []

                d = np.linspace(0, 10e-6, 101)


                for i in range(1, 7):
                    for j in range(1, 7):
                        power = np.zeros_like(d)
                        for k, x in enumerate(d):
                            overlap = \
                                lum.overlap(f"::model::FDE::data::mode{i}", f"::model::FDE::data::mode{j}", x, 0, 0)[
                                    0, 0]
                            modes['mode_coupling'].append([i, j, x, overlap])

                file_path = results_folder.joinpath(f'{wavelength_nm}nm_bend-{bend_radius_mm}mm_mode-overlaps.json')
                with open(file_path, "w") as outfile:
                    json.dump(modes, outfile, cls=NumpyArrayEncoder)

            modes = {}
            modes['wavelength'] = wavelength_nm
            modes['bend_radius'] = bend_radius
            modes['cladding_n'] = n
            modes['core_n'] = n + deltan_core
            modes['trench_n'] = n - deltan_ring
            modes['deltan_core'] = deltan_core
            modes['deltan_ring'] = deltan_ring
            modes['core_radius'] = core_radius
            modes['trench_ID'] = trench_ID
            modes['trench_OD'] = trench_OD

            for i in range(1, 7):
                try:
                    modes[str(i)] = {}
                    modes[str(i)]['n'] = n
                    modes[str(i)]['loss'] = lum.getresult(f"::model::FDE::data::mode{i}", "loss")
                    modes[str(i)]['neff'] = float(lum.getresult(f"::model::FDE::data::mode{i}", "neff").flatten()[0])
                    modes[str(i)]['mode_effective_area'] = lum.getresult(f"::model::FDE::data::mode{i}",
                                                                         "mode effective area")
                    modes[str(i)]['TE_polarization_fraction'] = lum.getresult(f"::model::FDE::data::mode{i}",
                                                                              "TE polarization fraction")
                except:
                    pass

            file_path = results_folder.joinpath(f'{wavelength_nm}nm_bend-{bend_radius_mm}mm_mode-summary.json')
            with open(file_path, "w") as outfile:
                json.dump(modes, outfile, cls=NumpyArrayEncoder)