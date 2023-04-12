import Lumerical_Analysis
from pathlib import Path
import logging
import time

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

    d_core = 8.3e-6
    CCDR1 = 2.4
    CCDR2 = 3.6
    deltan_core = 2e-3
    deltan_ring = 5e-3

    core_radius = d_core / 2
    trench_ID = d_core * CCDR1 / 2
    trench_OD = d_core * CCDR2 / 2
    n = 1.45

    named_parameters = {
        "core": {
            "radius": core_radius,
            'index': n + deltan_core,
            '*delta_index': deltan_core

        },
        "ring": {
            "inner radius": trench_ID,
            "outer radius": trench_OD,
            'index': n - deltan_ring,
            '*delta_index': -deltan_ring

        },
        'clad': {
            'index': n
        }

    }


    model_path = Path(
        "C:\\Users\\fathom-lumerical\\Fathom Radiant Dropbox\\Engineering\\Fiber\\Simulation\\Lumerical\\SIF_Trench 8.3-20-30.lms")

    results_folder = Path(f'Lumerical_Results/TAF_{d_core * 1e6:.1f}-{CCDR1}-{CCDR2}_{deltan_core}-{deltan_ring}')
    results_folder.mkdir(parents=True, exist_ok=True)


    lum = Lumerical_Analysis.Lumerical_Mode(named_parameters=named_parameters)

    lum.open()

    lum.load_model(model_path)

    lum.set_default_analysis_parameters()
    lum.set_named_parameters()

    lum.set_bending_radius(0)
    lum.set_wavelength(800)

    lum.set_named_parameters()

    lum.set_analysis_parameter('n1', lum.named_parameters['clad']['index'] + deltan_core)
    lum.set_analysis_parameter('n2', lum.named_parameters['clad']['index'] - deltan_core)

    lum.find_modes()

    lum.get_mode_summary(save_to_file=True, folder=results_folder)
