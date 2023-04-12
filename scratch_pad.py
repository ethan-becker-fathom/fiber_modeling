import logging
from pathlib import Path

def update_named_parameters(named_parameters):
    for name, params in named_parameters.items():
        for param, value in params.items():
            if param == '*delta_index':
                params['index'] = named_parameters['clad']['index'] + value


if __name__ == '__main__':
    # logging.basicConfig(
    #     format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s',
    #     level=logging.DEBUG,
    #     handlers=[
    #         logging.StreamHandler(),
    #         # logging.FileHandler(f'log_files/{Path(__file__).stem}-{time.strftime("%Y-%m-%d--%H-%M-%S")}.log'),
    #     ]
    # )
    #
    # d_core = 8.3e-6
    # CCDR1 = 2.4
    # CCDR2 = 3.6
    # deltan_core = 2e-3
    # deltan_ring = 5e-3
    #
    # core_radius = d_core / 2
    # trench_ID = d_core * CCDR1 / 2
    # trench_OD = d_core * CCDR2 / 2
    # n = 1.45
    #
    # named_parameters = {
    #     "core": {
    #         "radius": core_radius,
    #         'index': n + deltan_core,
    #         '*delta_index': deltan_core
    #
    #     },
    #     "ring": {
    #         "inner radius": trench_ID,
    #         "outer radius": trench_OD,
    #         'index': n - deltan_ring,
    #         '*delta_index': -deltan_ring
    #
    #     },
    #     'clad': {
    #         'index': n
    #     }
    # }
    #
    # print(named_parameters)
    #
    # named_parameters['clad']['index'] = 1.4
    # update_named_parameters(named_parameters)
    #
    # print(named_parameters)

    print(Path(__file__))
