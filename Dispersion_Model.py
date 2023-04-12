import logging

import numpy as np
from matplotlib import pyplot as plt
import scipy


def normal_dist(t, t0, sigma):
    return np.exp(-np.square(t - t0)) / (2 * np.square(sigma)) / (sigma * np.sqrt(2 * np.pi))


def power_to_db(power: float, reference_power: float = 1) -> float:
    return 10 * np.log10(power / reference_power)


def db_to_power(db: float, reference_power: float = 1) -> float:
    return 10 ** ((db) / 10) * reference_power


def dispersion_SNR(fm_power: float,
                   hom_power: float,
                   fm_attenuation: float,
                   hom_attenuation: float,
                   dispersion: float,
                   distance: float,
                   clock_rate: float = 16e9,
                   pulse_width: float = 62.5e-12 / 3,
                   num_pulses_analysis: int = 5,
                   plot_pulses=False,
                   plot_signal=False):
    num_points = 10001

    time = np.linspace(-0.5e-9, 0.5e-9, 10001)
    dt = time[0] - time[1]
    UI = 1 / clock_rate
    UI_int = int(UI / dt)

    mid_point = int(num_points / 2)
    mid_time = time[mid_point]

    peak_fm_power_dB = fm_power - distance * fm_attenuation
    peak_fm_power = db_to_power(peak_fm_power_dB)
    fm_pulse = peak_fm_power * scipy.stats.norm.pdf(time, 0, pulse_width)

    hom_pulses = []
    for pulse_number in range(num_pulses_analysis):
        hom_delay = dispersion * distance + (1 / clock_rate) * pulse_number
        peak_hom_power_dB = hom_power - distance * hom_attenuation
        peak_hom_power = db_to_power(peak_hom_power_dB)
        hom_pulse = peak_hom_power * scipy.stats.norm.pdf(time, hom_delay, pulse_width)
        hom_pulses.append(hom_pulse)

    signal = fm_pulse + hom_pulses[0]

    eq_tap_1 = np.roll(signal, -UI_int) * -signal[mid_point - UI_int] / signal[mid_point]
    eq_tap_2 = signal
    eq_tap_3 = np.roll(signal, UI_int) * -signal[mid_point + UI_int] / signal[mid_point]

    signal = eq_tap_1 + eq_tap_2 + eq_tap_3

    noise = np.zeros_like(hom_pulses[0])
    for hom_pulse in hom_pulses[1:]:
        noise += hom_pulse

    SNR = signal[mid_point] / noise[mid_point]
    SNR_dB = power_to_db(SNR)

    if plot_pulses:
        plt.plot(time, fm_pulse)
        for hom_pulse in hom_pulses:
            plt.plot(time, hom_pulse)

        plt.yticks([])
        plt.show()

    if plot_signal:
        plt.figure(figsize=(10, 6))

        plt.plot(time, eq_tap_1, color='green', linestyle='solid', label='EQ 1')
        plt.plot(time, eq_tap_3, color='green', linestyle='solid', label='EQ 3')

        plt.plot(time, signal, color='blue', linestyle='solid', label='Signal')
        plt.plot(time, noise, color='red', linestyle='solid', label='Noise')

        plt.plot(time, fm_pulse, color='blue', linestyle='dotted', label='FM Signal Contribution')
        plt.plot(time, hom_pulses[0], color='blue', linestyle='dashed', label='HOM Signal Contribution')

        for i, hom_pulse in enumerate(hom_pulses[1:]):
            if i == 0:
                plt.plot(time, hom_pulse, color='red', linestyle='dashed', label='HOM Noise Contributions')
            else:
                plt.plot(time, hom_pulse, color='red', linestyle='dashed')

        plt.suptitle(f"Signal at {distance} m propagation. w/ Attenuation", fontsize=20)
        plt.title(f"16GHz signal. 1/e2 Pulse width ~ 20.8ps", fontsize=10)
        plt.legend()

        plt.xlabel("Time (s)")
        plt.ylabel("Power (a.u.)")

        # plt.xticks([])
        plt.yticks([])

        plt.show()

    return SNR_dB


def SNR_vs_distance(fm_power: float,
                    hom_power: float,
                    fm_attenuation: float,
                    hom_attenuation: float,
                    dispersion: float,
                    clock_rate: float = 16e9,
                    pulse_width: float = 62.5e-12 / 3,
                    num_pulses_analysis: int = 5,
                    min_distance: int = 0,
                    max_distance: int = 100,
                    num_distance_points: int = 1001
                    ):
    # logging.debug( f'fm_Power:{fm_power} - hom_power{hom_power} - fm_attenuation{fm_attenuation} - hom_attenuation:{hom_attenuation} - dispersion:{dispersion}')

    distances = np.linspace(min_distance, max_distance, num_distance_points)
    SNRs = np.zeros_like(distances)

    for i, distance in enumerate(distances):
        SNR = dispersion_SNR(fm_power=fm_power,
                             hom_power=hom_power,
                             fm_attenuation=fm_attenuation,
                             hom_attenuation=hom_attenuation,
                             dispersion=dispersion,
                             distance=distance,
                             clock_rate=clock_rate,
                             pulse_width=pulse_width,
                             num_pulses_analysis=num_pulses_analysis,
                             plot_pulses=False,
                             plot_signal=True)

        SNRs[i] = SNR

    return distances, SNRs


def plot_SNR_v_distance(distances, SNRs):
    d_worst = distances[np.argmin(SNRs)]
    SNR_worst = np.min(SNRs)
    # print(SNR_worst)

    plt.plot(distances, SNRs, label='850nm')
    plt.title(f"Worst case SNR is {SNR_worst:.2f} dB at {d_worst:.1f} m")
    plt.xlabel("Distance (m)")
    plt.ylabel("SNR (dB)")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s',
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler(),
            # logging.FileHandler(f'log_files/{https://picinternational.net/}-{time.strftime("%Y-%m-%d--%H-%M-%S")}.log'),
        ]
    )
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    # s/m
    dispersion_850 = -4.055206934694106e-12

    # dB/m
    attenuation_850 = 0.267

    fm_power_850 = -0.4994730537
    hom_power_850 = -10.85289436

    # distances, SNRs = SNR_vs_distance(
    #
    #     fm_power=fm_power_850,
    #     hom_power=hom_power_850,
    #     fm_attenuation=0,
    #     hom_attenuation=attenuation_850,
    #     dispersion=dispersion_850
    #
    # )

    # plot_SNR_v_distance(distances, SNRs)

    SNR = dispersion_SNR(fm_power=fm_power_850,
                         hom_power=hom_power_850,
                         fm_attenuation=0,
                         hom_attenuation=attenuation_850,
                         dispersion=dispersion_850,
                         distance=15,
                         plot_pulses=False,
                         plot_signal=True)
