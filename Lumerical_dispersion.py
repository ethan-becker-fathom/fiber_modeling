import sys, os
import time
import json
import numpy as np

sys.path.append("C:\\Program Files\\Lumerical\\v222\\api\\python\\")

import lumapi

lum = lumapi.MODE()
# lum = lumapi.MODE(hide=True)

lum.load(
    "C:\\Users\\fathom-lumerical\\Fathom Radiant Dropbox\\Engineering\\Fiber\\Simulation\\Lumerical\\SIF_Trench 8.3-20-30.lms")

d_core = 8.3e-6
CCDR1 = 2.4
CCDR2 = 3.6
deltan_core = 2e-3
deltan_ring = 5e-3

lum.switchtolayout()
lum.setnamed("core","radius",d_core/2)
lum.setnamed("ring","inner radius",d_core*CCDR1/2)
lum.setnamed("ring","outer radius",d_core*CCDR2/2)


for item in lum.getanalysis().split('\n'):
    try:
        print(item, lum.getanalysis(item))
    except:
        print(item)

lum.setanalysis('maximum number of modes to store', 6)


bend_radii = [0, .3, .2, .1, .075, .05, .045, .04, .035, .03, .025, .02, .015, .01, .005]
# bend_radii = [.01]
wavelengths_nm = [850, 860, 870, 880, 890, 900, 910, 920, 930, 940, 950, 960, 970, 980, 990, 1000, 1010, 1020, 1030, 1040, 1050]

#bend_radii = [0, .01, .005]
#wavelengths_nm = [850, 900, 950, 1000]

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
    lum.setnamed('clad','index',n)
    lum.setnamed('core','index',n+deltan_core)
    lum.setnamed('ring','index',n-deltan_ring)

    for bend_radius in bend_radii:

        wavelength = wavelength_nm / 1e9
        # wavelength_nm = wavelength * 1e9

        # bend_radius = .05
        bend_radius_mm = bend_radius * 1e3

        lum.setanalysis('wavelength', wavelength)
        lum.setanalysis('n1',n+deltan_core)
        lum.setanalysis('n2',n-deltan_core)

        if bend_radius == 0:
            lum.setanalysis('bent waveguide', 0)
        else:
            lum.setanalysis('bent waveguide', 1)
            lum.setanalysis('bend radius', bend_radius)

        print(wavelength_nm, bend_radius_mm, n)

        lum.findmodes()

        modes = {}
        for i in range(1, 7):
            try:
                modes[i] = {}
                modes[i]['wavelength'] = wavelength_nm
                modes[i]['bend_radius'] = bend_radius
                modes[i]['n'] = n
                modes[i]['loss'] = lum.getresult(f"::model::FDE::data::mode{i}", "loss")
                modes[i]['neff'] = float(lum.getresult(f"::model::FDE::data::mode{i}", "neff").flatten()[0])
            except:
                pass

        # print(modes)

        with open(f"Lumerical_Results\\TAF_8.3-20-30_dispersion-{wavelength_nm}nm_bend-{bend_radius_mm}mm.json",
                  "w") as outfile:
            json.dump(modes, outfile)

        # lum.matlabsave(f"Lumerical_Results\\TAF_8.3-20-30_wavelength-{wavelength_nm}nm_bend-{bend_radius_mm}mm.mat")
