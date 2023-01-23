import sys, os
import time
import json

sys.path.append("C:\\Program Files\\Lumerical\\v222\\api\\python\\")

import lumapi

lum = lumapi.MODE()
# lum = lumapi.MODE(hide=True)

lum.load(
    "C:\\Users\\fathom-lumerical\\Fathom Radiant Dropbox\\Engineering\\Fiber\\Simulation\\Lumerical\\SIF_Trench 8.3-20-30.lms")

d_core = 8.3e-6
CCDR1 = 2.4
CCDR2 = 3.6

# lum.switchtolayout()
# lum.setnamed("core","radius",d_core/2)
# lum.setnamed("trench","inner radius",d_core*CCDR1/2)
# lum.setnamed("trench","outer radius",d_core*CCDR2/2)


for item in lum.getanalysis().split('\n'):
    try:
        print(item, lum.getanalysis(item))
    except:
        print(item)

lum.setanalysis('maximum number of modes to store', 6)

bend_radii = [0, .3, .2, .1, .075, .05, .045, .04, .035, .03, .025, .02, .015, .01, .005]
# bend_radii = [.01]
# wavelengths_nm = [850, 860, 870, 880, 890, 900, 910, 920, 930, 940, 950, 960, 970, 980, 990, 1000, 1010, 1020, 1030, 1040, 1050]
wavelengths_nm = [900, 910, 920, 930, 940, 950, 960, 970, 980, 990, 1000, 1010, 1020, 1030, 1040, 1050]

for wavelength_nm in wavelengths_nm:

    for bend_radius in bend_radii:

        wavelength = wavelength_nm / 1e9
        # wavelength_nm = wavelength * 1e9

        # bend_radius = .05
        bend_radius_mm = bend_radius * 1e3

        lum.setanalysis('wavelength', wavelength)

        if bend_radius == 0:
            lum.setanalysis('bent waveguide', 0)
        else:
            lum.setanalysis('bent waveguide', 1)
            lum.setanalysis('bend radius', bend_radius)

        print(wavelength_nm, bend_radius_mm)

        lum.findmodes()

        modes = {}
        for i in range(1, 7):
            try:
                modes[i] = {}
                modes[i]['loss'] = lum.getresult(f"::model::FDE::data::mode{i}", "loss")
                modes[i]['neff'] = float(lum.getresult(f"::model::FDE::data::mode{i}", "neff").flatten()[0])
            except:
                pass

        print(modes)

        with open(f"Lumerical_Results\\TAF_8.3-20-30_wavelength-{wavelength_nm}nm_bend-{bend_radius_mm}mm.json",
                  "w") as outfile:
            json.dump(modes, outfile)

        # lum.matlabsave(f"Lumerical_Results\\TAF_8.3-20-30_wavelength-{wavelength_nm}nm_bend-{bend_radius_mm}mm.mat")
