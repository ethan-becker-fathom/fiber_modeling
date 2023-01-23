import sys, os
import time
import json
import shutil

sys.path.append("C:\\Program Files\\Lumerical\\v222\\api\\python\\")

import lumapi

lum = lumapi.MODE()
# lum = lumapi.MODE(hide=True)

lum.load(
    "C:\\Users\\fathom-lumerical\\Fathom Radiant Dropbox\\Engineering\\Fiber\\Simulation\\Lumerical\\SIF_Trench 8.3-20-30.lms")

lum.findmodes()

# lum.matlabsave()
lum.exportcsvresults("test.csv")