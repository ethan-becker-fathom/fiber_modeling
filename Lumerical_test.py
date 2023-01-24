import sys
sys.path.append("C:\\Program Files\\Lumerical\\v222\\api\\python\\")

import lumapi

lum = lumapi.MODE()

lum.load(
    "C:\\Users\\fathom-lumerical\\Fathom Radiant Dropbox\\Engineering\\Fiber\\Simulation\\Lumerical\\SIF_Trench 8.3-20-30.lms")

# lum.findmodes()

print(lum.getresult("FDE::data::mode1", "farfield"))


E1 = lum.getresult("FDE::data::mode1")
print(E1.split('\n'))
# print(E1)
# print(type(E1))
# print(E1.keys())
# E2 = lum.getresult("FDE::data::mode2","E")



# for dataset in E1.split('\n'):
#     print(dataset)
#     print(lum.getresult("FDE::data::mode1", dataset))

# data1 = lum.getresultdata("FDE::data::mode2")
# print(data1)

# lum.matlabsave("test.mat")
