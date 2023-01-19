import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

def coupling_loss_lateral_offset_fiber(mfd_1_um, mfd_2_um, offset_um):
    mfr_1_um = mfd_1_um / 2
    mfr_2_um = mfd_2_um / 2
    
    return (((2 * mfr_1_um * mfr_2_um) / (mfr_1_um ** 2 + mfr_2_um ** 2)) ** 2) * np.exp(
        (-2 * offset_um ** 2) / (mfr_1_um ** 2 + mfr_2_um ** 2))

mfd1 = 9.1 #mode field diameter 1 [um]
mfd2 = 9.1 #mode field diameter 2[um]
offset = 3 #offset from nominal [um]
T = coupling_loss_lateral_offset_fiber(mfd1,mfd2,offset) #test

#inverse of coupling_loss_lateral_offset_fiber
def offsetFromTrans(mfd_1_um, mfd_2_um, T):
    mfr_1_um = mfd_1_um / 2
    mfr_2_um = mfd_2_um / 2

    return ((-0.5*(mfr_1_um**2+mfr_2_um**2)*np.log(T*((mfr_1_um**2+mfr_2_um**2)/(2*mfr_1_um*mfr_2_um))**2))**0.5)

import csv
#import loss values
yieldPercent = []
dBLoss = []
with open("gradeBcdf.csv", 'r') as file:
    data = csv.reader(file)
    for item in data:
        dBLoss.__iadd__([float(item[0])])
        yieldPercent.__iadd__([float(item[1])])

yieldFrac = np.array(yieldPercent)*0.01 #yield fraction (0.XY = XY%)
transFrac = [10**(-a/10) for a in dBLoss] #transmission fraction (0.XY = XY%)
zScore = [st.norm.ppf(a) for a in yieldFrac] #zscore derived from yield fraction
offset = [offsetFromTrans(mfd1, mfd2, a) for a in transFrac] #offset [um] derived from transmissionFrac

###show data
##for i in range(len(yieldFrac)):
##    print(["%.4f" % yieldFrac[i],"%.4f" % zScore[i],"%.4f" % transFrac[i], "%.4f" % offset[i]])
##
##plt.plot(zScore,offset)
##plt.show()

#linear algebra representation of linear regression: x = inv(ATA)AT*b for Ax=b
def linRegress(x,y):
    A = np.array([[a,1] for a in x])
    invAtA = np.linalg.inv(np.matmul(np.transpose(A),A))
    b = np.matmul(np.matmul(invAtA,np.transpose(A)),y)
    return(b)

def power_to_db(power: float, reference_power: float = 1) -> float:
    return 10 * np.log10(power / reference_power)

#for z = (d-mu)/sig = d/sig-mu/sig, the slope is 1/sig, and the yint is mu/sig
a,b = linRegress(offset,zScore)
sigma = 1/a
mu = -sigma*b


print("mean offset = ",mu)
print("st.dev =",sigma)
loss = coupling_loss_lateral_offset_fiber(mfd1,mfd2,mu)
print("mean coupling efficiency =",loss)
print("db = ", power_to_db(loss))