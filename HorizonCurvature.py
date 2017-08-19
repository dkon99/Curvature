import pandas as pd
import numpy as np
import scipy.linalg
import datetime
import matplotlib.pyplot as plt


# Load data from csv file
path = 'c:/data'
gridfile='GridExport.csv'
hf=path + '\\' +gridfile
grid=pd.read_csv(hf)


 # create array for calculations

Z = np.array(grid['Value'])
Z=Z.reshape(190,197)



# set shape of curvature arrays
Colmax, Rowmax = Z.shape
data = np.zeros((9,3))
Kmean = np.zeros(Z.shape)
Kgaus = np.zeros(Z.shape)
Kmax = np.zeros(Z.shape)
Kmin = np.zeros(Z.shape)
Kplus = np.zeros(Z.shape)
Kminus = np.zeros(Z.shape)
Si = np.zeros(Z.shape)

ti = datetime.datetime.now()


# curvature loop

''' Idea is to loop through each column and trace of horizon array
    - pull out a patch at each point, Z
    - Calc least sqaures fit for patch
    -Use scalars from least squares fit in Roberts formulas for Curvature'''


# @autojit
def runK():
    for ci in range(1, Colmax - 1):
        # column loop
        for ri in range(1, Rowmax - 1):
            #range loop
            i = 0
            for xs in range(-1,2,1):  # xs range to cacl surface

                x = xs + ci #x for each sample in patch
                for ys in range(-1,2,1):
                    y = ys + ri    #y for each sample in patch
                    # print(x,y)
                    data[i] = [xs,ys,Z[x, y]]
                    #data[i] = [x, y, Z[x, y]]  # Data for surface with center xs,ys. Z is surface data for patch
                    i += 1
            #print (len(data[0]))
            A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]
            C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])
            f, d, e, c, a, b = C
            Kmean[ci, ri] = (a * (1 + e ** 2) - c * d * e + b * (1 + d ** 2)) / (1 + d ** 2 + e ** 2) ** (3 / 2)
            Kgaus[ci, ri] = (4 * a * b - c ** 2) / (1 + d ** 2 + e ** 2) ** 2
            Kmax[ci, ri] = Kmean[ci, ri] + (Kmean[ci, ri] ** 2 - Kgaus[ci, ri]) ** .5
            # Kmin[ci, ri] = Kmean[ci, ri] - (Kmean[ci, ri] ** 2 - Kgaus[ci, ri]) ** .5
            # Kplus[ci, ri] = (a + b) + ((a - b) ** 2 + c ** 2) ** .5
            # Kminus[ci, ri] = (a + b) - ((a - b) ** 2 + c ** 2) ** .5
            # Si[ci, ri] = 2 / np.pi * np.arctan((Kmin[ci, ri] + Kmax[ci, ri]) / (Kmax[ci, ri] - Kmin[ci, ri]))
        print(ci, "out of", Colmax)

runK()
tstop = datetime.datetime.now()
trun = tstop - ti
print(trun)

plt.figure(figsize=(12,6))

plt.subplot(221)
plt.imshow(Z, cmap='Accent')
plt.ylim(min(plt.ylim()), max(plt.ylim()))

plt.subplot(222)
plt.imshow(Kmax,cmap='afmhot')
plt.ylim(min(plt.ylim()), max(plt.ylim()))
plt.show()


