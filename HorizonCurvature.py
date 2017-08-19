import pandas as pd
import numpy as np
import scipy.linalg
import datetime
import matplotlib.pyplot as plt

path = 'c:/data'
gridfile='GridExport.csv'
hf=path + '\\' +gridfile
grid=pd.read_csv(hf)


img = np.array(grid['Value'])
Z=img.reshape(190,197)




Colmax, Rowmax = Z.shape
data = np.zeros((81, 3))
Kmean = np.zeros(Z.shape)
Kgaus = np.zeros(Z.shape)
Kmax = np.zeros(Z.shape)
Kmin = np.zeros(Z.shape)
Kplus = np.zeros(Z.shape)
Kminus = np.zeros(Z.shape)
Si = np.zeros(Z.shape)

ti = datetime.datetime.now()


# @autojit
def runK():
    for ci in range(4, Colmax - 7):
        for ri in range(4, Rowmax - 7):
            i = 0
            for xs in range(0, 9):  # xs range to cacl surface
                x = xs + ci - 1
                for ys in range(0, 9):
                    y = ys + ri - 1
                    # print(x,y)
                    data[i] = [xs, ys, Z[x, y]]
                    i += 1

            A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]
            C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])
            f, d, e, c, a, b = C
            Kmean[ci, ri] = (a * (1 + e ** 2) - c * d * e + b * (1 + d ** 2)) / (1 + d ** 2 + e ** 2) ** (3 / 2)
            Kgaus[ci, ri] = (4 * a * b - c ** 2) / (1 + d ** 2 + e ** 2) ** 2
            Kmax[ci, ri] = Kmean[ci, ri] + (Kmean[ci, ri] ** 2 - Kgaus[ci, ri]) ** .5
            Kmin[ci, ri] = Kmean[ci, ri] - (Kmean[ci, ri] ** 2 - Kgaus[ci, ri]) ** .5
            Kplus[ci, ri] = (a + b) + ((a - b) ** 2 + c ** 2) ** .5
            Kminus[ci, ri] = (a + b) - ((a - b) ** 2 + c ** 2) ** .5
            Si[ci, ri] = 2 / np.pi * np.arctan((Kmin[ci, ri] + Kmax[ci, ri]) / (Kmax[ci, ri] - Kmin[ci, ri]))
        print(ci, "out of, ", Colmax)

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


