import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import time
from numba import jit


filename = 'c:/data/test.MIG.0.h5'
print(filename)
sgydata = h5py.File(filename)



Kmean = np.zeros((101, 101,151))
Kgaus = np.zeros((101, 101,151))
Kmax = np.zeros((101, 101,151))


data = np.zeros((9, 3))

tm = 75
smp = 8
@jit
def curv():
    for tm in range(8,19):
        for ln_idx in range(1, 100):
            for tr_idx in range(1, 100):
                i = 0
                strttm = time.time()
                for pln in range(-1, 2):
                    for ptr in range(-1, 2):
                        #print(ln_idx,tr_idx)
                        mdpnt = sgydata['Traces'][ln_idx, tr_idx, tm-8:tm+8]
                        obspnt = sgydata['Traces'][ln_idx + pln, tr_idx + ptr, tm-8:tm+8]
                        corr = np.correlate(mdpnt, obspnt, "same")
                        #print (ln_idx,tr_idx,np.argmax(corr))

                        data[i] = [pln, ptr, np.argmax(corr)]
                        i += 1

                A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]
                C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])
                f, d, e, c, a, b = C
                Kmean[ln_idx, tr_idx,tm] = (a * (1 + e ** 2) - c * d * e + b * (1 + d ** 2)) / (1 + d ** 2 + e ** 2) ** (3 / 2)
                Kgaus[ln_idx, tr_idx,tm] = (4 * a * b - c ** 2) / (1 + d ** 2 + e ** 2) ** 2
                Kmax[ln_idx, tr_idx,tm] = Kmean[ln_idx, tr_idx,tm] + (Kmean[ln_idx, tr_idx,tm] ** 2 - Kgaus[ln_idx, tr_idx,tm]) ** .5
                runtime = time.time()-strttm
                print(tm,ln_idx, tr_idx, Kmax[ln_idx, tr_idx,tm],runtime)


curv()

