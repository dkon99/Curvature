import pandas as pd
import numpy as np
import scipy.linalg
import datetime
import matplotlib.pyplot as plt

''' 
Horizon Curvature



Idea is to loop through each column and trace of horizon array
    - pull out a patch at each point
    - Calc least sqaures fit for patch
    -Use scalars from least squares fit in Roberts formulas for Curvature

 Least squares formula is
    y = ax^2 + by^2 + cxy + dx + ey + f

    for scipy.linalg.lstsq 

     y = A * p

     y = [y1,y2...yn]
     x = [x1,x2...xn]
     [A]T = [  [x^2],
            [y^2],
            [xy],
            [x],
            [y]
            [1,1,...1n]  ]
            The transpose puts the rows as above into columns
     p=[a,b,c,d,e,f]


    '''

# Load data from csv file
path = 'c:/data'
gridfile='GridExport.csv'
hf=path + '\\' +gridfile
grid=pd.read_csv(hf)


 # create array for calculations

Z = np.array(grid['Value'])
Z = Z.reshape(190, 197) # need to pass 2d array of z at x,y to func

stepout=7


# curvature loop


# @autojit
def runK(Z,stepout):
    ''' Param Z = 2D area (z(x,y))
        Param stepout = number of adjacent data to use
        ie stepout =1 is 3x3 patch
           stepout =2 is 5x5 patch...
     '''

    #

    Colmax, Rowmax = Z.shape

    data = np.zeros(((1+2*stepout)**2, 3))
    Kmean = np.zeros(Z.shape)
    Kgaus = np.zeros(Z.shape)
    Kmax = np.zeros(Z.shape)
    Kmin = np.zeros(Z.shape)
    Kplus = np.zeros(Z.shape)
    Kminus = np.zeros(Z.shape)
    Si = np.zeros(Z.shape)

    ti = datetime.datetime.now()



    for ci in range(stepout, Colmax - stepout):
        # column loop
        for ri in range(stepout, Rowmax - stepout):
            #range loop
            i = 0
            for xs in range(-stepout,stepout+1,1):  # xs range to cacl surface

                x = xs + ci #x for each sample in patch
                for ys in range(-stepout,stepout+1,1):
                    y = ys + ri    #y for each sample in patch
                    # print(x,y)
                    data[i] = [xs,ys,Z[x, y]]
                    #data[i] = [x, y, Z[x, y]]  # Data for surface with center xs,ys. Z is surface data for patch
                    i += 1

            A=np.vstack([data[:,0]**2,data[:,1]**2,np.prod(data[:,:2], axis=1),data[:,0],data[:,1],np.ones(data.shape[0])]).T
            p, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])
            a,b,c,d,e,f=p

            Kmean[ci, ri] = (a * (1 + e ** 2) - c * d * e + b * (1 + d ** 2)) / (1 + d ** 2 + e ** 2) ** (3 / 2)
            Kgaus[ci, ri] = (4 * a * b - c ** 2) / (1 + d ** 2 + e ** 2) ** 2
            Kmax[ci, ri] = Kmean[ci, ri] + (Kmean[ci, ri] ** 2 - Kgaus[ci, ri]) ** .5
            # Kmin[ci, ri] = Kmean[ci, ri] - (Kmean[ci, ri] ** 2 - Kgaus[ci, ri]) ** .5
            # Kplus[ci, ri] = (a + b) + ((a - b) ** 2 + c ** 2) ** .5
            # Kminus[ci, ri] = (a + b) - ((a - b) ** 2 + c ** 2) ** .5
            # Si[ci, ri] = 2 / np.pi * np.arctan((Kmin[ci, ri] + Kmax[ci, ri]) / (Kmax[ci, ri] - Kmin[ci, ri]))
        print(ci, "out of", Colmax)

    tstop = datetime.datetime.now()
    trun = tstop - ti
    print(trun)

    return Kmax



Kmax=runK(Z,stepout)



plt.figure(figsize=(12,6))

plt.subplot(221)
plt.imshow(Z, cmap='Accent')
plt.ylim(min(plt.ylim()), max(plt.ylim()))

plt.subplot(222)
plt.imshow(Kmax,cmap='afmhot')
plt.ylim(min(plt.ylim()), max(plt.ylim()))
plt.show()


