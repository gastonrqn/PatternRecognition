# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 19:43:46 2017

@author: grequeni
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from scipy.stats import multivariate_normal
from matplotlib import cm
from scipy import misc


def ReadPhantom(fileName):
    return misc.imread(fileName)


# Parametros
phantomFileName = 'phantomCustom.bmp'
K = 3 # cantidad de clases
D = 2 # dimension de las muestras x_i
phantomColors = [[255,0,0], [0,255,0], [0,0,255]]

'''# caso 1a
mu_1 = [1,1]
mu_2 = [1.5, 1.5]
mu_3 = [2,2]
sigma_1 = [[3, 0], [0, 3]]
sigma_2 = [[3, 0], [0, 3]]
sigma_3 = [[3, 0], [0, 3]]
#'''


'''# caso 1b
mu_1 = [0,0]
mu_2 = [3, 3]
mu_3 = [6,6]
sigma_1 = [[3, 0], [0, 3]]
sigma_2 = [[3, 0], [0, 3]]
sigma_3 = [[3, 0], [0, 3]]
#'''

'''# caso 2a
mu_1 = [1,1]
mu_2 = [1.5, 1.5]
mu_3 = [2,2]
sigma_1 = [[10, 0], [0, 10]]
sigma_2 = [[3, 0], [0, 3]]
sigma_3 = [[1, 0], [0, 1]]
#'''


'''# caso 2b
mu_1 = [0,0]
mu_2 = [3, 3]
mu_3 = [6,6]
sigma_1 = [[10, 0], [0, 10]]
sigma_2 = [[3, 0], [0, 3]]
sigma_3 = [[1, 0], [0, 1]]
#'''

'''# caso 3a
mu_1 = [0,0]
paramsS = [50, 50, 0.4]
sigma_1 = [[paramsS[0]**2, np.prod(paramsS)], [np.prod(paramsS),paramsS[1]**2]]

mu_2 = [0,0]
paramsS = [10, 10, 0.2] # params for sigma (sigma_1, sigma_2, rho)
sigma_2 = [[paramsS[0]**2, np.prod(paramsS)], [np.prod(paramsS),paramsS[1]**2]]

mu_3 = [0,0]
paramsS = [2, 2, -0.7]
sigma_3 = [[paramsS[0]**2, np.prod(paramsS)], [np.prod(paramsS),paramsS[1]**2]]
#'''

'''# caso 3b
mu_1 = [0,0]
paramsS = [50, 50, 0.4]
sigma_1 = [[paramsS[0]**2, np.prod(paramsS)], [np.prod(paramsS),paramsS[1]**2]]

mu_2 = [-50,0]
paramsS = [10, 10, 0.2] # params for sigma (sigma_1, sigma_2, rho)
sigma_2 = [[paramsS[0]**2, np.prod(paramsS)], [np.prod(paramsS),paramsS[1]**2]]

mu_3 = [40,40]
paramsS = [2, 2, -0.7]
sigma_3 = [[paramsS[0]**2, np.prod(paramsS)], [np.prod(paramsS),paramsS[1]**2]]
#'''


# caso 3c
mu_1 = [-25,0]
paramsS = [3, 8, 0.2]
sigma_1 = [[paramsS[0]**2, np.prod(paramsS)], [np.prod(paramsS),paramsS[1]**2]]

mu_2 = [0,4]
paramsS = [8, 3, 0.9]
sigma_2 = [[paramsS[0]**2, np.prod(paramsS)], [np.prod(paramsS),paramsS[1]**2]]

mu_3 = [0, -10]
paramsS = [3, 2, -0.4] # params for sigma (sigma_1, sigma_2, rho)
sigma_3 = [[paramsS[0]**2, np.prod(paramsS)], [np.prod(paramsS),paramsS[1]**2]]
#'''


mu = [mu_1, mu_2, mu_3]
sigma = [sigma_1, sigma_2, sigma_3]
    

# Leer phantom
phantom = ReadPhantom(phantomFileName)
(height, width, pixelChannels) = phantom.shape
if pixelChannels != 3:
    raise "Error, sólo soporta phantom RGB"

# Matriz de clases por cada pixel (grount truth)
gt = np.empty((height, width), dtype=int);
gt[:,:] = -1
for k in range(0, K):
    mask = (phantom[:,:,0] == phantomColors[k][0]) & (phantom[:,:,1] == phantomColors[k][1]) & (phantom[:,:,2] == phantomColors[k][2])
    gt[mask] = k

if np.any(gt < 0):
    raise "Error: Hay pixels no clasificados."


# Generar muestras aleatorias
xx = []
xx_pointer = []
for k in range(0, K):
    xx.append(np.random.multivariate_normal(mu[k], sigma[k], np.count_nonzero(gt == k)).T)
    xx_pointer.append(0) # me sirve despues para recorrer cada tira de muestras
    
# Generar imagen sintética
sintIm = np.empty((height, width, D), dtype=float);
for i in range(0, height):
    for j in range(0, width):
        k = gt[i,j] # clase k
        nextSample = xx_pointer[k]
        for d in range(0, D): # en el (i,j), tengo x de dimension D
            sintIm[i,j,d] = xx[k][d][nextSample]
        xx_pointer[k] = xx_pointer[k] + 1

# Plot de los puntos sinteticos
plt.close(1)
plt.figure(1)
plt.title("Nubes de puntos de la imagen sintetica")
for k in range(0, K):
    plt.plot(sintIm[gt == k, 0], sintIm[gt == k, 1], 'o', color=tuple([float(i)/255.0 for i in phantomColors[k]]) , alpha=0.4)
plt.axis('equal')
plt.show()

# Clasificación
aPrioriProb = []
randomVar = []
discriminant = np.empty((height, width, K), dtype=float)
for k in range(0, K):
    aPrioriProb.append(1.0/K) #float(np.count_nonzero(gt == k)) / float(height*width))
    print "P(C_" + str(k) + ") = " + str(aPrioriProb[k]) + "\n"
    randomVar.append(multivariate_normal(mu[k], sigma[k]))
    discriminant[:,:,k] = randomVar[k].pdf(sintIm) * aPrioriProb[k]

classifiedPixels = np.argmax(discriminant, axis = 2)

classIm = np.empty((height, width, 3), dtype=uint8);
for k in range(0, K):
    classIm[classifiedPixels == k] = phantomColors[k]

plt.close(2)
plt.figure(2)
plt.title("Imagen Clasificada")
plt.imshow(classIm)


# Matriz de confusión
confusion = np.empty((K,K), dtype = int)
for realClass in range(0, K):
    for guessClass in range(0, K):
        confusion[guessClass, realClass] = np.count_nonzero((gt == realClass) & (classifiedPixels == guessClass))

aciertosPorClase = (np.diag(confusion) / np.sum(confusion, axis=0, dtype=float)) * 100.0
aciertoTotal = (np.sum(np.diag(confusion), dtype=float) / np.sum(confusion, dtype=float)) * 100.0

print "Matriz de Confusion"  
print confusion

print "Version LaTeX:\n"

print "Clasificado {\color{red}$C_1$} & " + str(confusion[0,0]) + " & " + str(confusion[0,1]) + " & " + str(confusion[0,2]) + " \\\\"
print "\hline"
print "Clasificado {\color{green}$C_2$} & " + str(confusion[1,0]) + " & " + str(confusion[1,1]) + " & " + str(confusion[1,2]) + " \\\\"
print "\hline"
print "Clasificado {\color{blue}$C_3$} & " + str(confusion[2,0]) + " & " + str(confusion[2,1]) + " & " + str(confusion[2,2]) + " \\\\"
print "\hline"
print "\% Acierto & " + str("%.2f" % aciertosPorClase[0]) + "\% & " + str("%.2f" % aciertosPorClase[1]) + "\% & " + str("%.2f" % aciertosPorClase[2]) + "\% \\\\"
print "\hline"
print "\% Acierto Total & \multicolumn{3}{ c |}{" + str("%.2f" % aciertoTotal) + "\%} \\\\"
print "\hline"