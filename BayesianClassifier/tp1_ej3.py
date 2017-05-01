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


def ReadImage(fileName):
    return misc.imread(fileName)


# Parametros
imageFileName = 'circular.jpg'
K = 4 # cantidad de clases C_0 desierto, C_1 circ gris, C_2 circ negro, C_3 circ rojo
D = 3 # dimension de las muestras x_i
classesColors = [[220,220,220], [150,150,150], [0,0,0], [255, 0, 0]]
classesColors01 = [tuple([float(i)/255.0 for i in classesColors[k]]) for k in range(0,K)]
classesMarkers = ['o', '^', 'o', 'o']


''' CrossValidation - a:
classifiedImageFileName = 'ej3-caso1-equi-clasif.png'
trainFilesForClass = [['C1_50x50_01.png'],
                      ['C2_50x50_01.png'],
                      ['C3_50x50_01.png'],
                      ['C4_50x50_01.png']]
                       
testFilesForClass =  [['C1_50x50_02.png','C1_50x50_03.png','C1_50x50_04.png','C1_50x50_05.png','C1_50x50_06.png','C1_50x50_07.png'],
                      ['C2_50x50_02.png','C2_50x50_03.png','C2_50x50_04.png','C2_50x50_05.png','C2_50x50_06.png','C2_50x50_07.png'],
                      ['C3_50x50_02.png','C3_50x50_03.png','C3_50x50_04.png','C3_50x50_05.png','C3_50x50_06.png','C3_50x50_07.png'],
                      ['C4_50x50_02.png','C4_50x50_03.png','C4_50x50_04.png','C4_50x50_05.png','C4_50x50_06.png','C4_50x50_07.png']]
# '''
                      
'''# CrossValidation - b:
classifiedImageFileName = 'ej3-caso2-equi-clasif.png'
trainFilesForClass = [['C1_50x50_01.png','C1_50x50_03.png'],
                      ['C2_50x50_01.png','C2_50x50_03.png'],
                      ['C3_50x50_01.png','C3_50x50_02.png'],
                      ['C4_50x50_01.png','C4_50x50_02.png']]
                       
testFilesForClass =  [['C1_50x50_02.png','C1_50x50_04.png','C1_50x50_05.png','C1_50x50_06.png','C1_50x50_07.png'],
                      ['C2_50x50_02.png','C2_50x50_04.png','C2_50x50_05.png','C2_50x50_06.png','C2_50x50_07.png'],
                      ['C3_50x50_03.png','C3_50x50_04.png','C3_50x50_05.png','C3_50x50_06.png','C3_50x50_07.png'],
                      ['C4_50x50_03.png','C4_50x50_04.png','C4_50x50_05.png','C4_50x50_06.png','C4_50x50_07.png']]
# '''


'''# CrossValidation - c:
classifiedImageFileName = 'ej3-caso3-equi-clasif.png'
trainFilesForClass = [['C1_50x50_01.png','C1_50x50_03.png','C1_50x50_02.png'],
                      ['C2_50x50_01.png','C2_50x50_03.png','C2_50x50_05.png'],
                      ['C3_50x50_01.png','C3_50x50_02.png','C3_50x50_04.png'],
                      ['C4_50x50_01.png','C4_50x50_02.png','C4_50x50_05.png']]
                       
testFilesForClass =  [['C1_50x50_04.png','C1_50x50_05.png','C1_50x50_06.png','C1_50x50_07.png'],
                      ['C2_50x50_02.png','C2_50x50_04.png','C2_50x50_06.png','C2_50x50_07.png'],
                      ['C3_50x50_03.png','C3_50x50_05.png','C3_50x50_06.png','C3_50x50_07.png'],
                      ['C4_50x50_03.png','C4_50x50_04.png','C4_50x50_06.png','C4_50x50_07.png']]
# '''
                      
# CrossValidation - d:
classifiedImageFileName = 'ej3-caso4-equi-clasif.png'
trainFilesForClass = [['C1_50x50_01.png','C1_50x50_02.png','C1_50x50_03.png','C1_50x50_04.png','C1_50x50_05.png','C1_50x50_06.png','C1_50x50_07.png'],
                      ['C2_50x50_01.png','C2_50x50_02.png','C2_50x50_03.png','C2_50x50_04.png','C2_50x50_05.png','C2_50x50_06.png','C2_50x50_07.png'],
                      ['C3_50x50_01.png','C3_50x50_02.png','C3_50x50_03.png','C3_50x50_04.png','C3_50x50_05.png','C3_50x50_06.png','C3_50x50_07.png'],
                      ['C4_50x50_01.png','C4_50x50_02.png','C4_50x50_03.png','C4_50x50_04.png','C4_50x50_05.png','C4_50x50_06.png','C4_50x50_07.png']]
                       
testFilesForClass =  trainFilesForClass
# '''


appendFolderName = np.vectorize(lambda s: '' if s=='' else 'circular_classes_examples/50x50/' + s)
trainFilesForClass = appendFolderName(trainFilesForClass)
testFilesForClass = appendFolderName(testFilesForClass)

# Equiprobables a priori
aPrioriProb = [1.0/K, 1.0/K, 1.0 / K, 1.0 / K]

'''# Probabilidades a priori a ojo (hay 41 círculos aprox, 12 gris claro, 12 oscuros, 17 rojos)
probDesierto = 0.4
probCirculo = 1 - probDesierto
aPrioriProb = [probDesierto,
               probCirculo * (12.0/41.0),
               probCirculo * (12.0/41.0),
               probCirculo * (17.0/41.0)]
#'''


# Estimar parámetros de las Normales de cada clase p(x|C_k)~N(mu[k], sigma[k])
trainSamplesForClass = []
mu = []
sigma = []
randomVar = []
for k in range(0, K):    
    # Obtener x_i's de la clase C_k (las guardo como columnas de esta matriz)
    trainSamplesForClass.append(np.empty((D,0), dtype=float))
    for fileName in trainFilesForClass[k]:
        if fileName == '':
            continue
        trainRegion = ReadImage(fileName)
        (height, width, pixelChannels) = trainRegion.shape
        featureVectorsInColumns = np.reshape(trainRegion, (height*width,pixelChannels), 'C').T
        # Si tienen canal alpha, se elimina
        if pixelChannels > D:
            featureVectorsInColumns = featureVectorsInColumns[0:D, :]
        
        trainSamplesForClass[k] = np.append(trainSamplesForClass[k], featureVectorsInColumns, axis = 1)
    
    # Calcular mu_k
    mu.append(np.average(trainSamplesForClass[k], axis=1))
    
    # Calcular Sigma_k
    N_k = trainSamplesForClass[k].shape[1] # cantidad de muestras
    M = trainSamplesForClass[k] - np.dot(np.reshape(mu[k], (D, 1)), np.ones((1, N_k))) # en cada columna tengo la muestra menos la media
    sigma.append(np.dot(M, M.T)*(1.0/(N_k-1)))
    
    # Random Variable
    randomVar.append(multivariate_normal(mu[k], sigma[k]))
    

# Plot de los feature vectors de entrenamiento (un sampleado, 
# porque si son muchos se cuelga el display)
def SampleRandomIndexes(vectorLength, n):
    return [int(x) for x in np.floor(vectorLength * np.random.sample(n))]
    
def SampleRandomColumns(matrix2d, n):
    cols = matrix2d.shape[1]
    return matrix2d[:,SampleRandomIndexes(cols,n)]

samplesPerClass = 50
plotSampling = [SampleRandomColumns(trainSamplesForClass[k],samplesPerClass) for k in range(0,K)]
plt.close(1)
fig = plt.figure(1)
plt.title("Nubes de puntos de entrenamiento")
ax = fig.add_subplot(111, projection='3d')
scatterProxyForLegend = []
for k in range(0, K):
    ax.scatter(plotSampling[k][0,:],
               plotSampling[k][1,:],
               plotSampling[k][2,:],
               c=classesColors01[k],
               marker=classesMarkers[k])
    scatterProxyForLegend.append(matplotlib.lines.Line2D([0],[0], linestyle="none", c=classesColors01[k], marker = classesMarkers[k]))

plt.axis('equal')
ax.set_title("Nubes de puntos de entrenamiento")
ax.set_xlabel('Rojo')
ax.set_ylabel('Verde')
ax.set_zlabel('Azul')
ax.legend(scatterProxyForLegend, ['$C_1$', '$C_2$', '$C_3$', '$C_4$'], numpoints = 1)
plt.show()


def BayesianClasifier(imgRGB):
    (height, width, pixelChannels) = imgRGB.shape
    if pixelChannels > D:
        imgRGB = imgRGB[:,:,0:D]
    
    discriminant = np.empty((height, width, K), dtype=float)
    for k in range(0, K):
        discriminant[:,:,k] = randomVar[k].pdf(imgRGB) * aPrioriProb[k]
    
    classifiedPixels = np.argmax(discriminant, axis = 2)
    return classifiedPixels
    
def ClassifiedImage(classifiedPixels):
    (height, width) = classifiedPixels.shape
    classIm = np.empty((height, width, D), dtype=uint8);
    for k in range(0, K):
        classIm[classifiedPixels == k] = classesColors[k]
    return classIm

# Clasificar imagen padre
mainImage = ReadImage(imageFileName)
classifiedPixels = BayesianClasifier(mainImage)
classIm = ClassifiedImage(classifiedPixels)

plt.close(2)
plt.figure(2)
plt.title("Imagen Clasificada")
plt.imshow(classIm)
plt.imsave(arr=classIm, fname='informe/img/'+classifiedImageFileName)

# Clasificar imagenes de entrenamiento y calcular confusion
confusionTrain = np.zeros((K,K), dtype = int)
for realClass in range(0, K):
    for fileName in trainFilesForClass[realClass]:
        if fileName == '':
            continue
        im = ReadImage(fileName)
        clfPixels = BayesianClasifier(im)
        for guessClass in range(0, K):
            confusionTrain[guessClass, realClass] += np.count_nonzero(clfPixels == guessClass)
            

aciertosPorClase = (np.diag(confusionTrain) / np.sum(confusionTrain, axis=0, dtype=float)) * 100.0
aciertoTotal = (np.sum(np.diag(confusionTrain), dtype=float) / np.sum(confusionTrain, dtype=float)) * 100.0


def PrintConfusionForLatex(confusion, aciertosPorClase, aciertoTotal):
    print "Clasificado $C_1$ & " + str(confusion[0,0]) + " & " + str(confusion[0,1]) + " & " + str(confusion[0,2]) + " & " + str(confusion[0,3]) + " \\\\"
    print "\hline"
    print "Clasificado $C_2$ & " + str(confusion[1,0]) + " & " + str(confusion[1,1]) + " & " + str(confusion[1,2]) + " & " + str(confusion[1,3]) + " \\\\"
    print "\hline"
    print "Clasificado $C_3$ & " + str(confusion[2,0]) + " & " + str(confusion[2,1]) + " & " + str(confusion[2,2]) + " & " + str(confusion[2,3]) + " \\\\"
    print "\hline"
    print "Clasificado $C_4$ & " + str(confusion[3,0]) + " & " + str(confusion[3,1]) + " & " + str(confusion[3,2]) + " & " + str(confusion[3,3]) + " \\\\"
    print "\hline"
    print "\% Acierto & " + str("%.2f" % aciertosPorClase[0]) + "\% & " + str("%.2f" % aciertosPorClase[1]) + "\% & " + str("%.2f" % aciertosPorClase[2]) + "\% & " + str("%.2f" % aciertosPorClase[3]) + "\% \\\\"
    print "\hline"
    print "\% Acierto Total & \multicolumn{4}{ c |}{" + str("%.2f" % aciertoTotal) + "\%} \\\\"
    print "\hline"
    print ""

print "Matriz de Confusion de Entrenamiento"  
print confusionTrain
print "Para LATEX: \n"  

PrintConfusionForLatex(confusionTrain, aciertosPorClase, aciertoTotal)


# Clasificar imagenes de test y calcular confusion
confusionTest = np.zeros((K,K), dtype = int)
for realClass in range(0, K):
    for fileName in testFilesForClass[realClass]:
        if fileName == '':
            continue
        im = ReadImage(fileName)
        clfPixels = BayesianClasifier(im)
        for guessClass in range(0, K):
            confusionTest[guessClass, realClass] += np.count_nonzero(clfPixels == guessClass)
            

aciertosPorClase = (np.diag(confusionTest) / np.sum(confusionTest, axis=0, dtype=float)) * 100.0
aciertoTotal = (np.sum(np.diag(confusionTest), dtype=float) / np.sum(confusionTest, dtype=float)) * 100.0

print "Matriz de Confusion de Test"  
print confusionTest
print "Para LATEX: \n"  

PrintConfusionForLatex(confusionTest, aciertosPorClase, aciertoTotal)










'''    
filesForClass = [['C1_50x50_01.png','C1_50x50_02.png','C1_50x50_03.png','C1_50x50_04.png','C1_50x50_05.png','C1_50x50_06.png','C1_50x50_07.png'],
                 ['C2_50x50_01.png','C2_50x50_02.png','C2_50x50_03.png','C2_50x50_04.png','C2_50x50_05.png','C2_50x50_06.png','C2_50x50_07.png'],
                 ['C3_50x50_01.png','C3_50x50_02.png','C3_50x50_03.png','C3_50x50_04.png','C3_50x50_05.png','C3_50x50_06.png','C3_50x50_07.png'],
                 ['C4_50x50_01.png','C4_50x50_02.png','C4_50x50_03.png','C4_50x50_04.png','C4_50x50_05.png','C4_50x50_06.png','C4_50x50_07.png']]

trainImagesPerClass = 1



# CrossValidation
def AddToAllElements(e, xxs):
    for x in xxs:
        x.append(e)
    return xxs

def AllSubsetsWithNElements(l, N):
    result = []
    if (N == 0) | (N > len(l)):
        result.append([])
        return result
    else:
        for i in range(0, len(l)):
            subsetsToAdd = AddToAllElements(l[i], AllSubsetsWithNElements(l[i+1:len(l)], N-1))
            for subset in subsetsToAdd:
                result.append(subset)
    return result


# Leer phantom
phantom = ReadImage(imageFileName)
(height, width, pixelChannels) = phantom.shape
if pixelChannels != 3:
    raise "Error, sólo soporta phantom RGB"

# Matriz de clases por cada pixel (grount truth)
gt = np.empty((height, width), dtype=int);
gt[:,:] = -1
for k in range(0, K):
    mask = (phantom[:,:,0] == classesColors[k][0]) & (phantom[:,:,1] == classesColors[k][1]) & (phantom[:,:,2] == classesColors[k][2])
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
    plt.plot(sintIm[gt == k, 0], sintIm[gt == k, 1], 'o', color=tuple([float(i)/255.0 for i in classesColors[k]]) , alpha=0.4)
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
    classIm[classifiedPixels == k] = classesColors[k]

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
'''