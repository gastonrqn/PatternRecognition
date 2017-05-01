# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 19:43:46 2017

@author: grequeni
"""

#==============================================================================
# n, dim = 300, 2
# np.random.seed(0)
# sigma_x=np.array([[1.5,0.2],[0.2,3.4]])
# eigenvals, Phi=np.linalg.eig(sigma_x)
# np.diag(eigenvals)
# np.diag(np.sqrt(1/eigenvals))
# 
# C = np.array([[0., -0.23], [0.83, .23]])
# X = np.r_[np.dot(np.random.randn(n, dim), C),
#               np.dot(np.random.randn(n, dim), C) + np.array([1, 1])]
# # y = np.hstack((np.zeros(n), np.ones(n)))
#==============================================================================
# %matplotlib qt
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from scipy.stats import multivariate_normal
from matplotlib import cm

def plot_cov_ellipse(cov, pos, volume=.5, ax=None, fc='none', ec=[0,0,0], a=1, lw=2):
    """
    Plots an ellipse enclosing *volume* based on the specified covariance
    matrix (*cov*) and location (*pos*). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        volume : The volume inside the ellipse; defaults to 0.5
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
    """

    import numpy as np
    from scipy.stats import chi2
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    kwrg = {'facecolor':fc, 'edgecolor':ec, 'alpha':a, 'linewidth':lw}

    # Width and height are "full" widths, not radius
    width, height = 2 * np.sqrt(chi2.ppf(volume,2)) * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwrg)

    ax.add_artist(ellip)

cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)

currFigure = 1

plt.clf()
''' Caso 1
mean_1 = [0, 0]
mean_2 = [2.5, 4.5]

cov_1 = [[2, 0], [0, 2]]
cov_2 = cov_1
#'''

''' Caso 2
mean_1 = [0, 0]
mean_2 = [2.5, 4.5]

sigma_1= np.sqrt(2.0)
sigma_2= 2.0
rho=-0.6
cov_1 = [[sigma_1**2, rho*sigma_1*sigma_2], [rho*sigma_1*sigma_2, sigma_2**2]]

cov_2 = cov_1 #[[1.5, 0.2], [0.2, 2.5]]  # diagonal covariance
#'''

''' Caso 3a
mean_1 = [-2, -4]
mean_2 = [2.5, 4.5]

sigma_1= 3.0
sigma_2= 2.0
rho=0.6
cov_1 = [[sigma_1**2, rho*sigma_1*sigma_2], [rho*sigma_1*sigma_2, sigma_2**2]]

sigma_1= np.sqrt(2.0)
sigma_2= 2.0
rho=-0.6
cov_2 = [[sigma_1**2, rho*sigma_1*sigma_2], [rho*sigma_1*sigma_2, sigma_2**2]]
#'''

''' Caso 3b
mean_1 = [0.5, -5]
mean_2 = [0, 4.5]

sigma_1= 3.0
sigma_2= 2.0
rho=0.2
cov_1 = [[sigma_1**2, rho*sigma_1*sigma_2], [rho*sigma_1*sigma_2, sigma_2**2]]

sigma_1= 1
sigma_2= 0.5
rho=-0.3
cov_2 = [[sigma_1**2, rho*sigma_1*sigma_2], [rho*sigma_1*sigma_2, sigma_2**2]]
#'''

#''' Caso 3c
mean_1 = [0, 0]
mean_2 = [0, 0]

sigma_1= 2.0
sigma_2= 2.0
rho=0
cov_1 = [[sigma_1**2, rho*sigma_1*sigma_2], [rho*sigma_1*sigma_2, sigma_2**2]]

sigma_1= 5.0
sigma_2= 5.0
rho=0
cov_2 = [[sigma_1**2, rho*sigma_1*sigma_2], [rho*sigma_1*sigma_2, sigma_2**2]]
#'''


x_1, y_1 = np.random.multivariate_normal(mean_1, cov_1, 400).T
x_2, y_2 = np.random.multivariate_normal(mean_2, cov_2, 400).T

#plt.pcolormesh(x, y, np.where(rv_1 > rv_2), cmap='red_blue_classes', norm=colors.Normalize(0., 1.))
#np.where(rv_1 > rv_2)

plt.close(currFigure)
plt.figure(currFigure)
# only plot samples inside plot limits
filter_1 = (x_1 > -10) & (x_1 < 10) & (y_1 > -10) & (y_1 < 10)
filter_2 = (x_2 > -10) & (x_2 < 10) & (y_2 > -10) & (y_2 < 10)
plt.plot(x_1[filter_1], y_1[filter_1], 'o', color='blue', alpha=0.4,)
plt.plot(x_2[filter_2], y_2[filter_2], 'o', color='red', alpha=0.4,)
# sin filtros:
#plt.plot(x_1, y_1, 'o', color='blue', alpha=0.4,)
#plt.plot(x_2, y_2, 'o', color='red', alpha=0.4,)
plot_cov_ellipse(cov_1, mean_1)
plot_cov_ellipse(cov_2, mean_2)
plt.axis('equal')
plt.show()

x, y = np.mgrid[-10:10:.05, -10:10:.05]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
rv_1 = multivariate_normal(mean_1, cov_1)
rv_2 = multivariate_normal(mean_2, cov_2)
cont = plt.contourf(x, y, rv_1.pdf(pos)-rv_2.pdf(pos), levels=[-1,0,1], colors=[(1,0,0,0.5), (0,0,1,0.5)])
#plt.colorbar(cont)



'''
currFigure = currFigure + 1
w1x = []
w1y = []
w2x = []
w2y = []
#w_1_x, w_1_y
for i in range(0, 500):
    # conjunto 1
    densidad1 = multivariate_normal.pdf([x_1[i], y_1[i]], mean=mean_1, cov=cov_1)
    densidad2 = multivariate_normal.pdf([x_1[i], y_1[i]], mean=mean_2, cov=cov_2)
    if densidad1 > densidad2:
        w1x.append(x_1[i])
        w1y.append(y_1[i])
    else:
        w2x.append(x_1[i])
        w2y.append(y_1[i])
        
    # conjunto 2
    densidad1 = multivariate_normal.pdf([x_2[i], y_2[i]], mean=mean_1, cov=cov_1)
    densidad2 = multivariate_normal.pdf([x_2[i], y_2[i]], mean=mean_2, cov=cov_2)
    if densidad1 > densidad2:
        w1x.append(x_2[i])
        w1y.append(y_2[i])
    else:
        w2x.append(x_2[i])
        w2y.append(y_2[i])
    
plt.close(currFigure)
plt.figure(currFigure)
plt.plot(w1x, w1y, 'o', color='blue', alpha=0.2,)
plt.plot(w2x, w2y, 'o', color='red', alpha=0.2,)
plt.axis('equal')
plt.show()
'''


zDensities = np.maximum(rv_1.pdf(pos),rv_2.pdf(pos))
colors = np.empty(zDensities.shape, dtype='string')
colors[rv_1.pdf(pos) > rv_2.pdf(pos)] = 'b'
colors[rv_1.pdf(pos) <= rv_2.pdf(pos)] = 'r'

currFigure = currFigure + 1
plt.close(currFigure)
fig = plt.figure(currFigure)
ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(x, y, np.maximum(rv_1.pdf(pos),rv_2.pdf(pos)), cmap = cm.jet, antialiased=True, linewidth=0)
ax.plot_surface(x, y, zDensities, facecolors = colors, antialiased=True, linewidth=0)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()
