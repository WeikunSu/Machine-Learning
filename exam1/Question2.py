##This is the code for quesiton 2
import matplotlib.pyplot as plt 
import numpy as np 
import scipy.stats
from math import *

## MAP decision score fucntion
def decisionScoreMAP(xv_est, yv_est, r):
    global k
    global sigma_x
    global sigma_y
    global sigma_noise
    firstterm = 0
    secondterm = 0
    for i in range(k):
        firstterm += (r[i]-((xv_est - X_ref[0][i])**2 + (yv_est - X_ref[1][i])**2)**(0.5))**2 / sigma_noise**2
    secondterm = (xv_est/sigma_x)**2 + (yv_est/sigma_y)**2
    # firstterm = xv_est**2 + yv_est**2
    return firstterm + secondterm

## Set parameters here
k_max = 4
sigma_noise = 0.3
sigma_x = 0.25
sigma_y = 0.25

## Set true position X_T
X_T = np.zeros((2,1))
theta = 2*pi*np.random.rand(1)
X_T[0][0] = np.random.rand(1)*cos(theta)
X_T[1][0] = np.random.rand(1)*sin(theta)

plt.figure()
for k in range(1,k_max+1):
    ## Set reference posrition(landmark) X_ref
    X_ref = np.zeros((2,k))
    for i in range(k):
        theta_i = i * 2*pi/k
        X_ref[0][i] = cos(theta_i)
        X_ref[1][i] = sin(theta_i)

    ## Get the measurement r
    r = np.zeros(k)
    d = np.zeros(k)
    j = 0
    while j < k:
        d[j] = sqrt((X_T[0][0]-X_ref[0][j])**2 + (X_T[1][0]-X_ref[1][j])**2)
        r[j] = d[j] + np.random.normal(0,sigma_noise**2)
        # Make sure the measurement r[j] is non-negative
        if r[j] >= 0:
            j += 1

    ## MAP estimation
    plt.subplot(2,2,k)
    x_est = np.linspace(-2, 2, 1000)
    y_est = np.linspace(-2, 2, 1000)
    xv_est, yv_est = np.meshgrid(x_est, y_est)
    f = decisionScoreMAP(xv_est, yv_est, r)
    fmin = f.min(); fmax= f.max()
    c = plt.contour(x_est, y_est, f, fmin+int(fmax-fmin)*np.linspace(0,0.4,10)**2)
    plt.scatter(X_ref[0],X_ref[1], marker='o', c='r', label='Referance position')
    plt.scatter(X_T[0],X_T[1], marker='+', c='g', label = 'True position')
    plt.legend(fontsize='15')
    plt.title('MAP estimation for K = %d'%k)

plt.show()

