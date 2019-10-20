##This is the code for quesiton 3
import matplotlib.pyplot as plt 
import numpy as np 
import scipy.optimize
from math import *
# Generate the samples [x; y]
def sampleGenerater(N, sigma_noise, W_true):
    x = 2*np.random.rand(N) - 1   # x is uniform distributed in [-1,1]
    X_mat = np.zeros((4,N))       # X_mat is a matrix [x^3; x^2; x; 1]
    X_mat[0] = x**3         
    X_mat[1] = x**2
    X_mat[2] = x
    X_mat[3] = np.ones(N)
    y = np.dot(W_true, X_mat) + np.random.normal(0, sigma_noise, N)
    return x, y, X_mat

def MAPscore(W_est):
    global N
    global X_mat
    global y
    global i
    global sigma_noise
    firstterm = 0
    for k in range(N):
        firstterm += sigma_noise**(-2) * (y[k] - np.dot(W_est, X_mat[:,k]))**2
    secondterm = i**(-2) * np.dot(W_est, np.transpose(W_est))
    h = firstterm + secondterm
    return h

# Set the parameters 
sigma_noise = 0.01                 # standard variance of noise (sufficiently large)
a = 1                             # parameter of the function y = a(x-r1)(x-r2)(x-r3)
N = 10                            # number of samples to be generated
r = np.array([-0.5, 0, 0.5])      # the roots of the polynomial function
num_experiment = 100              # the number of experiment, setted to 100

# Generate true parameter vector W_true (transfer roots to parameter a, b, c, d)
W_true = np.zeros(4)
W_true[0] = a
W_true[1] = -(r[0] + r[1] + r[2])
W_true[2] = r[0]*r[1] + r[0]*r[2] + r[1]*r[2]
W_true[3] = -(r[0]*r[1]*r[2])

# 
counter = 0
num_sigma_prior = 20                            # the number of sigma_prior we want to evaluate 
err_min = np.zeros(num_sigma_prior)
err_25 = np.zeros(num_sigma_prior)
err_median = np.zeros(num_sigma_prior)
err_75 = np.zeros(num_sigma_prior)
err_max = np.zeros(num_sigma_prior)
sigma_prior = 10**np.linspace(-3, 3, num_sigma_prior)
for i in sigma_prior:                          # set the range of our sigma_prior from 10^(-3) to 10^(3)
    error_L2 = np.zeros(num_experiment)           
    for j in range(num_experiment):      
        x, y, X_mat = sampleGenerater(N, sigma_noise, W_true)
        W_est = scipy.optimize.fmin(MAPscore, np.array([1,0,0,0]), disp=False)
        error_L2[j] = sqrt(np.dot(np.transpose(W_est - W_true), W_est - W_true))
    error_L2_sorted = sorted(error_L2)
    err_min[counter] = error_L2_sorted[0]
    err_25[counter] = error_L2_sorted[int (0.25*num_experiment-1)]
    err_median[counter] = error_L2_sorted[int(0.5*num_experiment-1)]
    err_75[counter] = error_L2_sorted[int(0.75*num_experiment-1)]
    err_max[counter] = error_L2_sorted[num_experiment-1]
    counter += 1
plt.semilogx(sigma_prior, err_min, label = 'minimun value')
plt.semilogx(sigma_prior, err_25, label = '25% value')
plt.semilogx(sigma_prior, err_median, label = 'median value')
plt.semilogx(sigma_prior, err_75, label = '75% value')
plt.semilogx(sigma_prior, err_max, label = 'maximun value')

# plt.loglog(sigma_prior, err_min, label = 'minimun value')
# plt.loglog(sigma_prior, err_25, label = '25% value')
# plt.loglog(sigma_prior, err_median, label = 'median value')
# plt.loglog(sigma_prior, err_75, label = '75% value')
# plt.loglog(sigma_prior, err_max, label = 'maximun value')

plt.legend(fontsize = 20)
plt.xlabel(r'$\gamma$', fontsize = 20)
plt.ylabel(r'${||w_{true}-w_{MAP}||}^2_2$', fontsize = 20)
plt.show()

