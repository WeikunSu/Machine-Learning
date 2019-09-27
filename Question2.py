import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.optimize

## Generate the plot of log-likelihood-ratio function
x = np.linspace(-5, 5, 1000)
l = lambda x: math.log(2) - abs(x) + (1/2)*abs(x - 1)
# x_int = scipy.optimize.fsolve(l, [-2,1])

plt.plot(x,l(x))
# plt.hlines(0,-5, 5, linestyles = '--')
# plt.scatter(x_int, l(x_int))
# plt.vlines(x_int, min(l(x)), 0, linestyles = '--')
plt.ylim(min(l(x)))
plt.xlim(min(x),max(x))
plt.xlabel(r'$x$', fontsize = 20)
plt.ylabel(r'$\ell (x)$', fontsize = 20)
plt.title('Log-likelihood-ratio function', fontsize = 20)
plt.show()

# print('root of the log-likelihood-function is %f and %f.' %(x_int[0],x_int[1]))