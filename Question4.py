import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.optimize
import scipy.stats

## Generate the plot of pdf of class-conditional probabilities and posterior probabilities (subquetion 2)
N1 = scipy.stats.norm(0,1)
N2 = scipy.stats.norm(1,math.sqrt(2))
x = np.linspace(-4, 6, 100)

#find the intersection of two curves
h = lambda x: N1.pdf(x) - N2.pdf(x)
x_int = scipy.optimize.fsolve(h, [-2,1])

#plot the class-conditional pdf (subplot_1)
plt.subplot(211)
plt.plot(x, N1.pdf(x), label = r'$p(x|L=1)$')
plt.plot(x, N2.pdf(x), label = r'$p(x|L=2)$')
plt.grid(True)
plt.scatter(x_int, N1.pdf(x_int))
plt.vlines(x_int, 0, 0.5, linestyles = '--')
plt.xlabel(r'$x$', fontsize = 20)
plt.ylabel(r'$p(x|L)$', fontsize = 20)
plt.title('class-conditional probabilities', fontsize = 20)
plt.legend(fontsize = 20)

#plot the pdf of posteriod (subplot_2)
plt.subplot(212)
plt.plot(x, N1.pdf(x)/ (N1.pdf(x) + N2.pdf(x)), label = r'$p(L=1|x)$')
plt.plot(x, N2.pdf(x)/ (N1.pdf(x) + N2.pdf(x)), label = r'$p(L=2|x)$')
plt.scatter(x_int, N1.pdf(x_int)/ (N1.pdf(x_int) + N2.pdf(x_int)))
plt.vlines(x_int, 0, 1, linestyles = '--')
plt.xlabel(r'$x$', fontsize = 20)
plt.ylabel(r'$p(L|x)$', fontsize = 20)
plt.grid(True)
plt.title('posterior probabilities', fontsize = 20)
plt.subplots_adjust(hspace = 0.4)
plt.legend(fontsize = 20)
plt.show()
print('The desicion boundaries are x = %f, and x = %f'%(x_int[0],x_int[1]))

## Calculate the probability of error (subquestion 3)
a = x_int[0]
b = x_int[1]
Pr_error = 0.5 * (N1.cdf(a) + (1-N1.cdf(b)) + (N2.cdf(b) - N2.cdf(a)))
print('The probability of error is %f' %Pr_error)