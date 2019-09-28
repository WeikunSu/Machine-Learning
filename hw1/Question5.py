import numpy as np

#random vector generator
def normalRandomVecGenerator(N, n, mu, Sigma): 
    A = np.linalg.cholesky(Sigma)
    b = mu
    x = np.empty([n, N])
    for i in range(N):
        z = np.random.normal(0, 1, n)
        x[:,i] = np.dot(A,np.transpose(z)) + np.transpose(b)   
    return x

###########
#change the parameters of random vectors here, (N, n, mu, Sigma)
#N is the number of samples(random vectors) you want to generate
#n is the dimention of the random vectors 
#mu and Sigma are the mean and covariance matrix of the Gaussian distribution you want
x = normalRandomVecGenerator(6,3,[0,-40,4], [[2,2,0],[2,3,1],[0,1,1.5]])
###########
print(x)
print(r'Each column of the matrix is a random vector x_i')

