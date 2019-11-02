#This is the code for Question 1

import matplotlib.pyplot as plt 
import numpy as np 
import math
import scipy.stats


def GMMgen(N, alpha, mu, Sigma):
    thr = [0]
    thr.extend(np.cumsum(alpha))
    u = np.random.rand(N)
    L = np.zeros((N))
    x = np.zeros((N, 2))
    for i in range(len(alpha)):
        indices = np.where(np.logical_and(u >= thr[i], u < thr[i+1]) == True)
        L[indices] = i * np.ones(len(indices))
        for k in indices[0]:
            x[k,:] = np.random.multivariate_normal(mu[i,:], Sigma[:,:,i])
    return x, L

def EMforGMM(X, M):
    delta = 0.00001
    Sigma = np.zeros([2,2,M])
    n_sample = len(X)
    mu = X[0:M] + 0.1*np.ones([M,2])
    for i in range(M):
        Sigma[:,:,i] = 0.2*(i+1)*np.eye(2)
    W = np.ones((n_sample, M))/M
    alpha = W.sum(axis=0)/W.sum()
    converged = 0
    counter = 0
    while not converged:
        temp = np.zeros([n_sample,M])
        for i in range(M):
            temp[:,i] = alpha[i]*scipy.stats.multivariate_normal.pdf(X, mu[i],Sigma[:,:,i])
        W_new = temp/temp.sum(axis=1).reshape(-1,1)
        alpha_new = W_new.sum(axis=0)/W_new.sum()
        w = W_new / W_new.sum(axis=0)
        mu_new = np.transpose(np.dot(np.transpose(X), w))
        Sigma_new = np.zeros([2,2,M])
        for i in range(M):
            # for j in range(n_sample):
            #     Sigma_new[:,:,i] += W_new[j,i] * (np.reshape(X[j] - mu_new[i],[2,1]) * (X[j] - mu_new[i]))/W_new[i].sum(axis=0)
            # Sigma_new[:,:,i] += 0.0001*np.eye(2)
            v = X - mu_new[i]
            u = np.reshape(w[:,i],[-1,1]) * v
            Sigma_new[:,:,i] = np.dot(np.transpose(u), v) + 0.0001*np.eye(2)
        D_alpha = np.sum(abs(alpha_new - alpha))
        D_mu = np.sum(abs(mu_new - mu))
        D_Sigma = np.sum(abs(Sigma_new - Sigma))
        converged = (D_Sigma + D_alpha + D_mu) < delta
        alpha = alpha_new
        mu = mu_new
        Sigma = Sigma_new
        counter += 1
    return alpha, mu, Sigma

def logLH(X, alpha, mu, Sigma):
    n_sample = len(X)
    M = len(alpha)
    pdfx = np.zeros([n_sample,M])
    for i in range(M):
        pdfx[:,i] = alpha[i] * scipy.stats.multivariate_normal.pdf(X, mu[i], Sigma[:,:,i])
    LLH = sum(np.log(pdfx.sum(axis=1)))
    return LLH

        


if __name__ == '__main__':
    ## Set up the true GMM model
    mu_true = np.array([[1,1],[-1,1],[-1,-1],[1,-1]])
    Sigma_true = np.zeros([2,2,4])
    Sigma_true[:,:,0] = 0.1*np.array([[1,0],[0,1]])
    Sigma_true[:,:,1] = 0.1*np.array([[1,0.7],[0.7,1]])
    Sigma_true[:,:,2] = 0.1*np.array([[3,0],[0,1]])
    Sigma_true[:,:,3] = 0.1*np.array([[1,0],[0,3]])
    alpha_true = np.array([0.1,0.2,0.3,0.4])

    K = 10                             # K-fold c.v.
    M_max = 6
    counter = 1
    for N in [10,100,1000,10000]:
        
        X, L = GMMgen(N, alpha_true, mu_true, Sigma_true)
        # plt.scatter(X[:,0], X[:,1], marker='.'), plt.show()
        dummy = np.ceil(np.linspace(0,N,K+1))
        index_partition_limit = np.zeros([K,2])
        for k in range(K):
            index_partition_limit[k,:] = [dummy[k],dummy[k+1]-1]
        #print(index_partition_limit)
        M = np.linspace(1,M_max,M_max)
        logLH_est = np.zeros(M_max)
        for i in range(M_max):
            logLH_validate = np.zeros(K)
            for k in range(K):
                ind_validate = np.arange(index_partition_limit[k,0], index_partition_limit[k,1]+1)
                ind_validate = ind_validate.astype(int)
                X_validate = X[ind_validate,:]
                ind_train = np.hstack((np.arange(0,index_partition_limit[k,0]),np.arange(index_partition_limit[k,1]+1, N)))
                ind_train = ind_train.astype(int)
                X_train = X[ind_train,:]
                alpha, mu, Sigma = EMforGMM(X_train, i+1)
                logLH_validate[k] = logLH(X_validate, alpha, mu, Sigma)
            logLH_est[i] = np.max(logLH_validate)
        plt.subplot(2,2,counter)
        plt.plot(M,logLH_est, label='N=%d'%N)
        plt.xlabel('Order', fontsize=20)
        plt.ylabel('Log Likelihood', fontsize=20)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.title('N=%d'%N, fontsize =20)
        plt.subplots_adjust(hspace=0.4)
        counter += 1
    plt.show()