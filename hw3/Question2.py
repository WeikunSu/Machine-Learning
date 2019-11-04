#This is the code for Question 1

import matplotlib.pyplot as plt 
import numpy as np
import math
import scipy.stats
import scipy.optimize

def GMMgen(N, alpha, mu, Sigma):
    thr = [0]
    thr.extend(np.cumsum(alpha))
    u = np.random.rand(N)
    L = np.zeros((N))
    x = np.zeros((N, 2))
    for i in range(len(alpha)):
        indices = np.where(np.logical_and(u >= thr[i], u < thr[i+1]) == False)
        L[indices] = i * np.ones(len(indices))
        for k in indices[0]:
            x[k,:] = np.random.multivariate_normal(mu[i,:], Sigma[:,:,i])
    return x, L

def fisherLDA(X,L):
    X_p = X[np.where(L == 1),:]
    X_n = X[np.where(L == 0),:]
    muhat_p = np.mean(X_p, axis=1)
    muhat_n = np.mean(X_n, axis=1)
    Sigmahat_p = np.cov(np.transpose(X_p[0]))
    Sigmahat_n = np.cov(np.transpose(X_n[0]))
    S_b = (muhat_p - muhat_n) * np.transpose(muhat_p - muhat_n)
    S_w = Sigmahat_p + Sigmahat_n
    D, V = np.linalg.eig(np.dot(np.linalg.inv(S_w),S_b))
    w = V[:, np.argmax(D)]
    Y = np.dot(w,np.transpose(X))
    w = (np.sign(np.mean(Y[np.where(L == 1)]) - np.mean(Y[np.where(L == 0)])))  * w
    Y = (np.sign(np.mean(Y[np.where(L == 1)]) - np.mean(Y[np.where(L == 0)])))  * Y
    mu_yp = np.mean(Y[np.where(L == 1)])
    mu_yn = np.mean(Y[np.where(L == 0)])
    std_yp = np.std(Y[np.where(L == 1)])
    std_yn = np.std(Y[np.where(L == 0)])
    Pr_p = len(X_p) / (len(X_p)+len(X_n))
    Pr_n = len(X_n) / (len(X_p)+len(X_n))
    # b = (2 * std_yn**2 * std_yp**2 * (np.log(Pr_p*std_yn) - np.log(Pr_n*std_yp)) + mu_yn*std_yp**2 + mu_yp*std_yn**2) / (std_yp**2 - std_yn**2)
    # D_ind = np.where(Y+b >= 0)
    func = lambda y: (std_yn*Pr_p/std_yp*Pr_n)*math.e**(-((y-mu_yp)**2/(2*std_yp**2))+((y-mu_yn)**2/(2*std_yn**2)))-1
    b = -scipy.optimize.fsolve(func, 0)
    D_ind = np.where(Y+b >= 0)
    D = np.zeros(len(L))
    D[D_ind] = 1 * np.ones(len(D_ind))
    return Y, D, w, b


def logisticModelEstimator(X, L, w0, b0):
    X_p = X[np.where(L == 1),:]
    X_n = X[np.where(L == 0),:]
    np.reshape(w0,[1,2])
    theta0 = np.hstack((w0,b0))
    Lfunc = lambda theta: sum(np.log(1+math.e**(np.dot(theta[:2],np.transpose(X_p[0]))+theta[2]))) - sum(np.log(1-1/(1+math.e**(np.dot(theta[:2],np.transpose(X_n[0]))+theta[2]))))
    argmin = scipy.optimize.minimize(Lfunc, theta0, method='Nelder-Mead',options={'disp':False})
    w = argmin.x[:2]
    b = argmin.x[2]
    y = 1/(1+math.e**(np.dot(w,np.transpose(X))+b))
    D_indices = np.where(y >= 0.5)
    D = np.zeros(len(L))
    D[D_indices] = len(D_indices)
    return w, b, D
    
def MAPestimate(X, mu, Sigma, Pr):
    rv_0 = scipy.stats.multivariate_normal(np.reshape(mu[0],(2)), Sigma[:,:,0])
    rv_1 = scipy.stats.multivariate_normal(np.reshape(mu[1],(2)), Sigma[:,:,1])
    D = np.zeros(len(X))
    for i in range(len(X)):
        postiori_0 = rv_0.pdf(X[i,:])*Pr[0]
        postiori_1 = rv_1.pdf(X[i,:])*Pr[1]
        if postiori_0 >= postiori_1 :
            D[i] = 0
        else:
            D[i] = 1
    return D



if __name__ == '__main__':
    mu_true = np.array([[1,1],[-1,-1]])
    Sigma_true = np.zeros([2,2,2])
    Sigma_true[:,:,0] = 0.4*np.array([[2,0.7],[0.7,1]])
    Sigma_true[:,:,1] = 0.4*np.array([[1,-0.4],[-0.4,2]])
    N = 999
    prior = np.array([0.3,0.7])

    #Generate samples
    X,L = GMMgen(N, prior, mu_true, Sigma_true)
    plt.figure()
    plt.scatter(X[np.where(L == 0),0],X[np.where(L == 0),1], marker='x', label='Class -')
    plt.scatter(X[np.where(L == 1),0],X[np.where(L == 1),1], marker='+', label='Class +')
    plt.legend(fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Samples with true label', fontsize=20)

    #fisherLDA 
    Y, D_fisherLDA, w_LDA, b_LDA = fisherLDA(X, L)
    errors_fisherLDA = np.count_nonzero(D_fisherLDA != L)
    plt.figure()
    plt.scatter(X[np.where(np.logical_and(D_fisherLDA == 0,L == 0)),0],X[np.where(np.logical_and(D_fisherLDA == 0,L == 0)),1], c='r', marker='.', label='D = -, L = -')
    plt.scatter(X[np.where(np.logical_and(D_fisherLDA == 1,L == 0)),0],X[np.where(np.logical_and(D_fisherLDA == 1,L == 0)),1], c='r', marker='+', label='D = +, L = -')
    plt.scatter(X[np.where(np.logical_and(D_fisherLDA == 0,L == 1)),0],X[np.where(np.logical_and(D_fisherLDA == 0,L == 1)),1], c='b', marker='.', label='D = -, L = +')
    plt.scatter(X[np.where(np.logical_and(D_fisherLDA == 1,L == 1)),0],X[np.where(np.logical_and(D_fisherLDA == 1,L == 1)),1], c='b', marker='+', label='D = +, L = +')
    plt.legend(fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('FisherLDA estimate \n (w = [ %f , %f ] b = %f \n error counts =%d )'%(w_LDA[0],w_LDA[1],b_LDA, errors_fisherLDA), fontsize=20)
    

    #logistic function estimate
    w, b, D_logistic = logisticModelEstimator(X,L,w_LDA, b_LDA)
    errors_logistice = np.count_nonzero(D_logistic != L)
    plt.figure()
    plt.scatter(X[np.where(np.logical_and(D_logistic == 0,L == 0)),0],X[np.where(np.logical_and(D_logistic == 0,L == 0)),1], c='r', marker='.', label='D = -, L = -')
    plt.scatter(X[np.where(np.logical_and(D_logistic == 1,L == 0)),0],X[np.where(np.logical_and(D_logistic == 1,L == 0)),1], c='r', marker='+', label='D = +, L = -')
    plt.scatter(X[np.where(np.logical_and(D_logistic == 0,L == 1)),0],X[np.where(np.logical_and(D_logistic == 0,L == 1)),1], c='b', marker='.', label='D = -, L = +')
    plt.scatter(X[np.where(np.logical_and(D_logistic == 1,L == 1)),0],X[np.where(np.logical_and(D_logistic == 1,L == 1)),1], c='b', marker='+', label='D = +, L = +')
    plt.legend(fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Logistic estimate \n (w=[ %f , %f ] , b=%f) \n error counts =%d'%(w[0],w[1],b,errors_logistice), fontsize=20)

    #MAPclassifier
    D_MAP = MAPestimate(X, mu_true, Sigma_true, prior)
    errors_MAP = np.count_nonzero(D_MAP != L)
    plt.figure()
    plt.scatter(X[np.where(np.logical_and(D_MAP == 0,L == 0)),0],X[np.where(np.logical_and(D_MAP == 0,L == 0)),1], c='r', marker='.', label='D = -, L = -')
    plt.scatter(X[np.where(np.logical_and(D_MAP == 1,L == 0)),0],X[np.where(np.logical_and(D_MAP == 1,L == 0)),1], c='r', marker='+', label='D = +, L = -')
    plt.scatter(X[np.where(np.logical_and(D_MAP == 0,L == 1)),0],X[np.where(np.logical_and(D_MAP == 0,L == 1)),1], c='b', marker='.', label='D = -, L = +')
    plt.scatter(X[np.where(np.logical_and(D_MAP == 1,L == 1)),0],X[np.where(np.logical_and(D_MAP == 1,L == 1)),1], c='b', marker='+', label='D = +, L = +')
    plt.legend(fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('MAP estimate \n error counts = %d'%errors_MAP, fontsize=20)
    print(w_LDA, w)

    plt.show()