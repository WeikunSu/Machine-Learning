import matplotlib.pyplot as plt 
import numpy as np
from sklearn import svm
import math

def GMMgen(N, alpha, mu, Sigma, radius, angle):
    thr = [0]
    thr.extend(np.cumsum(alpha))
    u = np.random.rand(N)
    L = np.zeros((N))
    x = np.zeros((N, 2))
    i = 0
    indices = np.squeeze(np.where(np.logical_and(u >= thr[i], u < thr[i+1]) == True))
    L[indices] = i * np.ones(len(indices))
    for k in indices:
        x[k,:] = np.random.multivariate_normal(mu, Sigma)
    i = 1
    indices = np.squeeze(np.where(np.logical_and(u >= thr[i], u < thr[i+1]) == True))
    L[indices] = i * np.ones(len(indices))
    for k in indices:
        r = (radius[1] - radius[0]) * np.random.rand() + radius[0]
        ang = (angle[1] - angle[0]) * np.random.rand() + angle[0]
        x[k,0] = r*math.cos(ang)
        x[k,1] = r*math.sin(ang)
    return x, L



if __name__ == "__main__":
    ## Set up the parameters of our data
    N = 1000
    prior = np.array([0.35,0.65])
    mu = np.array([0,0])
    Sigma = np.eye(2)
    radius = np.array([2,3])
    angle = np.array([-math.pi,math.pi])
    K = 10 

    ## Generate the training data
    x, L = GMMgen(N, prior, mu, Sigma, radius, angle)
    plt.figure()
    plt.scatter(x[np.where(L == 0),0],x[np.where(L == 0),1], marker='.', label='Class -')
    plt.scatter(x[np.where(L == 1),0],x[np.where(L == 1),1], marker='+', label='Class +')
    plt.title('Training data',fontsize=20)
    plt.legend(fontsize = 20)

    ## Generate test data
    x_test, L_test = GMMgen(N, prior, mu, Sigma, radius, angle)
    
    #### Linear Kernel

    ## Optimize C_linear
    dummy = np.ceil(np.linspace(0,N,K+1))
    index_partition_limit = np.zeros([K,2])
    for k in range(K):
        index_partition_limit[k,:] = [dummy[k],dummy[k+1]-1]
    C_linear_list = 10**(np.linspace(-3,2, 6))
    P_error_mean = np.zeros((len(C_linear_list,)))
    for i in range(len(C_linear_list)):
        print(i,len(C_linear_list))
        C = C_linear_list[i]
        P_error = np.zeros((K,))
        for k in range(K):
            ind_validate = np.arange(index_partition_limit[k,0], index_partition_limit[k,1]+1)
            ind_validate = ind_validate.astype(int)
            x_validate = x[ind_validate,:]
            L_validate = L[ind_validate]
            ind_train = np.hstack((np.arange(0,index_partition_limit[k,0]),np.arange(index_partition_limit[k,1]+1, N)))
            ind_train = ind_train.astype(int)
            x_train = x[ind_train,:]
            L_train = L[ind_train]
            SVMk = svm.SVC(C=C,kernel='linear').fit(x_train,L_train)
            D_validate = SVMk.predict(x_validate)
            P_error[k] = np.count_nonzero(L_validate != D_validate) / len(L_validate)
        P_error_mean[i] = np.mean(P_error)
    C_best_ind = np.squeeze(np.where(P_error_mean == (min(P_error_mean))))
    C_best = C_linear_list[C_best_ind]
    if type(C_best) == np.float64:
        C_best = C_best
    else:
        C_best = C_best[0]

    ## Show the process of searching C_linear
    plt.figure()
    plt.semilogx(C_linear_list, P_error_mean, marker='x')
    plt.title('Linear kernel \n Smallest probability of error is %f'%min(P_error_mean),fontsize=20)
    plt.xlabel('C', fontsize=20)
    plt.ylabel('Probability of error', fontsize=20)
    print(C_best)

    ## Plot the decision on training samples
    SVM_best = svm.SVC(C=C_best, kernel='linear').fit(x, L)
    D = SVM_best.predict(x)
    error_count = np.count_nonzero(L != D)
    P_error_linear = np.count_nonzero(L != D) / len(D)
    plt.figure()
    plt.scatter(x[np.where(np.logical_and(D == 0, L == 0)),0],x[np.where(np.logical_and(D == 0, L == 0)),1], marker='.', c='g', label='D = -, L = -')
    plt.scatter(x[np.where(np.logical_and(D == 0, L == 1)),0],x[np.where(np.logical_and(D == 0, L== 1)),1], marker='.', c='r', label='D = -, L = +')
    plt.scatter(x[np.where(np.logical_and(D == 1, L == 0)),0],x[np.where(np.logical_and(D == 1, L == 0)),1], marker='+', c='r', label='D = +, L = -')
    plt.scatter(x[np.where(np.logical_and(D == 1, L == 1)),0],x[np.where(np.logical_and(D == 1, L == 1)),1], marker='+', c='g', label='D = +, L = +')
    plt.title('SVM (linear kernel) classification on training data \n Probability of error: %f \n Error count = %d'%(P_error_linear,error_count),fontsize=20)
    plt.legend(fontsize = 20)
    
    ## Plot the decision on test samples
    SVM_best = svm.SVC(C=C_best, kernel='linear').fit(x, L)
    D_test = SVM_best.predict(x_test)
    P_error_linear = np.count_nonzero(L_test != D_test) / len(D_test)
    plt.figure()
    plt.scatter(x_test[np.where(np.logical_and(D_test == 0, L_test == 0)),0],x_test[np.where(np.logical_and(D_test == 0, L_test == 0)),1], marker='.', c='g', label='D = -, L = -')
    plt.scatter(x_test[np.where(np.logical_and(D_test == 0, L_test == 1)),0],x_test[np.where(np.logical_and(D_test == 0, L_test == 1)),1], marker='.', c='r', label='D = -, L = +')
    plt.scatter(x_test[np.where(np.logical_and(D_test == 1, L_test == 0)),0],x_test[np.where(np.logical_and(D_test == 1, L_test == 0)),1], marker='+', c='r', label='D = +, L = -')
    plt.scatter(x_test[np.where(np.logical_and(D_test == 1, L_test == 1)),0],x_test[np.where(np.logical_and(D_test == 1, L_test == 1)),1], marker='+', c='g', label='D = +, L = +')
    plt.title('SVM (linear kernel) classification on test data \n Probability of error: %f'%P_error_linear,fontsize=20)
    plt.legend(fontsize = 20)
    

    #### Gaussian Kernel

    ## Optimize C_gaussian
    dummy = np.ceil(np.linspace(0,N,K+1))
    index_partition_limit = np.zeros([K,2])
    for k in range(K):
        index_partition_limit[k,:] = [dummy[k],dummy[k+1]-1]
    C_gaussian_list = 10**(np.linspace(-3,2, 6))
    Sigma_list = 10**(np.linspace(-2,2,5))
    P_error_mean = np.zeros((len(Sigma_list),len(C_gaussian_list)))
    for j in range(len(Sigma_list)):
        Sigma = Sigma_list[j]
        print(j,len(Sigma_list))
        for i in range(len(C_gaussian_list)):
            C = C_gaussian_list[i]
            P_error = np.zeros((K,))
            for k in range(K):
                ind_validate = np.arange(index_partition_limit[k,0], index_partition_limit[k,1]+1)
                ind_validate = ind_validate.astype(int)
                x_validate = x[ind_validate,:]
                L_validate = L[ind_validate]
                ind_train = np.hstack((np.arange(0,index_partition_limit[k,0]),np.arange(index_partition_limit[k,1]+1, N)))
                ind_train = ind_train.astype(int)
                x_train = x[ind_train,:]
                L_train = L[ind_train]
                SVMk = svm.SVC(C=C,kernel='rbf', gamma=Sigma).fit(x_train,L_train)
                D_validate = SVMk.predict(x_validate)
                P_error[k] = np.count_nonzero(L_validate != D_validate) / len(L_validate)
            P_error_mean[j,i] = np.mean(P_error)
    Sigma_best_ind ,C_best_ind = np.squeeze(np.where(P_error_mean == (P_error_mean.min())))
    C_best = C_gaussian_list[C_best_ind]
    Sigma_best = Sigma_list[Sigma_best_ind]
    if type(C_best) == np.float64 :
        C_best = C_best
        Sigma_best = Sigma_best
    else:
        C_best = C_best[0]
        Sigma_best = Sigma_best[0]

    ## Show the process of searching C_gaussian and Sigma_gaussian
    plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    c = plt.contour(C_gaussian_list,Sigma_list,P_error_mean)
    c.clabel()
    plt.xlabel('C',fontsize=20)
    plt.ylabel('Sigma',fontsize=20)
    plt.title('Gaussian kernel \n Smallest probability of error is %f \n C_best = %f , sigma_best = %f'%(P_error_mean.min(),C_best,Sigma_best),fontsize=20)
    
    
    ## Plot the decision on training samples
    SVM_best = svm.SVC(C=C_best, kernel='rbf',gamma=Sigma_best).fit(x, L)
    D = SVM_best.predict(x)
    error_count = np.count_nonzero(L != D)
    P_error_gaussian = np.count_nonzero(L != D) / len(D)
    plt.figure()
    plt.scatter(x[np.where(np.logical_and(D == 0, L == 0)),0],x[np.where(np.logical_and(D == 0, L == 0)),1], marker='.', c='g', label='D = -, L = -')
    plt.scatter(x[np.where(np.logical_and(D == 0, L == 1)),0],x[np.where(np.logical_and(D == 0, L== 1)),1], marker='.', c='r', label='D = -, L = +')
    plt.scatter(x[np.where(np.logical_and(D == 1, L == 0)),0],x[np.where(np.logical_and(D == 1, L == 0)),1], marker='+', c='r', label='D = +, L = -')
    plt.scatter(x[np.where(np.logical_and(D == 1, L == 1)),0],x[np.where(np.logical_and(D == 1, L == 1)),1], marker='+', c='g', label='D = +, L = +')
    plt.title('SVM (Gaussian kernel) classification on training data \n Probability of error: %f \n Error count = %d'%(P_error_gaussian,error_count),fontsize=20)
    plt.legend(fontsize = 20)
    
    ## Plot the decision on test samples
    D_test = SVM_best.predict(x_test)
    P_error_gaussian = np.count_nonzero(L_test != D_test) / len(D_test)
    plt.figure()
    plt.scatter(x_test[np.where(np.logical_and(D_test == 0, L_test == 0)),0],x_test[np.where(np.logical_and(D_test == 0, L_test == 0)),1], marker='.', c='g', label='D = -, L = -')
    plt.scatter(x_test[np.where(np.logical_and(D_test == 0, L_test == 1)),0],x_test[np.where(np.logical_and(D_test == 0, L_test == 1)),1], marker='.', c='r', label='D = -, L = +')
    plt.scatter(x_test[np.where(np.logical_and(D_test == 1, L_test == 0)),0],x_test[np.where(np.logical_and(D_test == 1, L_test == 0)),1], marker='+', c='r', label='D = +, L = -')
    plt.scatter(x_test[np.where(np.logical_and(D_test == 1, L_test == 1)),0],x_test[np.where(np.logical_and(D_test == 1, L_test == 1)),1], marker='+', c='g', label='D = +, L = +')
    plt.title('SVM (Gaussian kernel) classification on test data \n Probability of error: %f'%P_error_gaussian,fontsize=20)
    plt.legend(fontsize = 20)

    
    
    plt.show()



