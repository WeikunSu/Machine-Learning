#This is the code for exam2 Question 1

import matplotlib.pyplot as plt 
import numpy as np
import scipy.stats
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf


def GMMgen(N, alpha, mu, Sigma):
    thr = [0]
    thr.extend(np.cumsum(alpha))
    u = np.random.rand(N)
    L = np.zeros((N))
    x = np.zeros((N, len(mu[0,:])))
    for i in range(len(alpha)):
        indices = np.squeeze(np.where(np.logical_and(u >= thr[i], u < thr[i+1]) == True))
        L[indices] = i * np.ones(len(indices))
        for k in indices:
            x[k,:] = np.random.multivariate_normal(mu[i,:], Sigma[:,:,i])
    return x, L

def MAPestimate(X, mu, Sigma, Pr):
    n_class = len(Pr)
    rv = [0]*n_class
    for j in range(n_class):
        rv[j] = scipy.stats.multivariate_normal(np.reshape(mu[j],(len(mu[0]))), Sigma[:,:,j])

    D = np.zeros(len(X))
    for i in range(len(X)):
        posterior_i = np.zeros(n_class)
        for j in range(n_class):
            posterior_i[j] = rv[j].pdf(X[i,:]) * Pr[j]
        D[i] = np.squeeze(np.where(posterior_i == max(posterior_i)))

    return D


if __name__ == "__main__":
    ## debug
    plot_true_distribution = False
    plot_MAP_classification = True

    ##
    
    ## set up the true distribution
    prior = np.array([0.1, 0.2, 0.3, 0.4])
    mu_true = 0.8*np.array([[1,1,1],[-1,1,0],[-1,-1,-1],[1,-1,0]])
    Sigma_true = np.zeros([3,3,4])
    Sigma_true[:,:,0] = 0.3*np.array([[1,0,0],
                                      [0,1,0],
                                      [0,0,1]])
    Sigma_true[:,:,1] = 0.3*np.array([[1,0,1],
                                      [0,2,2],
                                      [1,2,4]])
    Sigma_true[:,:,2] = 0.3*np.array([[4,1,0],
                                      [1,1,1],
                                      [0,1,2]])
    Sigma_true[:,:,3] = 0.3*np.array([[1,1,1],
                                      [1,4,1],
                                      [1,1,2]])

    ## Part 1 ###########################
    N = 1000             # number of samples
    x, L = GMMgen(N, prior,mu_true, Sigma_true)
    # plot true samples
    fig_1 = plt.figure()
    ax_1 = Axes3D(fig_1)
    clist=['r','g','b','m']
    labellist = ['L=1','L=2','L=3','L=4']
    for i in range(len(prior)):
        ax_1.scatter(x[np.where(L == i),0],x[np.where(L == i),1],x[np.where(L == i),2],marker='o', c=clist[i], label= labellist[i])
    ax_1.set_title('Part 1: 1000 samples with true labels',fontsize=20)
    ax_1.legend(fontsize=20)



    ## Part 2 ##########################
    # Generate samples
    N_test = 10000             # number of samples
    x_test, L_test = GMMgen(N_test, prior,mu_true, Sigma_true)
    D_MAP = MAPestimate(x_test,mu_true, Sigma_true,prior)
    P_error_MAP = np.count_nonzero(D_MAP != L_test) / N_test
    # plot MAP classified samples
    fig_2 = plt.figure()
    ax_2 = Axes3D(fig_2)
    clist=['r','g','b','m']
    labellist = ['D=1','D=2','D=3','D=4']
    for i in range(len(prior)):
        ax_2.scatter(x_test[np.where(D_MAP == i),0],x_test[np.where(D_MAP == i),1],x_test[np.where(D_MAP == i),2],marker='o', c=clist[i], label= labellist[i])
    ax_2.set_title('Part 2: 10000 samples classified by optimal MAP classifier \n Probability of error:%.4f'%P_error_MAP,fontsize=20)
    ax_2.legend(fontsize=20)


    ## Part 3 ##############################
    N_list = [100,1000,10000,100000]
    K = 10
    fig_3 = plt.figure('model order select')
    ax_3 = plt.subplot(111)


    for N_MLP in N_list:
        x_MLP, L_MLP = GMMgen(N_MLP, prior,mu_true, Sigma_true)
        dummy = np.ceil(np.linspace(0,N_MLP,K+1))
        index_partition_limit = np.zeros([K,2])
        for k in range(K):
            index_partition_limit[k,:] = [dummy[k],dummy[k+1]-1]
        num_node_list = 10**(np.linspace(1, 4, 4))
        P_correct_mean = np.zeros((len(num_node_list,)))
        for i in range(len(num_node_list)):
            print(i,len(num_node_list))
            num_node = num_node_list[i]
            P_correct = np.zeros((K,))
            for k in range(K):
                ind_validate = np.arange(index_partition_limit[k,0], index_partition_limit[k,1]+1)
                ind_validate = ind_validate.astype(int)
                x_validate = x_MLP[ind_validate,:]
                L_validate = L_MLP[ind_validate]
                ind_train = np.hstack((np.arange(0,index_partition_limit[k,0]),np.arange(index_partition_limit[k,1]+1, N_MLP)))
                ind_train = ind_train.astype(int)
                x_train = x_MLP[ind_train,:]
                L_train = L_MLP[ind_train]
                model = tf.keras.models.Sequential([
                    tf.keras.layers.Dense(num_node, activation='tanh'),
                    tf.keras.layers.Dense(4, activation='softplus')
                    ])
                model.compile(optimizer='sgd',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
                model.fit(x_train,L_train,epochs=5,verbose=0)
                pre_MLP = model.predict(x_validate)
                D_MLP = np.argmax(pre_MLP[:],axis=1)
                P_correct[k] = np.count_nonzero(D_MLP == L_validate) / N_MLP
            P_correct_mean[i] = np.mean(P_correct)
        num_node_best_ind = np.squeeze(np.where(P_correct_mean == (max(P_correct_mean))))
        num_node_best = num_node_list[num_node_best_ind]
        print(num_node_best)                           # output num_node_best
        
            

        #plot the searching process
        
        ax_3.semilogx(num_node_list, P_correct_mean, marker='x', label='N = %d'%N_MLP)
    ax_3.set_title('The performance of optimizer with different number of nodes',fontsize=20)
    ax_3.set_xlabel('Number of nodes', fontsize=20)
    ax_3.set_ylabel('Probability of correct decisions', fontsize=20)
    ax_3.legend(fontsize=20)

    # apply to test data
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(num_node_best, activation='tanh'),
        tf.keras.layers.Dense(4, activation='softplus')
    ])
    model.compile(optimizer='sgd',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    model.fit(x_MLP,L_MLP,epochs=5,verbose=0)
    pre_MLP = model.predict(x_test)
    D_MLP = np.argmax(pre_MLP[:],axis=1)
    P_error_MLP = np.count_nonzero(D_MLP != L_test) / N_test

    fig_4 = plt.figure()
    ax_4 = Axes3D(fig_4)
    clist=['r','g','b','m']
    labellist = ['D_MLP=1','D_MLP=2','D_MLP=3','D_MLP=4']
    for i in range(len(prior)):
        ax_4.scatter(x_test[np.where(D_MLP == i),0],x_test[np.where(D_MLP == i),1],x_test[np.where(D_MLP == i),2],marker='o', c=clist[i], label= labellist[i])
    ax_4.set_title('Part 3: 10000 samples classified by multilayer perceptron neural network \n Probability of error:%f'%P_error_MLP,fontsize=20)
    ax_4.legend(fontsize=20)








    plt.show()