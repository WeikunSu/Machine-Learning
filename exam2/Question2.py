## Code for exam2 Question 2

import matplotlib.pyplot as plt 
import numpy as np
import scipy.stats
import tensorflow as tf
import math

def GMMgen(N, alpha, mu, covEvalues):
    thr = [0]
    covEvectors = np.zeros([2,2,3])
    covEvectors[:,:,0] = np.array([[1, -1],[1, 1]]) / math.sqrt(2)
    covEvectors[:,:,1] = np.array([[1, 0],[0, 1]])
    covEvectors[:,:,2] = np.array([[1, -1],[1, 1]]) / math.sqrt(2)
    thr.extend(np.cumsum(alpha))
    u = np.random.rand(N)
    L = np.zeros((N))
    x = np.zeros((N, len(mu[0,:])))
    for i in range(len(alpha)):
        indices = np.squeeze(np.where(np.logical_and(u >= thr[i], u < thr[i+1]) == True))
        L[indices] = i * np.ones(len(indices))
        for k in indices:
            x[k,:] = np.dot(np.dot(covEvectors[:,:,i],covEvalues),np.random.randn(2,1)).T + mu[i]
    return x, L




if __name__ == "__main__":
    ## set the parameter for data
    N_train = 1000
    N_test = 10000
    prior = [0.33,0.34,0.33]
    mu_true = np.array([[-18,-8],[0,0],[18,8]])
    covEvalues = [[3.2 ,0],[0 ,0.6]]

    # generate training samples and test samples
    x_train, L_train = GMMgen(N_train, prior, mu_true,covEvalues)
    x_test, L_test = GMMgen(N_test,prior,mu_true,covEvalues)


    # plot the samples
    clist=['r','g','b']
    labellist = ['L=1','L=2','L=3']
    plt.figure('True label')
    plt.subplot(121)
    for i in range(len(prior)):
        plt.scatter(x_train[np.where(L_train == i),0],x_train[np.where(L_train == i),1],marker='.', c=clist[i], label= labellist[i])
    plt.title('Samples for training',fontsize=20)
    plt.legend(fontsize=20)
    plt.xlabel(f'$x_1$',fontsize=20)
    plt.ylabel(f'$x_2$',fontsize=20)

    plt.subplot(122)
    for i in range(len(prior)):
        plt.scatter(x_test[np.where(L_test == i),0],x_test[np.where(L_test == i),1],marker='.', c=clist[i], label= labellist[i])
    plt.title('Samples for testing',fontsize=20)
    plt.legend(fontsize=20)
    plt.xlabel(f'$x_1$',fontsize=20)
    plt.ylabel(f'$x_2$',fontsize=20)


    ## K-fold to choose the best number of nodes in hidden layer

    K = 10

    #
    dummy = np.ceil(np.linspace(0,N_train,K+1))
    index_partition_limit = np.zeros([K,2])
    for k in range(K):
        index_partition_limit[k,:] = [dummy[k],dummy[k+1]-1]
    
    act_list = ['sigmoid','softplus']            ##['sigmoid','softplus']
    num_node_list = 10**(np.linspace(1, 3, 5))
    mse_mean = np.zeros((len(act_list), len(num_node_list,)))

    for j in range(len(act_list)):
        act = act_list[j]
        print(j,len(act_list))
        for i in range(len(num_node_list)):
            num_node = num_node_list[i]
            mse = np.zeros((K,))
            for k in range(K):
                ind_validate = np.arange(index_partition_limit[k,0], index_partition_limit[k,1]+1)
                ind_validate = ind_validate.astype(int)
                x_validate = x_train[ind_validate,:]
                ind_train_k = np.hstack((np.arange(0,index_partition_limit[k,0]),np.arange(index_partition_limit[k,1]+1, N_train)))
                ind_train_k = ind_train_k.astype(int)
                x_train_k = x_train[ind_train_k,:]
                
                model = tf.keras.models.Sequential([
                    tf.keras.layers.Dense(num_node, activation=act),       ## 'softplus'  'sigmoid'
                    tf.keras.layers.Dense(1, activation='linear')
                ])
                model.compile(optimizer='adam',
                            loss='mse',
                            metrics=['mse'])
                model.fit(x_train_k[:,0],x_train_k[:,1],epochs=100,verbose=0)
                x_pre = model.predict(x_validate[:,0])
                mse[k] = np.mean((x_pre.reshape(1,-1) - x_validate[:,1])**2)
            mse_mean[j,i] = np.mean(mse)

    act_best_ind ,node_best_ind = np.squeeze(np.where(mse_mean == (mse_mean.min())))
    num_node_best = num_node_list[node_best_ind]
    act_best = act_list[act_best_ind]

    ## plot the process of model selection
    plt.figure('K-fold result')
    for j in range(len(act_list)):
        plt.semilogx(num_node_list, mse_mean[j,:], marker='x', label = act_list[j])
    plt.title('C.V. process of model selection \n num_node_best = %d, act_best = %s'%(num_node_best,act_best), fontsize=20)
    plt.xlabel('number of nodes',fontsize=20)
    plt.ylabel('MSE',fontsize=20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.legend(fontsize = 20)

    ## Apply the best model to test data
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(num_node_best, activation=act_best),       ## 'softplus'  'sigmoid'
        tf.keras.layers.Dense(1, activation='linear')
        ])
    model.compile(optimizer='adam',
                loss='mse',
                metrics=['mse'])
    model.fit(x_train[:,0],x_train[:,1],epochs=100,verbose=0)
    x_pre = model.predict(x_test[:,0])
    mse = np.mean((x_pre.reshape(1,-1) - x_test[:,1])**2)

    ## Plot the performance of best model
    plt.figure('The performance of best model')
    plt.scatter(x_test[:,0],x_test[:,1],marker='.', label='Test samples')
    plt.scatter(x_test[:,0],x_pre,marker='.', label='Estimated results')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(f'$x_1$',fontsize=20)
    plt.ylabel(f'$x_2$',fontsize=20)
    plt.title('Performance of best model \n num_node = %d, activation function = %s \n MSE = %f'%(num_node_best,act_best,mse), fontsize=20)
    plt.legend(fontsize=20)
    

    plt.show()
