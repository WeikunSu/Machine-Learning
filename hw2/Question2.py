import matplotlib.pyplot as plt 
import numpy as np 
import math
import scipy.stats

class twoClassDistribution:
    def __init__(self, N, mu_1, mu_2, Sigma_1, Sigma_2, Pr_1, Pr_2):
        self.N = N
        self.mu_1 = np.array(mu_1)
        self.mu_2 = np.array(mu_2)
        self.Sigma_1 = Sigma_1
        self.Sigma_2 = Sigma_2
        self.Pr_1 = Pr_1
        self.Pr_2 = Pr_2
    
    def generateRandomVector(self):
        N_1 = int(self.N * self.Pr_1 + 0.5)
        N_2 = int(self.N * self.Pr_2 + 0.5)
        A_1 = np.linalg.cholesky(self.Sigma_1)
        A_2 = np.linalg.cholesky(self.Sigma_2)
        x_1 = np.dot(A_1, np.random.randn(2,N_1)) + np.dot(self.mu_1, np.ones([1,N_1]))
        x_2 = np.dot(A_2, np.random.randn(2,N_2)) + np.dot(self.mu_2, np.ones([1,N_2]))
        return [x_1, x_2]
    
    def plotTrueLabel(self, x_1, x_2):
        plt.scatter(x_1[0,:], x_1[1,:], c='b', marker='.')
        plt.scatter(x_2[0,:], x_2[1,:], c='r', marker='x')
    
    def plotEstimatedLabel(self, x_1, x_2):
        rv_1 = scipy.stats.multivariate_normal(self.mu_1.reshape(2), self.Sigma_1)
        rv_2 = scipy.stats.multivariate_normal(self.mu_2.reshape(2), self.Sigma_2)
        x_11 = []; x_21 = []; x_12 = []; x_22 = []
        flag_11 = 0; flag_21 = 0; flag_12 = 0; flag_22 = 0  #These flags are used to detect the empty array. Without these flags, we will have trouble scattering empty array. 
        for i in range(len(x_1[1,:])):
            if (rv_1.pdf(x_1[:,i]) * self.Pr_1)/(rv_2.pdf(x_1[:,i]) * self.Pr_2) >= 1:
                x_11.append(list(x_1[:,i]))
                flag_11 = 1
            else:
                x_21.append(list(x_1[:,i]))
                flag_21 = 1
        for i in range(len(x_2[1,:])):
            if (rv_1.pdf(x_2[:,i]) * self.Pr_1)/(rv_2.pdf(x_2[:,i]) * self.Pr_2) > 1:
                x_12.append(list(x_2[:,i]))
                flag_12 = 1
            else:
                x_22.append(list(x_2[:,i]))
                flag_22 = 1
        x_11 = np.array(x_11)
        x_12 = np.array(x_12)
        x_21 = np.array(x_21)
        x_22 = np.array(x_22)
        if flag_11 == 1:
            plt.scatter(x_11[:,0], x_11[:,1], c = 'g', marker='.')
        if flag_21 == 1:
            plt.scatter(x_21[:,0], x_21[:,1], c = 'r', marker='x')
        if flag_12 == 1:    
            plt.scatter(x_12[:,0], x_12[:,1], c = 'r', marker='.')
        if flag_22 == 1:    
            plt.scatter(x_22[:,0], x_22[:,1], c = 'g', marker='x')

if __name__ == '__main__':
    Case_1 = twoClassDistribution(400, [[0],[0]], [[3],[3]], np.eye(2), np.eye(2), 0.5, 0.5)
    Case_2 = twoClassDistribution(400, [[0],[0]], [[3],[3]], [[3,1],[1,0.8]], [[3,1],[1,0.8]], 0.5, 0.5)
    Case_3 = twoClassDistribution(400, [[0],[0]], [[2],[2]], [[2,0.5],[0.5,1]], [[2,-1.9],[-1.9,5]] ,0.5,0.5)
    Case_4 = twoClassDistribution(400, [[0],[0]], [[3],[3]], np.eye(2), np.eye(2), 0.05, 0.95)
    Case_5 = twoClassDistribution(400, [[0],[0]], [[3],[3]], [[3,1],[1,0.8]], [[3,1],[1,0.8]], 0.05, 0.95)
    Case_6 = twoClassDistribution(400, [[0],[0]], [[2],[2]], [[2,0.5],[0.5,1]], [[2,-1.9],[-1.9,5]], 0.05, 0.95)

    #1
    [x_1,x_2] = Case_1.generateRandomVector()
    plt.figure()
    plt.subplot(211)
    Case_1.plotTrueLabel(x_1, x_2)
    plt.subplot(212)
    Case_1.plotEstimatedLabel(x_1, x_2)

    #2
    [x_1,x_2] = Case_2.generateRandomVector()
    plt.figure()
    plt.subplot(211)
    Case_2.plotTrueLabel(x_1, x_2)
    plt.subplot(212)
    Case_2.plotEstimatedLabel(x_1, x_2)

    #3
    [x_1,x_2] = Case_3.generateRandomVector()
    plt.figure()
    plt.subplot(211)
    Case_3.plotTrueLabel(x_1, x_2)
    plt.subplot(212)
    Case_3.plotEstimatedLabel(x_1, x_2)

    #4
    [x_1,x_2] = Case_4.generateRandomVector()
    plt.figure()
    plt.subplot(211)
    Case_4.plotTrueLabel(x_1, x_2)
    plt.subplot(212)
    Case_4.plotEstimatedLabel(x_1, x_2)

    #5
    [x_1,x_2] = Case_5.generateRandomVector()
    plt.figure()
    plt.subplot(211)
    Case_5.plotTrueLabel(x_1, x_2)
    plt.subplot(212)
    Case_5.plotEstimatedLabel(x_1, x_2)

    #6
    [x_1,x_2] = Case_6.generateRandomVector()
    plt.figure()
    plt.subplot(211)
    Case_6.plotTrueLabel(x_1, x_2)
    plt.subplot(212)
    Case_6.plotEstimatedLabel(x_1, x_2)

    plt.show()

            



