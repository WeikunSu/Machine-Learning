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
        plt.scatter(x_1[0,:], x_1[1,:], c='b', marker='x', label='True Label 1')
        plt.scatter(x_2[0,:], x_2[1,:], c='r', marker='.', label='True Label 2')
        plt.xlabel(r'$x$', fontsize='15')
        plt.ylabel(r'$y$', fontsize='15')
        plt.legend(fontsize=15)
        plt.subplots_adjust(hspace=0.4)
    
    def plotEstimatedLabel(self, x_1, x_2):
        rv_1 = scipy.stats.multivariate_normal(self.mu_1.reshape(2), self.Sigma_1)
        rv_2 = scipy.stats.multivariate_normal(self.mu_2.reshape(2), self.Sigma_2)
        x_11 = []; x_21 = []; x_12 = []; x_22 = []  #x_ij means decide i to true label j
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
            plt.scatter(x_11[:,0], x_11[:,1], c = 'g', marker='x', label='Decision 1, Label 1')
        if flag_22 == 1:    
            plt.scatter(x_22[:,0], x_22[:,1], c = 'g', marker='.', label='Decision 2, Label 2')
        if flag_21 == 1:
            plt.scatter(x_21[:,0], x_21[:,1], c = 'r', marker='.', label='Decision 2, Label 1')
        if flag_12 == 1:    
            plt.scatter(x_12[:,0], x_12[:,1], c = 'r', marker='x', label='Decision 1, Label 2')
        plt.xlabel(r'$x$', fontsize='15')
        plt.ylabel(r'$y$', fontsize='15')
        plt.legend(fontsize=15)
        plt.subplots_adjust(hspace=0.4)

class FisherLDAClassifier:
    def __init__(self, x_1, x_2):
        muhat_1 = np.mean(x_1, axis=1)
        muhat_2 = np.mean(x_2, axis=1)
        Sigmahat_1 = np.cov(x_1)
        Sigmahat_2 = np.cov(x_2)
        S_b = (muhat_1 - muhat_2) * np.transpose(muhat_1 - muhat_2)
        S_w = Sigmahat_1 + Sigmahat_2
        D, V = np.linalg.eig(np.linalg.inv(S_w)*S_b)
        w = V[:, np.argmax(D)]
        self.y_1 = np.dot(w, x_1)        #project every y_1 to vector w
        self.y_2 = np.dot(w, x_2)        #project every y_2 to vector w
    
    def plotTrueLabel(self):
        plt.scatter(self.y_1, np.zeros(len(self.y_1)), marker='x', c='r', label='True Label 1')
        plt.scatter(self.y_2, np.zeros(len(self.y_2)), marker='o', c='',edgecolors='b', label='True Label 2')
        plt.legend(fontsize=15)
        plt.xlabel(r'$y_{LDA}$', fontsize=15)
        plt.subplots_adjust(hspace=0.4)

    def plotDecisionLabel(self):
        y_11 = np.array([]); y_21 = np.array([]); y_12 = np.array([]); y_22 = np.array([])
        Pr_1 = len(self.y_1) / (len(self.y_1)+len(self.y_2))
        Pr_2 = len(self.y_2) / (len(self.y_1)+len(self.y_2))
        for i in range(len(self.y_1)):
            if scipy.stats.norm.pdf(self.y_1[i], loc=np.mean(self.y_1), scale=np.std(self.y_1)) * Pr_1 >= scipy.stats.norm.pdf(self.y_1[i], loc=np.mean(self.y_2), scale=np.std(self.y_2)) * Pr_2:
                y_11 = np.append(y_11, self.y_1[i])
            else:
                y_21 = np.append(y_21, self.y_1[i])
        for i in range(len(self.y_2)):
            if scipy.stats.norm.pdf(self.y_2[i], loc=np.mean(self.y_1), scale=np.std(self.y_1)) * Pr_1 >= scipy.stats.norm.pdf(self.y_2[i], loc=np.mean(self.y_2), scale=np.std(self.y_2)) * Pr_2:
                y_12 = np.append(y_12, self.y_2[i])
            else:
                y_22 = np.append(y_22, self.y_2[i])
        plt.scatter(y_11, np.zeros(len(y_11)), c='g', marker='x', label='Decision 1, Label 1')
        plt.scatter(y_22, np.zeros(len(y_22)), c='', edgecolors='g', marker='o', label='Decision 2, Label 2')
        plt.scatter(y_21, np.zeros(len(y_21)), c='', edgecolors='r', marker='o', label='Decision 2, Label 1')
        plt.scatter(y_12, np.zeros(len(y_12)), c='r', marker='x', label='Decision 1, Label 2')
        plt.legend(fontsize=15)
        plt.xlabel(r'$y_{LDA}$', fontsize=15)
        plt.subplots_adjust(hspace=0.4)

if __name__ == '__main__':
    Case_1 = twoClassDistribution(400, [[0],[0]], [[3],[3]], np.eye(2), np.eye(2), 0.5, 0.5)
    Case_2 = twoClassDistribution(400, [[0],[0]], [[3],[3]], [[3,1],[1,0.8]], [[3,1],[1,0.8]], 0.5, 0.5)
    Case_3 = twoClassDistribution(400, [[0],[0]], [[2],[2]], [[2,0.5],[0.5,1]], [[2,-1.9],[-1.9,5]] ,0.5,0.5)
    Case_4 = twoClassDistribution(400, [[0],[0]], [[3],[3]], np.eye(2), np.eye(2), 0.05, 0.95)
    Case_5 = twoClassDistribution(400, [[0],[0]], [[3],[3]], [[3,1],[1,0.8]], [[3,1],[1,0.8]], 0.05, 0.95)
    Case_6 = twoClassDistribution(400, [[0],[0]], [[2],[2]], [[2,0.5],[0.5,1]], [[2,-1.9],[-1.9,5]], 0.05, 0.95)

    #plot case 1
    [x_1,x_2] = Case_1.generateRandomVector()
    plt.figure()
    plt.subplot(211)
    Case_1.plotTrueLabel(x_1, x_2)
    plt.title('Scatter of True Labels for Case 1', fontsize=15)
    plt.subplot(212)
    Case_1.plotEstimatedLabel(x_1, x_2)
    plt.title('Scatter of Decision Labels for Case 1', fontsize=15)
    Case_1_LDA = FisherLDAClassifier(x_1,x_2)
    plt.figure()
    plt.subplot(211)
    Case_1_LDA.plotTrueLabel()
    plt.title('LDA Projection of True Labels for Case 1', fontsize=15)
    plt.subplot(212)
    Case_1_LDA.plotDecisionLabel()
    plt.title('LDA Projection of Decision Labels for Case 1', fontsize=15)
    
    #plot case 2
    [x_1,x_2] = Case_2.generateRandomVector()
    plt.figure()
    plt.subplot(211)
    Case_2.plotTrueLabel(x_1, x_2)
    plt.title('Scatter of True Labels for Case 2', fontsize=15)
    plt.subplot(212)
    Case_2.plotEstimatedLabel(x_1, x_2)
    plt.title('Scatter of Decision Labels for Case 2', fontsize=15)
    Case_2_LDA = FisherLDAClassifier(x_1,x_2)
    plt.figure()
    plt.subplot(211)
    Case_2_LDA.plotTrueLabel()
    plt.title('LDA Projection of True Labels for Case 2', fontsize=15)
    plt.subplot(212)
    Case_2_LDA.plotDecisionLabel()
    plt.title('LDA Projection of Decision Labels for Case 2', fontsize=15)

    #plot case 3
    [x_1,x_2] = Case_3.generateRandomVector()
    plt.figure()
    plt.subplot(211)
    Case_3.plotTrueLabel(x_1, x_2)
    plt.title('Scatter of True Labels for Case 3', fontsize=15)
    plt.subplot(212)
    Case_3.plotEstimatedLabel(x_1, x_2)
    plt.title('Scatter of Decision Labels for Case 3', fontsize=15)
    Case_3_LDA = FisherLDAClassifier(x_1,x_2)
    plt.figure()
    plt.subplot(211)
    Case_3_LDA.plotTrueLabel()
    plt.title('LDA Projection of True Labels for Case 3', fontsize=15)
    plt.subplot(212)
    Case_3_LDA.plotDecisionLabel()
    plt.title('LDA Projection of Decision Labels for Case 3', fontsize=15)

    #plot case 4
    [x_1,x_2] = Case_4.generateRandomVector()
    plt.figure()
    plt.subplot(211)
    Case_4.plotTrueLabel(x_1, x_2)
    plt.title('Scatter of True Labels for Case 4', fontsize=15)
    plt.subplot(212)
    Case_4.plotEstimatedLabel(x_1, x_2)
    plt.title('Scatter of Decision Labels for Case 4', fontsize=15)
    Case_4_LDA = FisherLDAClassifier(x_1,x_2)
    plt.figure()
    plt.subplot(211)
    Case_4_LDA.plotTrueLabel()
    plt.title('LDA Projection of True Labels for Case 4', fontsize=15)
    plt.subplot(212)
    Case_4_LDA.plotDecisionLabel()
    plt.title('LDA Projection of Decision Labels for Case 4', fontsize=15)

    #plot case 5
    [x_1,x_2] = Case_5.generateRandomVector()
    plt.figure()
    plt.subplot(211)
    Case_5.plotTrueLabel(x_1, x_2)
    plt.title('Scatter of True Labels for Case 5', fontsize=15)
    plt.subplot(212)
    Case_5.plotEstimatedLabel(x_1, x_2)
    plt.title('Scatter of Decision Labels for Case 5', fontsize=15)
    Case_5_LDA = FisherLDAClassifier(x_1,x_2)
    plt.figure()
    plt.subplot(211)
    Case_5_LDA.plotTrueLabel()
    plt.title('LDA Projection of True Labels for Case 5', fontsize=15)
    plt.subplot(212)
    Case_5_LDA.plotDecisionLabel()
    plt.title('LDA Projection of Decision Labels for Case 5', fontsize=15)

    #plot case 6
    [x_1,x_2] = Case_6.generateRandomVector()
    plt.figure()
    plt.subplot(211)
    Case_6.plotTrueLabel(x_1, x_2)
    plt.title('Scatter of True Labels for Case 6', fontsize=15)
    plt.subplot(212)
    Case_6.plotEstimatedLabel(x_1, x_2)
    plt.title('Scatter of Decision Labels for Case 6', fontsize=15)
    Case_6_LDA = FisherLDAClassifier(x_1,x_2)
    plt.figure()
    plt.subplot(211)
    Case_6_LDA.plotTrueLabel()
    plt.title('LDA Projection of True Labels for Case 6', fontsize=15)
    plt.subplot(212)
    Case_6_LDA.plotDecisionLabel()
    plt.title('LDA Projection of Decision Labels for Case 6', fontsize=15)

    plt.show()
