#This is the code for question 1
import matplotlib.pyplot as plt 
import numpy as np 
import scipy.stats

class threeClassDistibution:
    def __init__(self, N, mu, Sigma, Pr):
        self.N = N
        self.mu = mu
        self.Sigma = Sigma
        self.Pr = Pr
    
    def generateData(self):
        thr = [0,self.Pr[0], np.sum(self.Pr[0:2]), np.sum(self.Pr[0:3])]
        u = np.random.rand(self.N)
        L = np.zeros((1,self.N))
        x = np.zeros((2,self.N))
        for i in range(3):
            indices = np.where(np.logical_and(u >= thr[i], u < thr[i+1]) == True)
            L[0, indices] = i * np.ones(len(indices))
            for k in indices[0]:
                x[:,k] = np.random.multivariate_normal(np.reshape(self.mu[:,i],(2)), self.Sigma[:,:][i])
        return x, L
    
    def plotTrueLabel(self, x, L):
        colarlist = ['r', 'g', 'b']
        labellist = ['True Label 1', 'True Label 2', 'True Label 3']
        plt.figure()
        for i in range(3):
            plt.scatter(x[0, np.where(L == i)], x[1, np.where(L == i)], marker='.', c=colarlist[i], label=labellist[i])
        plt.xlabel(r'$x$', fontsize='15')
        plt.ylabel(r'$y$', fontsize='15')
        plt.legend(fontsize=15)
        plt.title('True label')
    
    def obtainDecisionLabel(self, x, L):
        rv_1 = scipy.stats.multivariate_normal(np.reshape(self.mu[:,0],(2)), self.Sigma[:,:][0])
        rv_2 = scipy.stats.multivariate_normal(np.reshape(self.mu[:,1],(2)), self.Sigma[:,:][1])
        rv_3 = scipy.stats.multivariate_normal(np.reshape(self.mu[:,2],(2)), self.Sigma[:,:][2])
        D = np.zeros((1,self.N))
        for i in range(self.N):
            postiori_1 = rv_1.pdf(x[:,i])*self.Pr[0]
            postiori_2 = rv_2.pdf(x[:,i])*self.Pr[1]
            postiori_3 = rv_3.pdf(x[:,i])*self.Pr[2]
            if postiori_1 >= postiori_2 and postiori_1 >= postiori_3:
                D[0,i] = 0
            elif postiori_2 >= postiori_1 and postiori_2 >= postiori_3:
                D[0,i] = 1
            else:
                D[0,i] = 2
        return D
    
    def plotEstimatedLabel(self, x, L, D):
        colarlist = ['r', 'g', 'b']
        markerlist = ['.','*','x']
        labellist = [['D=1, L=1', 'D=2, L=1', 'D=3, L=1'],['D=1, L=2', 'D=2, L=2', 'D=3, L=2'],['D=1, L=3', 'D=2, L=3', 'D=3, L=3']]
        self.confusionmatrix = np.zeros((3,3))
        plt.figure()
        for i in range(3):
            for j in range(3):
                plt.scatter(x[0, np.where(np.logical_and(D == j, L == i)[0] == True)], x[1,np.where(np.logical_and(D == j, L == i)[0] == True)], c=colarlist[j], marker=markerlist[i], label=labellist[i][j])
                self.confusionmatrix[j,i] = len(x[0, np.where(np.logical_and(D == j, L == i)[0] == True)][0])
        plt.xlabel(r'$x$', fontsize='15')
        plt.ylabel(r'$y$', fontsize='15')
        plt.legend(fontsize=15)
        plt.title('Estimated label')



Sigma = np.array([[[0.0 for k in range(2)] for j in range(2)] for i in range(3)])

#Set parameters here
N = 10000
mu = np.array([[-1, 1, 0], [0, 0, 1]])
Sigma[:,:][0] = 0.1*np.array([[10,-4],[-4,5]]); Sigma[:,:][1] = 0.1*np.array([[5,0],[0,2]]); Sigma[:,:][2] = 0.1*np.eye(2)
Pr = [0.15,0.35,0.5]
#Generate dataset and plots
case = threeClassDistibution(N, mu, Sigma, Pr)
x,L = case.generateData()
case.plotTrueLabel(x, L)
D = case.obtainDecisionLabel(x, L)
case.plotEstimatedLabel(x, L, D)
#Print the result
numberofsample_1 = np.count_nonzero(L == 0)
numberofsample_2 = np.count_nonzero(L == 1)
numberofsample_3 = np.count_nonzero(L == 2)
print('Number of class 1 sample: %d'%numberofsample_1)
print('Number of class 2 sample: %d'%numberofsample_2)
print('Number of class 3 sample: %d'%numberofsample_3)
print()

print('Confusion Matrix:')
print(case.confusionmatrix)
print()

pr_error = 1 - (case.confusionmatrix[0,0]+case.confusionmatrix[1,1]+case.confusionmatrix[2,2])/np.sum(case.confusionmatrix)
print('Probability of error:%f'%pr_error)
print()

numberofmis = N - (case.confusionmatrix[0,0]+case.confusionmatrix[1,1]+case.confusionmatrix[2,2])
print('Number of misclassification:%d'%numberofmis)
plt.show()