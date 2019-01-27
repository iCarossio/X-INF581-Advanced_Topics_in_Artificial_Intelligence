from numpy import *
import copy
from sklearn import linear_model

class BR() :
    '''
        Binary Relevance
    '''

    h = None
    L = -1

    def __init__(self, h=linear_model.LogisticRegression()):
        self.hop = h

    def fit(self, X, Y):
        '''
            Train the model on the data.
        '''
        N,L = Y.shape
        self.L = L
        self.h = [ copy.deepcopy(self.hop) for j in range(self.L)]

        for j in range(self.L):
            self.h[j].fit(X, Y[:,j])
        return self

    def predict(self, X):
        '''
            Return predictions given X
        '''
        N,D = X.shape
        Y = zeros((N,self.L))
        for j in range(self.L):
            Y[:,j] = self.h[j].predict(X)
        return Y

    def predict_proba(self, X):
        '''
            Returns probabilities [ P(Y[i,1]|X[i]) ,..., P(Y[i,j]|X[i]) ] for each i-th row of X
        '''
        N,D = X.shape
        P = zeros((N,self.L))
        for j in range(self.L):
            P[:,j] = self.h[j].predict_proba(X)[:,1]
        return P

