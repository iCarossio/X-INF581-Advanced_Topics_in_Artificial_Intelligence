from sklearn.linear_model import LogisticRegression
from skmultilearn.problem_transform import LabelPowerset
from numpy import *

class LP():
    '''
        Label Powerset Method
    '''

    h = None

    def __init__(self, h=LogisticRegression()):
        self.h = LabelPowerset(h)

    def fit(self, X, Y):
        '''
            Train the model on training data X,Y
        '''
        return self.h.fit(X,Y)

    def predict(self, X):
        '''
            Return predictions Y, given X
        '''
        return self.h.predict(X)

    def predict_proba(self, X):
        '''
            Return matrix P, where P[i,j] = P(Y[i,j] = 1 | X[i])
            (where i-th row/example, and j-th label)
        '''
        return self.h.predict_proba(X)

