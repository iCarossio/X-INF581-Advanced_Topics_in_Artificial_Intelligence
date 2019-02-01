from numpy import *
import copy
from sklearn.linear_model import LogisticRegression

class CC() :
    '''
        Classifier Chain
    '''

    h = None
    L = -1

    def __init__(self, h=LogisticRegression()):
        ''' 
        Setup.

        Parameters
        ----------
        h : a sklearn model
            The instantiated base classifier

        '''
        self.base_classifier = h

    def fit(self, X, Y):
        ''' 
        Train the chain.

        Parameters
        ----------
        X : input matrix (N * D array)
        Y : label matrix (N * L array)

        '''
        N, self.L = Y.shape
        L         = self.L
        N, D      = X.shape

        # Copy the base model for each label ...
        self.h = [ copy.deepcopy(self.base_classifier) for j in range(L)]
        XY = zeros((N, D + L-1))
        XY[:,0:D] = X
        XY[:,D:] = Y[:,0:L-1]
        # ... and train each model.
        for j in range(self.L):
            self.h[j].fit(XY[:,0:D+j], Y[:,j])

        return self

    def explore_paths(self, x, epsilon=0.5):
        '''
            epsilon-Greedy exploration of the probability tree.

            Carry out an exploration of the probability tree given instance x,
            using the epsilon-greedy strategy.

            Parameters
            ----------
                x       : A D-dimensional array contains values for the D input features.
                epsilon : float value of epsilon considered in the search

            Returns
            -------
                branches : a list of branches involved in your search, where 
                           a branch is a tuple (parent,child,branch_score,path_score) where
                            - parent : an integer array to identify the path to this node
                            - child  : an integer array to identify the path to this node
                            - branch_score : the score by taking this branch
                            - path_score   : the score obtained so far along the path relevant to this branch
                y        : the best path (of those explored) to a goal node
                p        : the score/payoff associated with this path
        '''

        ###############################################
        #       This function is implemented as UCS 
        #       ε-approximate search and works for 
        #       any espilon
        ###############################################

        branches       = []    # to store the branches we go down
        priority_queue = []    # priority-queue to implement UCS ε-approximate search
        final_branches = []    # to store the final branches (when j = self.L)
        y  = zeros(self.L)     # an array to store labels (the best path)
        p  = 1.                # path score so far
        j  = -1                # current label index

        priority_queue.append((p,j,y)) # Initialize the priority queue

        while priority_queue:
            p,j,y = priority_queue.pop() # Pop next neihboor in the tree
            j+=1

            if j<self.L: # Stop condition
                
                if j>0:
                    xy = append(x, y[0:j]).reshape(1,-1) # Add previous predictions to the testing set
                else:
                    xy = x.reshape(1,-1) # Make x into a 2*D array

                P_j = self.h[j].predict_proba(xy)[0] # (N.B. [0], because it is the first and only row)
                for k, proba in enumerate(P_j): # Go trough labels and their probas
                    if proba > epsilon: # 
                        y_c = copy.deepcopy(y) # Make a deep copy of the paths to save them in branches
                        y_c[j] = k # Epsilon-greedy strategy
                        p_new = p*proba
                        branch = (y_c[0:j].astype(int),y_c[0:j+1].astype(int),proba,p_new)
                        branches.append(branch)
                        priority_queue.append((p_new,j,y_c))
                        priority_queue.sort() # Sort the list as a priority queue

                        if j == self.L-1:
                            final_branches.append(branch) # Keep complete branches 
        
        _, y, _, p = sorted(final_branches, key=lambda tup: tup[3], reverse=True)[0] # Get the best path according to its costs among the final branches

        return branches,y,p


    def predict(self, X, epsilon=0.5):
        ''' 
        Predict.

        Parameters
        ----------
        X : input matrix (N * D array)

        Returns
        -------

        A binary matrix (N * L array) of predictions.

        '''

        N,D = X.shape
        Yp = zeros((N,self.L))

        for n in range(N):
            x = X[n]
            paths,yp,w_max = self.explore_paths(x, epsilon)
            Yp[n] = yp

        return Yp