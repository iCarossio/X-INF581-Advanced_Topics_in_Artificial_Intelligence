from numpy import *
from utils import tile2pos
from copy import *

def explore_paths(paths,X,y,p=1.,t=0):
    '''
        Explore paths.
                          
        Explore all possible paths that the target may take, store them in 'paths' with an associated score (probability).

        Parameters
        ----------

        X: the T observations
        y: the path, of length T,
        p: score/probability of the path y.
        t: time-step of the path
        paths: reference to a list, when t = T - 1, then add (y,p) to this list, else update y and p 
               and recursively call the function with (paths,X,y,p,t+1)
    '''
    T = X.shape[0]

    # if path complete, add it and retutrn
    # ...

    # if probability 0, return
    # ...

    # for each state i:
    #   y_new <- mark it into the path y (N.B. good to make a copy(.) of the path here)
    #   p_new <- get the probabilitiy associated with it
    #   call the same function: explore_paths(paths,X,y_new,p_new,t+1) ...


        
#####################################

# Load some paths and their observations
X = loadtxt('X.dat')
y = loadtxt('y.dat', dtype=int)

seed = 0
X = X[seed:seed+5,:]
print(X)
y = y[seed:seed+5]

# Obtain all *possible* paths and a relative score (probability) associated with each
T,D = X.shape
paths = []
explore_paths(paths,X,y=-ones(T))

# Print out these paths and their joint-mode score (normalized to a distribution, st they sum to 1) ...
# (TODO), e.g.,  
#   [  4   3   8  13  18], 0.8   (equiv. grid path : [(0, 4), (0, 3), (1, 3), (2, 3), (3, 3)])
#   [  4   9   8  7  6],   0.2   (equiv. grid path : [(0, 4), (1, 4), (1, 3), (1, 2), (1, 1)])

# Print out the marginas dist. of the final node and associated scores (normalized to a distribution, st they sum to 1)
# (TODO), e.g.,
# 6,  0.95 (equiv. grid square: 1,1)
# 18, 0.05 (equiv. grid square: 3,3)

# Decide whether to 'pounce' or not
# (TODO), e.g., 'Yes'

# Compare to the true path (Evaluation):
print("y = %s, i.e., tile path: %s, finishing at y = %d (tile: %s)" % (str(y), str([(tile2pos(y[i])[0],tile2pos(y[i])[1]) for i in range(len(y))]),y[-1],str(tile2pos(y[-1]))))

