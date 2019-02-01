from numpy import *
from cc import CC
from utils import print_ptree

# Parameters
random.seed(0)    # <--- May be changed for grading
eps = 0.3         # <--- May be changed for grading

# Load a dataset, shuffle and split it.
L = 6
XY = genfromtxt('Music.csv', skip_header=1, delimiter=",")
random.shuffle(XY)
N,DL = XY.shape
X = XY[:,L:DL]
Y = XY[:,0:L]
N_train = N-10
X_test = X[N_train:]
Y_test = Y[N_train:]

# Instantiate and train a classifier chain 
# (except for 10 examples -- our test set)
cc = CC()
cc.fit(X[0:N_train], Y[0:N_train])

# Obtain predictions, and all paths explored to obtain each prediction
Y_pred = cc.predict(X_test,epsilon=eps)
E = (Y_pred != Y_test) * 1
print("\nResults:")
print("Loss per test instance: ", E.sum(axis=1), "; 0/1 loss: ", sum(E.sum(axis=1)>0)/10.)

# For a particular example, we get 
# * paths: the paths explored 
# * y_argmax: the most likely path 
# * y_max: the value of the most likely path
print("\nBranches:")
branches,y_argmax,y_max = cc.explore_paths(X_test[0],epsilon=eps)

# Print the branches
for (y_parent,y_child,prob_branch,prob_path) in branches:
    print("From %s to %s with cost %3.2f (path-cost so far: %3.2f)" % (str(y_parent),str(y_child),prob_branch,prob_path))

# Save the paths to a file in in a format that we can draw easily later
print_ptree(branches,"probability_tree_exploration.dot",y_best=y_argmax.tolist())
