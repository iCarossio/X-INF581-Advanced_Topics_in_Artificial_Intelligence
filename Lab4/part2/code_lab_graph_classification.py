import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from grakel.kernels import ShortestPath, PyramidMatch, RandomWalk, VertexHistogram, WeisfeilerLehman
from grakel import graph_from_networkx
from grakel.datasets import fetch_dataset
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


############## Question 1
# Generate simple dataset

Gs = list()
y = list()

##################
# your code here #
##################


############## Question 2
# Classify the synthetic graphs using graph kernels

# Split dataset into a training and a test set
# hint: use the train_test_split function of scikit-learn

##################
# your code here #
##################

# Transform NetworkX graphs to objects that can be processed by GraKeL
G_train = list(graph_from_networkx(G_train))
G_test = list(graph_from_networkx(G_test))


# Use the shortest path kernel to generate the two kernel matrices ("K_train" and "K_test")
# hint: the graphs do not contain node labels. Set the with_labels argument of the the shortest path kernel to False

##################
# your code here #
##################


clf = SVC(kernel='precomputed', C=1) # Initialize SVM
clf.fit(K_train, y_train) # Train SVM
y_pred = clf.predict(K_test) # Predict

# Compute the classification accuracy
# hint: use the accuracy_score function of scikit-learn

##################
# your code here #
##################


# Use the random walk kernel and the pyramid match graph kernel to perform classification

##################
# your code here #
##################


############## Question 3
# Classify the graphs of a real-world dataset using graph kernels

# Load the MUTAG dataset
# hint: use the fetch_dataset function of GraKeL

##################
# your code here #
##################


# Split dataset into a training and a test set
# hint: use the train_test_split function of scikit-learn

##################
# your code here #
##################


# Perform graph classification using different kernels and evaluate performance

##################
# your code here #
##################
