from numpy import * 
from utils import *
import pandas as pd
set_printoptions(precision=2)

########################
# DATA PREPARATION
########################

print("Load Data")

#df = pd.read_csv("music.csv")
#L = 6

df = pd.read_csv("imdb_vectorized.csv.gz")
L = 28

labels = array(df.columns.values.tolist())[0:L]
data = df.values

print("Split Data")
N = int(len(data)*2/3)

Y = data[:,0:L].astype(float)
X = data[:,L:].astype(float)

########################
# EXPLORATION
########################

from matplotlib.pyplot import *

print("Task 1. Exploration")

from utils import count_combinations

# TODO 

# HINT (plot 1): 
#label_freqs = ...
#figure()
#plot(range(L),-sort(-label_freqs))
#xticks(range(L), labels[argsort(-label_freqs)], rotation=45)
#xlabel("Labels")
#ylabel("Frequency")
#show()

# HINT (plot 2): 
#figure()
#xlabel("Label combinations")
#ylabel("Frequency")
# (log scale on y axis)
#semilogy(range(unique_combinations), -sort(-array(list(combinations_freq.values()))))
#savefig("frec_"+str(L)+".pdf")

print("Task 2. Marginal Dependence")

# TODO

# Indices of the top 10 labels
idx = argsort(-sum(Y,axis=0))[0:min(10,L)]
#S = ...
#fig = make_heatmap(S,labels[idx])
#show()

# TODO

#figure()
#G = make_graph(S,col_names=labels[idx])
#axis('off')

print("Task 3. Conditional Dependence")

from binary_relevance import BR
#from sklearn.tree import DecisionTreeClassifier as BaseModel
from sklearn.linear_model import LogisticRegression as BaseModel

E = get_errors(BR(BaseModel()),X,Y,2)

# TODO 


########################
# CLASSIFICATION
########################

print("Task 4. Evaluate LP vs BR (hamming loss, 0/1 loss, running time.")

from time import clock

# TODO
# temp = clock()
# ..
# ..
# print("Time   ", clock()-temp)
# ..


print("Task 5. Build model on *all* the data, and save to disk.")

#from sklearn.externals import joblib

# TODO

#joblib.dump(classifier, 'h.dat')

