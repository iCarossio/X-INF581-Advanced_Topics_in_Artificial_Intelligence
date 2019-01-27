from numpy import *
import pandas as pd

##########################################################
print("Load Raw Data ...")

data = pd.read_csv('imdb.csv.gz', encoding = "ISO-8859-1")

# (We won't care about the title)
del data['TITLE'] 
N,LD = data.shape
L = LD - 1

# Labels
Y = data.iloc[:,0:L].as_matrix().astype(int)
lbls = data.columns.values.tolist()

# Summaries
text = data["SUMMARY"]                               

print("\tLoaded Text = %d x 1  and Y = %d x %d" % (N,N,L))

##########################################################
print("Process Data; Create Bag-of-Words Feature Space ...")

from sklearn.feature_extraction.text import CountVectorizer

f = CountVectorizer(encoding = "ISO-8859-1", decode_error='ignore', max_df=0.95, min_df=2, max_features=1000, stop_words='english')

X = f.fit_transform(text).toarray()
N,D = X.shape

##########################################################
print("Stacking Together, Adding Header ...")
XY = column_stack([Y,X])
header = ','.join(lbls[0:-1])
words = ['' for i in range(D)]
for k in f.vocabulary_.keys():
    i = f.vocabulary_[k]
    words[i] = 'w_'+k
header = header + ',' + ','.join(words)
#print(header)

##########################################################
print("Writing out Vectorizor and Labels ...")

from sklearn.externals import joblib
joblib.dump(lbls, 'labels.dat')
joblib.dump(f, 'f.dat')

##########################################################
print("Writing out Data ...")
#savetxt('imdb_vectorized.csv', XY[0:50000], fmt='%d', delimiter=',', comments='', header=header)
savetxt('imdb_vectorized.csv.gz', XY, fmt='%d', delimiter=',', comments='', header=header)
