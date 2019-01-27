from numpy import argsort
from sklearn.externals import joblib

# Load labels
labels = joblib.load('labels.dat')
# Load vectorizor
f = joblib.load('f.dat')
# Load model
h = joblib.load('h.dat')
# Load plot summary
my_plot_summary = open('my_summary.txt', 'r').read()

# Test ...
xtest = f.transform([my_plot_summary]).toarray()
ppred = h.predict_proba(xtest)
for j in argsort(-ppred[0,:]):
    print("p(%s|x) = %4.3f" % (labels[j],ppred[0,j]))
