from sklearn.naive_bayes import GaussianNB
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split

#
# Define constants
#

datafilename = 'processed.cleveland.data' # input filename
age      = 0                              # column indexes in input file
sex      = 1
cp       = 2
trestbps = 3
chol     = 4
fbs      = 5
restecg  = 6
thalach  = 7
exang    = 8
oldpeak  = 9
slope    = 10
ca       = 11
thal     = 12
num      = 14 # this is the thing we are trying to predict

# Since feature names are not in the data file, we code them here
feature_names = [ 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num' ]

num_samples = 303 # size of the data file.
num_features = 13

#
# Open and read data file in csv format
#
# After processing:
#
# data   is the variable holding the features;
# target is the variable holding the class labels.

try:
    with open( datafilename ) as infile:
        indata = csv.reader( infile )
        data = np.empty(( num_samples, num_features))
        target = np.empty(( num_samples,), dtype=np.int )
        i = 0
        for j, d in enumerate( indata ):
            ok = True
            for k in range(0, num_features): # If a feature has a missing value
                if ( d[k] == "?"):         # we do't use that record.
                    ok = False
            if ok is True:
                data[i] = np.asarray( d[0:-1], dtype=np.float64 )
                target[i] = np.asarray( d[-1], dtype=np.int )
                i = i + 1
except IOError as iox:
    print 'there was an I/O error trying to open the data file: ' + str( iox )
    sys.exit()
except Exception as x:
    print 'there was an error: ' + str( x )
    sys.exit()

# How many records do we have?
num_samples = i
print "Number of samples:", num_samples

data = data[0:i, :]
target = target[0:i]

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_train =gnb.predict(X_train)
y_pred_test =gnb.predict(X_test)
print("accuracy train: %.2f" % ((y_train != y_pred_train).sum() / float(X_train.shape[0])))
print("accuracy test: %.2f" % ((y_test != y_pred_test).sum() / float(X_test.shape[0])))