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
num_features = 14

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
                data[i] = np.asarray( d, dtype=np.float64 )
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

# Here is are the sets of features:
X_train = pd.DataFrame(X_train, columns=feature_names.__getslice__(0, num_features))
X_test = pd.DataFrame(X_test, columns=feature_names.__getslice__(0, num_features))
# Here is the diagnosis for each set of features:
# target = pd.DataFrame(target, columns=[feature_names[-1]])

relevant_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca']

# Model building
prior = X_train.groupby('num').size().div(num_samples)

likelihood = {}
for feature in relevant_features:
    group = X_train.groupby(['num', feature]).size().div(num_samples).div(prior)
    attributes = {}
    for tuple in group.index:
        attributes[tuple] = group[tuple]
    likelihood[feature] = attributes


# Predicting
correct = 0
for index, row in X_test.iterrows():
    max_likelihood = -1
    likely_output = -1
    prob = 1
    for num in prior.index:
        prob =prior[num]
        for feature in relevant_features:
            attr = likelihood[feature]
            if (num, row[feature]) not in attr:
                like = 0
            else:
                like = like = attr[(num, row[feature])]
            prob = prob * like
        if prob > max_likelihood:
            max_likelihood = prob
            likely_output = num

    if likely_output == row['num']:
        correct += 1

print "Naive Bayes accuracy: %.2f" % (float(correct) / X_test.shape[0])


