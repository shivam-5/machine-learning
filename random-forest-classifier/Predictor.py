import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.datasets import load_files
from sklearn.tree import DecisionTreeClassifier
from RandomForestClassifier import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def convertToArray(numberString):
    numberArray = []
    for i in range(len(numberString) - 1):
        if numberString[i] == '0':
            numberArray.append(0)
        elif numberString[i] == '1':
            numberArray.append(1)
        elif numberString[i] == '2':
            numberArray.append(2)
        elif numberString[i] == '3':
            numberArray.append(3)
        elif numberString[i] == '4':
            numberArray.append(4)

    return numberArray


# open datafile, extract content into an array, and close.
datafile = open('./data/custom-moves.txt', 'r')
content = datafile.readlines()
datafile.close()

# Now extract data, which is in the form of strings, into an
# array of numbers, and separate into matched data and target
# variables.
data = []
target = []
# Turn content into nested lists
for i in range(len(content)):
    lineAsArray = convertToArray(content[i])
    dataline = []
    for j in range(len(lineAsArray) - 1):
        dataline.append(lineAsArray[j])

    data.append(dataline)
    targetIndex = len(lineAsArray) - 1
    target.append(lineAsArray[targetIndex])

(X_train, X_test, y_train, y_test) = train_test_split(data, target, test_size=0.1)

attribute_names = ['c' + str(x) for x in range(1, len(data[0]) + 1)]
class_name = ['target']
train_data = pd.DataFrame(np.c_[X_train, y_train], columns=attribute_names + class_name)
test_data = pd.DataFrame(np.c_[X_test, y_test], columns=attribute_names + class_name)

classifier = RandomForestClassifier(n_trees=100)
classifier.set_tree_params(min_instances=2, min_impurity=0.001, max_depth=4, max_features=9)
classifier.fit(train_data)

predicted_list = classifier.predict(test_data)
y_hat = [label[0] for label in predicted_list]
print(accuracy_score(y_test, y_hat))
