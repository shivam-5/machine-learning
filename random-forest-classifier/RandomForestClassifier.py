import numpy as np
import pandas as pd


# Random Forest Classifier
# Consists of multiple trees specified by @n_trees
# Implements Bootstrapping as well as feature bagging techniques to generalize over all trees
class RandomForestClassifier:
    # Holder for all trees in the forest
    trees = []

    def __init__(self, n_trees):
        self.n_trees = n_trees
        pass

    # Set the params for all trees that will be created in the forest
    # @param min_instances - specifies minimum number of instances required to split a node
    # #param max_instances - specifies maximum number of instances to be used in each tree (bootstrapping)
    # @param min_impurity - specifies minimum impurity required to split a node
    # @param max_depth - specifies max depth of all trees, default:None
    # @param max_features - specifies maximum number of features to be randomly selected for each tree(feature bagging)
    def set_tree_params(self, min_instances=5, max_instances=1000, min_impurity=0.001,
                        max_depth=None, max_features=None):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.max_features = max_features

    def fit(self, data):
        for i in range(0, self.n_trees):
            tree = DecisionTree(self.min_instances, self.min_impurity, self.max_depth)

            # Feature Bagging
            if self.max_features == None:
                # Take square root of total number of features as max_features if None specified
                self.max_features = int(np.sqrt(len(data.columns[:-1])))
            # Creates a sliced dataset randomly selecting maximum number of allowed features
            # Process: Drop target column, select sample from all features, add target column to new dataset
            sliced_features = data.drop(data.columns[-1], axis=1).sample(n=self.max_features, axis='columns')
            sliced_features[data.columns[-1]] = data[data.columns[-1]]

            # Bootstrapping Slice the dataset to randomly select(with replacement) max_instances allowed
            # or select all whichever is less
            sliced_features = sliced_features.sample(np.min([sliced_features.shape[0], self.max_instances]),
                                                     replace=True)
            tree.fit(sliced_features)
            self.trees.append(tree)

    def predict(self, data):
        # Create pandas dataframe with columns names for the test dataset
        forest_labels = pd.DataFrame(columns=[ii + 1 for ii in range(data.shape[0])])

        # Get predictions from individual trees and store in forest_labels
        for i in range(0, self.n_trees):
            labels = self.trees[i].predict(data).values
            forest_labels.loc[i, :] = labels

        # Return predicted label for each test instance sorted with frequency(also seen as priority)
        predicted_labels = list()
        for column in forest_labels.columns:
            # After applying value_counts, indexes are actual labels that needs to be returned
            # while values are the frequencies that can be ignored since we have the sorted list now
            predicted_labels.append(
                forest_labels[column].value_counts(sort=True).index.astype('int64').tolist()
            )
        return predicted_labels

    # Describes all the trees in forest
    def describe_forest(self):
        for i in range(len(self.trees)):
            print("Tree number: %d" % (i + 1))
            self.trees[i].describe_tree()
            print("\n")


# Decision Tree as part of Random Forest Classifier
# Can also be used as an independent classifier
class DecisionTree:
    root = None
    class_name = None

    def __init__(self, min_instances=2, min_impurity=0.01, max_depth=None):
        self.min_instances = min_instances
        self.min_impurity = min_impurity
        self.max_depth = max_depth

    def fit(self, data):
        # Extract feature names that will be used to split nodes
        # Store target class name and start building the tree
        column_values = data.columns.values
        feature_names = column_values[:-1].tolist()
        self.class_name = column_values[-1]
        self.root = self.build_tree(data, feature_names, depth=0)

    def predict(self, data):
        # Return label for each instance in test set
        return data.apply(self.predict_label, axis=1)

    def predict_label(self, instance):
        node = self.root
        # Traverse down the tree from root feature
        while len(node.children) != 0:
            # Get value of the feature in the instance
            feature_value = instance[node.feature_name]
            if feature_value in node.children.keys():
                # Traverse down that value to repeat same thing at next feature node
                node = node.children[feature_value]
            else:
                # Return class of current feature if value of the feature doesn't exist as a child split
                return node.label

        # Return label of current node if current node is leaf (no further split)
        return node.label

    # Recursive code to build each subtree tree and return root
    def build_tree(self, data, features, depth, label=None, parent_impurity=1):
        current = Node()

        # Add information regarding this node
        current.instances = data.shape[0]   # Number of remaining instances
        current.label = label               # Predicted label for this node
        current.depth = depth               # Depth of current node in tree

        # If minimum number of instances doesn't satisfy
        # or there are no more features reaming to split
        # or the tree has reached max depth allowed (if specified)
        # make it a leaf node
        if data.shape[0] < self.min_instances or len(features) == 0 \
                or (depth == self.max_depth and self.max_depth is not None):
            current.is_leaf = True
        else:
            # Get the next best feature split
            impurity_split, attribute_split, attr_value_tuples = self.get_attribute_split(data, features)

            # Accept the split only if new impurity is less than parent's impurity
            # and greater than threshold for minimum impurity
            # Else make it a leaf node
            if parent_impurity > impurity_split > self.min_impurity:
                # Add the new available data to node
                current.feature_name = attribute_split  # Feature that is split at this node
                current.impurity = impurity_split  # Impurity calculated at this node

                # Remove the selected feature and look for further children split for each of its values
                features.remove(attribute_split)
                for attr_tuple in attr_value_tuples:
                    key = attr_tuple[0]             # Feature value
                    max_label = attr_tuple[2]       # Most frequent class label for current feature value

                    # Split the instances containing from current set
                    # that has current feature value as key
                    df = data[data[attribute_split] == key]
                    sub_split = self.build_tree(df, features, depth + 1, max_label, current.impurity)

                    # If the child split is a leaf node and has same label as current feature node
                    # then remove the redundant child
                    # Else add it to list of children
                    if sub_split.is_leaf and sub_split.label == current.label:
                        sub_split = None
                    else:
                        current.children[key] = sub_split
            else:
                current.is_leaf = True
        return current

    # Find the next best feature split among all features
    # Accepts dataframe object with all data and list of features as parameters
    # Returns best feature to split based on gini impurity
    def get_attribute_split(self, df, features):
        split_attr = None
        split_impurity = 1
        attr_value_tuple = None

        # Get the gini impurity corresponding to each feature
        for attr in features:
            impurity, tuple = self.get_gini_impurity(df, attr)
            if impurity < split_impurity:
                # Store the info corresponding to feature with lowest impurity
                split_impurity, split_attr, attr_value_tuple = impurity, attr, tuple
        return split_impurity, split_attr, attr_value_tuple

    # Calculates gini impurity corresponding to a feature
    # Accepts dataframe object with all data and feature to calculate gini impurity of as parameters
    # Returns value of gini impurity for the feature
    def get_gini_impurity(self, df, feature):
        # Get the frequency of each value in the feature
        attr_values = df[feature].value_counts()
        impurity = 0
        n = df.shape[0] * 1.0  # convert to float

        # Each tuple will store feature value, its count and its most frequent class label
        attr_value_tuples = list()
        for key in attr_values.keys():
            # Get frequency of all class label corresponding to current value of the feature
            class_freq = pd.value_counts(df[self.class_name][df[feature] == key].values.flatten(), normalize=True)

            # Apply probability formula to calculate gini
            gini = 1 - np.sum(class_freq.values ** 2)
            attr_count = attr_values[key]

            # Add to weighted average over all feature values
            impurity = impurity + (attr_count / n) * gini

            # Add current value of feature, its count and the most frequent class label
            attr_value_tuples.append((key, attr_values[key], class_freq.idxmax()))
        return impurity, attr_value_tuples

    # Prints the tree in Breadth First Search traversal
    def describe_tree(self):
        nodes = [self.root]
        while len(nodes) != 0:
            node = nodes.pop(0)
            node.describe()
            for child in node.children.values():
                nodes.append(child)


# Creates a Node to be used in building a tree
class Node:
    def __init__(self):
        self.children = dict()      # Stores branches going off from current node
        self.feature_name = ""      # Stores feature name corresponding to current node
        self.label = ""             # Stores class label corresponding to current node
        self.impurity = -1          # Stores gini impurity corresponding to current node
        self.instances = -1         # Stores number of instances present in current node's subtree
        self.depth = None           # Stores depth of tree at current node
        self.is_leaf = False        # Use to identify internal or leaf node

    # Prints the node information in a nice readable way
    def describe(self):
        if self.is_leaf:
            print("Class: %s, Instances: %d, Depth: %d " % (self.label, self.instances, self.depth))
        else:
            print("Feature name: %s, Class: %s Impurity: %.3f, Instances: %d, Depth: %d "
                  % (self.feature_name, self.label, self.impurity, self.instances, self.depth))
            for child in self.children.values():
                if child.is_leaf:
                    print("child: %s" % str(child.label))
                else:
                    print("child: %s" % str(child.feature_name))
