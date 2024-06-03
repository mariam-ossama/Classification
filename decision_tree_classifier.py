import numpy as np
from collections import Counter

def entropy(y):
    hist = np.bincount(y)  # Function counts the occurrences of each label in the array y.
    ps = hist / len(y)   # Computes the probability of each label occurring.
    # The entropy formula is applied using a list comprehension to avoid taking the logarithm of zero probabilities.
    # The result is the sum of the negative logarithms of probabilities, representing the entropy.
    return -np.sum([p * np.log2(p) for p in ps if p > 0])
# This class represents a node in the decision tree.
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature # Index of the feature used for splitting at this node.
        self.threshold = threshold  # Value of the feature used for splitting.
        self.left = left
        self.right = right
        self.value = value  # If the node is a leaf node, it stores the predicted class label.

     # Returns True if the node is a leaf node or it has a non-null value.
    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)



    '''
    Recursively builds the decision tree by finding the best split at each node based on information gain.
    It stops growing the tree if the maximum depth is reached or if all labels at a node are the same.
    '''
    def _grow_tree(self, X, y, depth=0):
        if depth == self.max_depth or len(set(y)) == 1:  # Stopping condition
            return Node(value=np.bincount(y).argmax())  # Create a leaf node

        # Obtains the number of samples and features from the input data.
        n_samples, n_features = X.shape
        best_gain = 0
        best_criteria = None
        best_sets = None

        for feature_idx in range(n_features): # Iterating over each feature in the dataset
            feature_values = np.unique(X[:, feature_idx])
            for value in feature_values: # Iterating over each unique value in the current feature
                # Calculates the indices for the left and right splits based on the current feature value
                left_indices = np.where(X[:, feature_idx] <= value)[0]  # If smaller->left
                right_indices = np.where(X[:, feature_idx] > value)[0]  # If greater->right
                info_gain = self._information_gain(y, left_indices, right_indices)
                if info_gain > best_gain:
                    best_gain = info_gain
                    best_criteria = (feature_idx, value)
                    best_sets = (left_indices, right_indices)

        if best_gain == 0:
            return Node(value=np.bincount(y).argmax())  # Create a leaf node


        # Depth is incremented by 1 for each recursive call to track the depth of the tree.
        left_tree = self._grow_tree(X[best_sets[0]], y[best_sets[0]], depth + 1)
        right_tree = self._grow_tree(X[best_sets[1]], y[best_sets[1]], depth + 1)

        '''
        Returns a new node representing the current split, storing the best criteria and 
        references to the left and right subtrees.
        '''
        return Node(best_criteria[0], best_criteria[1], left_tree, right_tree)


    '''
    Calculates the information gain achieved by a split using the entropy before and after the split.
    '''
    def _information_gain(self, y, left_indices, right_indices):
        # Calculate entropy before split
        entropy_before = entropy(y)

        # Calculate entropy after split
        n = len(y)
        n_left = len(left_indices)
        n_right = len(right_indices)
        entropy_after = (n_left / n) * entropy(y[left_indices]) + \
                        (n_right / n) * entropy(y[right_indices])

        # Calculate information gain
        information_gain = entropy_before - entropy_after
        return information_gain

    def predict(self, X):
        return np.array([self._predict(x, self.tree) for x in X])





    '''
    Simply it traverses the tree to reach the leaf node that carry the value to be the predicted value
    '''
    def _predict(self, x, node):
        if node.is_leaf_node():
            return node.value  # Return the predicted value for leaf nodes

        if x[node.feature] <= node.threshold:
            return self._predict(x, node.left)
        else:
            return self._predict(x, node.right)


