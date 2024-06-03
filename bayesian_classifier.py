import numpy as np
import pandas as pd


class NaiveBayesClassifier:
    def __init__(self):
        self.class_probabilities = None
        '''
        This is a dictionary where the keys are feature indices, and 
        the values are dictionaries containing probabilities of each feature value for each class.

        '''
        self.feature_probabilities = {}
        '''
     used to train the Naive Bayes classifier. It calculates
     the class and feature probabilities based on the input data (X) and labels (y).
     It iterates over each feature and each class to compute these probabilities.
     '''
    def fit(self, X, y):
        # Calculate class probabilities(label)
        self.class_probabilities = {}
        total_samples = len(y)
        for label in np.unique(y):
            self.class_probabilities[label] = np.sum(y == label) / total_samples

        # Calculate feature probabilities
        for feature_index in range(X.shape[1]):
            feature_values = np.unique(X[:, feature_index])
            self.feature_probabilities[feature_index] = {}
            for feature_value in feature_values:
                self.feature_probabilities[feature_index][feature_value] = {}
                for label in np.unique(y):
                    mask = y == label
                    feature_count = np.sum(X[mask, feature_index] == feature_value)
                    self.feature_probabilities[feature_index][feature_value][label] = feature_count / np.sum(mask)



    '''
    The predict method takes input data (X) and returns predictions for each sample in the dataset.
    It calculates the probability of each class for each sample using the Naive Bayes formula.
    It selects the class with the highest probability as the predicted class for each sample.
    '''
    def predict(self, X):
        predictions = []
        for sample in X:
            max_probability = -1 # Since after computing the probability we apply the comparison so it should be inintialized by -1
            predicted_class = None
            for label in self.class_probabilities:
                # Initialize with prior probability
                probability = self.class_probabilities[label]
                for feature_index, feature_value in enumerate(sample):
                    if feature_value in self.feature_probabilities[feature_index]:
                        probability *= self.feature_probabilities[feature_index][feature_value][label]
                if probability > max_probability:
                    max_probability = probability
                    predicted_class = label
            predictions.append(predicted_class)
        return np.array(predictions)

    def accuracy(self, X, y):
        predictions = self.predict(X)
        correct_predictions = np.sum(predictions == y)
        total_samples = len(y)
        accuracy = correct_predictions / total_samples
        return accuracy


