#import pandas as pd
import tkinter as tk

from data_preprocessing import DataPreprocessor,train_test_split
from bayesian_classifier import *
from decision_tree_classifier import *

from gui import GUI



def print_labels_with_risk_indications(y_labels):
    for i, label in enumerate(y_labels):
        risk_indication = "1 (at a risk of developing diabetes)" if label == 1 else "0 (not at risk of developing diabetes)"
        print(f"Patient {i + 1} -> {label} ({risk_indication})")




if __name__ == '__main__':
    '''file_path = input("Enter the path of the CSV file: ")
    data = pd.read_csv(file_path)

    total_rows_in_csv = len(data)
    print(f'Total number of rows in the CSV file: {total_rows_in_csv}')

    percentage = float(input("Enter the percentage of the dataset to be used: ")) / 100
    rows_to_read = int(len(data) * percentage)

    data = pd.read_csv(file_path, nrows=rows_to_read)

    total_rows_read = len(data)
    print(f'Total number of rows read from the CSV file: {total_rows_read}')

    preprocessor = DataPreprocessor()
    preprocessor.fit(data)
    preprocessed_data = preprocessor.transform(data)

    X_train, X_test, y_train, y_test = train_test_split(preprocessed_data, test_size=percentage)

    print(f'Total number of rows assigned for training set: {len(X_train)}')
    print(f'Total number of rows assigned for testing set: {len(X_test)}')

    naiveBayes = NaiveBayesClassifier()
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.fit(X_train, y_train)

    predictions = nb_classifier.predict(X_test)
    print("NB predictions:\n", predictions)
    print("Number of NB predictions:", len(predictions))  # Print number of NB predictions
    print("Labels with risk indications for Naive Bayes Classifier:")
    print_labels_with_risk_indications(y_test)

    accuracy_nb = nb_classifier.accuracy(X_test, y_test)
    print("NB Accuracy:", accuracy_nb * 100)

    X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(preprocessed_data, test_size=percentage)

    print("Shapes of X_train_dt and y_train_dt:", X_train_dt.shape, y_train_dt.shape)  # Debugging statement

    decision_tree = DecisionTreeClassifier(max_depth=10)
    decision_tree.fit(X_train_dt, y_train_dt)

    predictions_dt = decision_tree.predict(X_test_dt)
    print("Decision Tree predictions:\n", predictions_dt)
    print("Number of Decision Tree predictions:", len(predictions_dt))  # Print number of Decision Tree predictions
    print("Labels with risk indications for Decision Tree Classifier:")
    print_labels_with_risk_indications(y_test_dt)

    correct_predictions = np.sum(predictions_dt == y_test_dt)
    total_predictions = len(y_test_dt)
    accuracy_dt = correct_predictions / total_predictions
    print("Decision Tree Accuracy:", accuracy_dt * 100)'''

    root = tk.Tk()
    gui = GUI(root)
    root.mainloop()