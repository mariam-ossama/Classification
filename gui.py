import tkinter as tk
from tkinter import filedialog
import pandas as pd
from data_preprocessing import DataPreprocessor, train_test_split
from bayesian_classifier import NaiveBayesClassifier
from decision_tree_classifier import DecisionTreeClassifier
import numpy as np

class GUI:
    def __init__(self, root):
        self.root = root
        self.file_path = None
        self.percentage = None
        root.title("Classification")

        self.file_label = tk.Label(root, text="Select CSV file:")
        self.file_label.pack()

        self.file_path_text = tk.Entry(root,width=30)
        self.file_path_text.pack()

        self.file_button = tk.Button(root, text="Browse", command=self.browse_file, width=20)
        self.file_button.pack()

        self.percentage_label = tk.Label(root, text="Enter percentage:")
        self.percentage_label.pack()

        self.percentage_entry = tk.Entry(root, width= 30)
        self.percentage_entry.pack()

        self.submit_button = tk.Button(root, text="Submit", command=self.process_data, width=20)
        self.submit_button.pack()

        self.result_frame = tk.Frame(root)
        self.result_frame.pack()

    def browse_file(self):
        self.file_path = filedialog.askopenfilename()
        self.file_path_text.delete(0, tk.END)
        self.file_path_text.insert(tk.END, self.file_path)

    def process_data(self):
        if self.file_path is None:
            print("Please select a CSV file.")
            return

        self.percentage = float(self.percentage_entry.get()) / 100

        data = pd.read_csv(self.file_path)
        total_records = len(data)

        rows_to_read = int(total_records * self.percentage)
        data = pd.read_csv(self.file_path, nrows=rows_to_read)

        preprocessor = DataPreprocessor()
        preprocessor.fit(data)
        preprocessed_data = preprocessor.transform(data)

        X_train, X_test, y_train, y_test = train_test_split(preprocessed_data, test_size=self.percentage)

        print("Total number of records in the CSV file:", total_records)
        print("Total number of records read:", rows_to_read)
        print("Total number of records assigned as training set:", len(X_train))
        print("Total number of records assigned as testing set:", len(X_test))

        nb_classifier = NaiveBayesClassifier()
        nb_classifier.fit(X_train, y_train)
        nb_predictions = nb_classifier.predict(X_test)
        nb_accuracy = nb_classifier.accuracy(X_test, y_test)

        decision_tree = DecisionTreeClassifier(max_depth=10)
        decision_tree.fit(X_train, y_train)
        dt_predictions = decision_tree.predict(X_test)

        # Calculate accuracy of decision tree predictions
        dt_accuracy = np.mean(dt_predictions == y_test)

        print("\nNaive Bayes Classifier Accuracy:", nb_accuracy * 100)
        print("Decision Tree Classifier Accuracy:", dt_accuracy * 100)

        self.display_results(nb_predictions, nb_accuracy, dt_predictions, dt_accuracy,y_test, y_test)

    def display_results(self, nb_predictions, nb_accuracy, dt_predictions, dt_accuracy, y_test_nb, y_test_dt):
        # Frame for Naive Bayes Classifier Results
        nb_frame = tk.Frame(self.result_frame)
        nb_frame.pack()

        nb_label = tk.Label(nb_frame, text="Naive Bayes Classifier Results:")
        nb_label.pack()

        nb_scrollbar = tk.Scrollbar(nb_frame)
        nb_scrollbar.pack(side="right", fill="y")

        nb_result_text = tk.Text(nb_frame, yscrollcommand=nb_scrollbar.set)
        nb_result_text.pack(side="left", fill="both", expand=True)

        nb_misclassified_indices = np.where(nb_predictions != y_test_nb)[0]
        nb_misclassified_count = len(nb_misclassified_indices)

        for i, prediction in enumerate(nb_predictions):
            risk_indication = "1 (at a risk of developing diabetes)" if prediction == 1 else "0 (not at risk of developing diabetes)"
            nb_result_text.insert(tk.END, f"Patient {i + 1} -> {prediction} ({risk_indication})\n")

        nb_result_text.insert(tk.END, f"NB Accuracy: {nb_accuracy * 100}\n")
        nb_result_text.insert(tk.END, f"Number of misclassified records: {nb_misclassified_count}\n\n")
        nb_scrollbar.config(command=nb_result_text.yview)

        # Frame for Decision Tree Classifier Results
        dt_frame = tk.Frame(self.result_frame)
        dt_frame.pack()

        dt_label = tk.Label(dt_frame, text="Decision Tree Classifier Results:")
        dt_label.pack()

        dt_scrollbar = tk.Scrollbar(dt_frame)
        dt_scrollbar.pack(side="right", fill="y")

        dt_result_text = tk.Text(dt_frame, yscrollcommand=dt_scrollbar.set)
        dt_result_text.pack(side="left", fill="both", expand=True)

        dt_misclassified_indices = np.where(dt_predictions != y_test_dt)[0]
        dt_misclassified_count = len(dt_misclassified_indices)

        for i, prediction in enumerate(dt_predictions):
            risk_indication = "1 (at a risk of developing diabetes)" if prediction == 1 else "0 (not at risk of developing diabetes)"
            dt_result_text.insert(tk.END, f"Patient {i + 1} -> {prediction} ({risk_indication})\n")

        dt_result_text.insert(tk.END, f"Decision Tree Accuracy: {dt_accuracy * 100}\n")
        dt_result_text.insert(tk.END, f"Number of misclassified records: {dt_misclassified_count}\n")
        dt_scrollbar.config(command=dt_result_text.yview)

        # Print misclassified records in console
        print("Misclassified Records by Naive Bayes Classifier:")
        for idx in nb_misclassified_indices:
            actual_label = y_test_nb[idx]
            predicted_label = nb_predictions[idx]
            print(f"Patient {idx + 1} -> Actual: {actual_label}, Predicted: {predicted_label}")

        print("\nMisclassified Records by Decision Tree Classifier:")
        for idx in dt_misclassified_indices:
            actual_label = y_test_dt[idx]
            predicted_label = dt_predictions[idx]
            print(f"Patient {idx + 1} -> Actual: {actual_label}, Predicted: {predicted_label}")


