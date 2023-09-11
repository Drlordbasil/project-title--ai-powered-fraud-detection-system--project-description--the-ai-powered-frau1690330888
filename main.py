from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def read_transaction_data(file_paths):
    return pd.concat([pd.read_csv(file_path) for file_path in file_paths])


def preprocess_data(transaction_data):
    # Code for cleansing, preprocessing, and normalizing the data goes here
    return transaction_data


def split_data(transaction_data):
    X = transaction_data.drop("fraudulent", axis=1)
    y = transaction_data["fraudulent"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def train_random_forest_classifier(X_train_scaled, y_train):
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train_scaled, y_train)
    return rf_classifier


def train_one_class_svm(X_train_scaled, y_train):
    svm_classifier = OneClassSVM()
    svm_classifier.fit(X_train_scaled[y_train == 0])
    return svm_classifier


def preprocess_incoming_data(incoming_transactions, scaler):
    # Code for preprocessing the incoming data goes here
    return scaler.transform(incoming_transactions)


def calculate_risk_score(transactions):
    # Code for calculating risk scores goes here
    return transactions


def predict_labels(classifier, incoming_transactions_scaled):
    return classifier.predict(incoming_transactions_scaled)


def visualize_results(suspicious_transactions):
