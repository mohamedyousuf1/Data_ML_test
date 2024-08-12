import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

"""
ModelTrainer class for training and evaluating various machine learning models.
Attributes:
    X_train (numpy.ndarray): Training data features.
    X_test (numpy.ndarray): Testing data features.
    y_train (numpy.ndarray): Training data labels.
    y_test (numpy.ndarray): Testing data labels.
    models (dict): Dictionary of machine learning models and their corresponding hyperparameters.
Methods:
    _build_and_train_neural_network(params): Builds and trains a neural network model.
    train_and_evaluate(): Trains and evaluates all the machine learning models and the neural network.
Returns:
    list: List of dictionaries containing the evaluation results for each model.
"""
class ModelTrainer:
    def __init__(self, X, y, csv_results='model_comparison_results.csv'):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.models = {
            'Logistic Regression': (LogisticRegression(), {'C': [0.01, 0.1, 1, 10]}),
            'Decision Tree': (DecisionTreeClassifier(), {'max_depth': [5, 10, 15]}),
            'Random Forest': (RandomForestClassifier(), {'n_estimators': [50, 100, 150]}),
            'Support Vector Machine': (SVC(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
            'k-Nearest Neighbors': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
            'Naive Bayes': (GaussianNB(), {}),
            'Gradient Boosting': (GradientBoostingClassifier(), {'n_estimators': [50, 100, 150]}),
            'XGBoost': (xgb.XGBClassifier(), {'n_estimators': [50, 100, 150]}),
            'LightGBM': (lgb.LGBMClassifier(), {'n_estimators': [50, 100, 150]})
        }
        self.csv_results = csv_results # File to save results
        
    def _build_and_train_neural_network(self, params):
        model = Sequential()
        model.add(Dense(64, input_dim=self.X_train.shape[1], activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        history = model.fit(self.X_train, self.y_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
        predictions = (model.predict(self.X_test) > 0.5).astype("int32")
        y_pred = predictions.flatten()
        
        return y_pred

    # Train and evaluate all models
    def train_and_evaluate(self):
        results = []
        neural_net_params = {'epochs': [50, 100], 'batch_size': [10, 20]}
        
        for name, (model, params) in self.models.items():
            print(f"Training {name}...")
            grid = GridSearchCV(estimator=model, param_grid=params, scoring='f1', cv=3)
            grid_result = grid.fit(self.X_train, self.y_train)
            best_model = grid_result.best_estimator_
            y_pred = best_model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            cm = confusion_matrix(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred, output_dict=True)

            results.append({
                'Model': name,
                'Best Params': grid_result.best_params_,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'Confusion Matrix': cm.tolist(),
                # 'Classification Report': report
            })
        
        # Training and evaluating the neural network separately
        for epochs in neural_net_params['epochs']:
            for batch_size in neural_net_params['batch_size']:
                print(f"Training Neural Network with epochs={epochs} and batch_size={batch_size}...")
                y_pred = self._build_and_train_neural_network({'epochs': epochs, 'batch_size': batch_size})

                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='weighted')
                recall = recall_score(self.y_test, y_pred, average='weighted')
                f1 = f1_score(self.y_test, y_pred, average='weighted')
                cm = confusion_matrix(self.y_test, y_pred)
                report = classification_report(self.y_test, y_pred, output_dict=True)

                results.append({
                    'Model': f'Neural Network (epochs={epochs}, batch_size={batch_size})',
                    'Best Params': {'epochs': epochs, 'batch_size': batch_size},
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1,
                    'Confusion Matrix': cm.tolist(),
                    # 'Classification Report': report
                })

        # Save results to CSV
        df_results = pd.DataFrame(results)
        df_results.to_csv(self.csv_results, index=False)
        return results
    