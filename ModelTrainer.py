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
from tensorflow.keras.layers import Dense, Input


class ModelTrainer:
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
    Default hyperparameters for each model are provided, but custom hyperparameters can be passed as arguments.
    the default hyperparameters are:
        Logistic Regression: {'C': [0.01, 0.1, 1, 10]}
        Decision Tree: {'max_depth': [5, 10, 15]}
        Random Forest: {'n_estimators': [50, 100, 150]}
        Support Vector Machine: {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        k-Nearest Neighbors: {'n_neighbors': [3, 5, 7]}
        Gradient Boosting: {'n_estimators': [50, 100, 150]}
        XGBoost: {'n_estimators': [50, 100, 150]}
        LightGBM: {'n_estimators': [50, 100, 150]}
    """
    def __init__(self, X, y, csv_results='model_comparison_results.csv',
                 logistic_regression_params=None,
                 decision_tree_params=None,
                 random_forest_params=None,
                 svm_params=None,
                 knn_params=None,
                 gradient_boosting_params=None,
                 xgboost_params=None,
                 lightgbm_params=None
                 ):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.models = {
            'Logistic Regression': (LogisticRegression(), logistic_regression_params or {'C': [0.01, 0.1, 1, 10]}),
            'Decision Tree': (DecisionTreeClassifier(), decision_tree_params or {'max_depth': [5, 10, 15]}),
            'Random Forest': (RandomForestClassifier(), random_forest_params or {'n_estimators': [50, 100, 150]}),
            'Support Vector Machine': (SVC(), svm_params or {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
            'k-Nearest Neighbors': (KNeighborsClassifier(), knn_params or {'n_neighbors': [3, 5, 7]}),
            'Naive Bayes': (GaussianNB(), {}),
            'Gradient Boosting': (GradientBoostingClassifier(), gradient_boosting_params or {'n_estimators': [50, 100, 150]}),
            'XGBoost': (xgb.XGBClassifier(), xgboost_params or {'n_estimators': [50, 100, 150]}),
            'LightGBM': (lgb.LGBMClassifier(), lightgbm_params or {'n_estimators': [50, 100, 150]})
        }
        self.csv_results = csv_results # File to save results
        
    def _build_and_train_neural_network(self, params):
        model = Sequential()
        model.add(Input(shape=(self.X_train.shape[1],)))
        model.add(Dense(params['dense_layer_1_units'], activation='relu'))
        model.add(Dense(params['dense_layer_2_units'], activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        history = model.fit(self.X_train, self.y_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
        predictions = (model.predict(self.X_test) > 0.5).astype("int32")
        y_pred = predictions.flatten()
        
        return y_pred

    # Train and evaluate all models
    def train_and_evaluate(self):
        results = []
        neural_net_params = {'epochs': [50, 100], 'batch_size': [10, 20], 'dense_layer_1_units': [32,64], 'dense_layer_2_units': [32,16]}
        
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
        for dense_layer_1_unit in neural_net_params['dense_layer_1_units']:
            for dense_layer_2_unit in neural_net_params['dense_layer_2_units']:
                print (f"Training Neural Network with dense_layer_1_unit={dense_layer_1_unit} and dense_layer_2_unit={dense_layer_2_unit}...")

                for epochs in neural_net_params['epochs']:
                    for batch_size in neural_net_params['batch_size']:
                        print(f"Training Neural Network with epochs={epochs} and batch_size={batch_size}...")
                        y_pred = self._build_and_train_neural_network({'epochs': epochs, 'batch_size': batch_size,
                                                                    'dense_layer_1_units': dense_layer_1_unit,
                                                                        'dense_layer_2_units': dense_layer_2_unit   } )

                        accuracy = accuracy_score(self.y_test, y_pred)
                        precision = precision_score(self.y_test, y_pred, average='weighted')
                        recall = recall_score(self.y_test, y_pred, average='weighted')
                        f1 = f1_score(self.y_test, y_pred, average='weighted')
                        cm = confusion_matrix(self.y_test, y_pred)
                        report = classification_report(self.y_test, y_pred, output_dict=True)

                        results.append({
                            'Model': f'Neural Network (epochs={epochs}, batch_size={batch_size})',
                            'Best Params': {'epochs': epochs, 'batch_size': batch_size, 'dense_layer_1_units': dense_layer_1_unit, 'dense_layer_2_units': dense_layer_2_unit},
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
    