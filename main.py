

from ModelTrainer import ModelTrainer
"""
This code trains and evaluates multiple models on any datasets to get a simple initial results, the heart disease dataset is used as just an example.
The code performs the following steps:
1. Fetches the heart disease dataset from the UCI Machine Learning Repository.
2. Preprocesses the data by filling missing values and converting the target variable to binary.
3. Trains and evaluates multiple models using the ModelTrainer class.
4. Prints the results for each model, including the best parameters, accuracy, precision, recall, F1 score, and confusion matrix.
Example usage:
1. Import the ModelTrainer class from the ModelTrainer module.
2. Fetch the heart disease dataset using the fetch_ucirepo function.
3. Preprocess the data by filling missing values and converting the target variable to binary.
4. Train and evaluate models using the ModelTrainer class.
5. Print the results for each model.
Note: Make sure to have the necessary modules and packages installed before running the code.
"""

    
# Example usage:
# load sample data from web
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
# data = pd.read_csv(url, header=None)
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 
  
# metadata 
print(heart_disease.metadata) 
  
# variable information 
print(heart_disease.variables) 

# preprocess data
# true false to 1 0
# fill missing values
X = X.fillna(X.mean())
# convert target to binary
y = y['num']

# Then apply your lambda function
y = y.apply(lambda x: 1 if x > 0 else 0)

# train and evaluate models
# Convert column names to strings to satisfy scikit-learn's requirements
X.columns = X.columns.astype(str)
# Initialize the ModelTrainer class with the data and hyperparameters
# you can use the default hyperparameters or specify your own hyperparameters for each model
trainer = ModelTrainer(X, y, csv_results= 'model_comparison_results.csv'  ,
                       logistic_regression_params={'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']},)
results = trainer.train_and_evaluate()
for model_result in results:
    print(f"Results for {model_result['Model']}:")
    print(f"Best Params: {model_result['Best Params']}")
    print(f"Accuracy: {model_result['Accuracy']}")
    print(f"Precision: {model_result['Precision']}")
    print(f"Recall: {model_result['Recall']}")
    print(f"F1 Score: {model_result['F1 Score']}")
    print(f"Confusion Matrix:\n{model_result['Confusion Matrix']}")
    # print(f"Classification Report:\n{model_result['Classification Report']}")
    print("\n")
