import numpy as np
import pandas as pd

# help to choose between models
from sklearn.model_selection import train_test_split

#utilities for scaling, transforming, and wrangling data.
from sklearn import preprocessing

# model family
from sklearn.ensemble import RandomForestRegressor

# Tools to perform cross validation
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# Tools to evaluate model performance or eveluation metrics
from sklearn.metrics import mean_squared_error, r2_score

#Joblib is an alternative to Python's pickle package, and we'll use it because
# it's more efficient for storing large numpy arrays.
from sklearn.externals import joblib

# read data from remote to pandas dataframe
dataset_url = ('http://mlr.cs.umass.edu/ml/machine-learning-databases/' +
                'wine-quality/winequality-red.csv')
data = pd.read_csv(dataset_url, sep= ";")

# separate label from training features
y = data.quality
X = data.drop('quality', axis=1)

#Split data into train and test setsPython
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,
                                                    random_state=123,
                                                    stratify=y)

# Preprocessing steps
"""
Fit the transformer on the training set (saving the means and standard deviations)
Apply the transformer to the training set (scaling the training data)
Apply the transformer to the test set (using the same means and standard deviations)
"""

scaler = preprocessing.StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
#
# print X_train_scaled.mean(axis=0)
#
# print X_train_scaled.std(axis=0)

X_test_scaled = scaler.transform(X_test)

# print X_test_scaled.mean(axis=0)
#
# print X_test_scaled.std(axis=0)

# pipeline with preprocessing and model

pipeline = make_pipeline(preprocessing.StandardScaler(),
                         RandomForestRegressor(n_estimators=100))

# declare hyperparameters to tune
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

# sklearn cross-validation with pipeline
clf = GridSearchCV(pipeline, hyperparameters, cv=10)

# Fit and tune model
clf.fit(X_train, y_train)


# print clf.best_params_

#Predict a new set of dataPython
y_pred = clf.predict(X_test)

print r2_score(y_test, y_pred)

print mean_squared_error(y_test, y_pred)

# save model to pkl file (don't exactly know what this is)
joblib.dump(clf, 'rf_regressor.pkl')

# To load: clf2 = joblib.load('rf_regressor.pkl')
