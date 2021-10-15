"""
Regression is a statistical process for estimating the relationship among variables,
often to make predictions about some outcome
"""
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import models
import joblib
import numpy as np
import pandas as pd
import warnings
# "error", "ignore", "always", "default", "module" or "once"
warnings.filterwarnings('always')


# How long takes The program to run
start_time = time.time()
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

tr_features = models.all_X_train2
tr_labels = models.y_train
te_features = models.all_X_test2
te_labels = models.y_test

print("--------- Check Data Shapes--------")
print(tr_features.shape, tr_labels.shape, te_features.shape, te_labels.shape)
# print(tr_features.head())
# print(tr_labels.head())

# Logistic Regression with Hyperparameters tuning


def print_results(results):
    print('BEST PARAMETERS: {}\n'.format(results.best_params_))
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 2), round(std * 2, 2), params))


print("--------LOGISTIC REGRESSION-----------")

lr = LogisticRegression()
parameters = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

# Cross Validation : 5 Kfolds

cv = GridSearchCV(lr, parameters, cv=5)
cv.fit(tr_features, tr_labels.values.ravel())
print(print_results(cv))


# Use the trained model to make predictions
print("Use the trained model to make predictions")
y_pred = cv.predict(te_features)
print(y_pred)

# Evaluate the predictions of the model on the holdout test set
print("Evaluation : Logistic Regression")

print("F1 score:", metrics.f1_score(te_labels, y_pred,
                                    average='weighted', labels=np.unique(y_pred)))

print("AUC: ", roc_auc_score(te_labels, y_pred))
print("MSE: ", mean_squared_error(te_labels, y_pred))

print("   ")
print("   ")


# Check the best parameter

print("Check the best parameter and Save the Model as a Picked File")
print(cv.best_estimator_)


# Write out the picked model
joblib.dump(cv.best_estimator_, 'LR_model.pkl')


print("---------------------------------------------------------")
print("-- Running Time : %s seconds " % (time.time() - start_time))
print("---------------------------------------------------------")
